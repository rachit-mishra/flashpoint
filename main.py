"""
FLASHPOINT — Geopolitical Intelligence Dashboard
FastAPI backend: serves /api/pulse, /api/history, /api/alerts/*
"""

import asyncio
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests as http_req
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from analyzer import analyze_batch, generate_situation_report, generate_country_brief, stream_scenario
from fetcher import get_all_news

load_dotenv()

app = FastAPI(title="Flashpoint API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ── Paths ───────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
# DB_PATH: use env var in production (e.g. Railway volume at /data), local fallback
DB_PATH  = Path(os.getenv("DB_PATH", str(BASE_DIR / "flashpoint.db")))

# ── In-memory cache ────────────────────────────────────────────────────────────
_cache: dict           = {}
_previous_tension: Optional[int] = None
_cache_lock            = asyncio.Lock()
CACHE_TTL_SECONDS      = 3600  # 1 hour

# ── Weight tables ──────────────────────────────────────────────────────────────
CATEGORY_WEIGHTS = {
    "Nuclear": 2.0, "Conflict": 1.5, "Terrorism": 1.5,
    "Economic": 0.9, "Diplomacy": 0.8,
}
SENTIMENT_WEIGHTS = {
    "Escalating": 1.3, "Uncertain": 1.1,
    "Neutral": 1.0, "De-escalating": 0.7,
}
MAX_RAW = 5.0 * 2.0 * 1.3  # 13.0

ALL_REGIONS = ["Europe", "Middle East", "Asia Pacific", "Africa", "Americas", "Conflict & War", "South Asia"]
REGION_DISPLAY = {"Conflict & War": "Global", "Asia Pacific": "Asia"}

# South Asia theatre — actors used for filtering
SOUTH_ASIA_ACTORS = {
    "india", "pakistan", "bangladesh", "sri lanka", "nepal", "bhutan", "maldives",
    "myanmar", "afghanistan", "china", "kashmir", "ladakh", "tibet",
    "modi", "raw", "isi", "loc", "lac", "bsf", "crpf",
}
SENTIMENT_ORDER = {"Escalating": 0, "Uncertain": 1, "Neutral": 2, "De-escalating": 3}


# ── SQLite helpers ──────────────────────────────────────────────────────────────

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS readings (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT    NOT NULL,
                score         INTEGER NOT NULL,
                label         TEXT    NOT NULL,
                total_events  INTEGER DEFAULT 0,
                regional_json TEXT    DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS subscriptions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                email        TEXT    NOT NULL UNIQUE,
                threshold    INTEGER NOT NULL DEFAULT 70,
                last_alerted TEXT,
                created_at   TEXT    NOT NULL
            );
        """)
        try:
            c.execute("ALTER TABLE readings ADD COLUMN regional_json TEXT DEFAULT '{}'")
        except Exception:
            pass   # column already exists on existing DBs
        c.commit()


def save_reading(score: int, label: str, total: int, regional: dict | None = None) -> None:
    import json as _json
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            "INSERT INTO readings (ts, score, label, total_events, regional_json) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), score, label, total,
             _json.dumps(regional or {})),
        )
        c.commit()


def get_history(limit: int = 120) -> list[dict]:
    """Return last `limit` readings oldest-first (for chart L→R time order)."""
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            "SELECT ts, score, label FROM readings ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"ts": r[0], "score": r[1], "label": r[2]} for r in reversed(rows)]


def get_regional_history(region: str, limit: int = 48) -> list[dict]:
    """Return last `limit` per-region readings oldest-first."""
    import json as _json
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            "SELECT ts, regional_json FROM readings ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    result = []
    for ts, rj in reversed(rows):
        try:
            d = _json.loads(rj or '{}')
            score = d.get(region)
            if score is not None:
                result.append({"ts": ts, "score": int(score)})
        except Exception:
            pass
    return result


def get_subscriptions() -> list[dict]:
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            "SELECT id, email, threshold, last_alerted FROM subscriptions"
        ).fetchall()
    return [{"id": r[0], "email": r[1], "threshold": r[2], "last_alerted": r[3]} for r in rows]


def upsert_subscription(email: str, threshold: int) -> None:
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            """INSERT INTO subscriptions (email, threshold, created_at)
               VALUES (?,?,?)
               ON CONFLICT(email) DO UPDATE SET threshold=excluded.threshold""",
            (email.lower().strip(), threshold, datetime.now(timezone.utc).isoformat()),
        )
        c.commit()


def delete_subscription(email: str) -> bool:
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute("DELETE FROM subscriptions WHERE email=?", (email.lower().strip(),))
        c.commit()
        return cur.rowcount > 0


def mark_alerted(sub_id: int) -> None:
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            "UPDATE subscriptions SET last_alerted=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), sub_id),
        )
        c.commit()


# ── Alert email (Resend API) ────────────────────────────────────────────────────

LEVEL_COLORS = {
    "Stable": "#22c55e", "Elevated": "#3b82f6",
    "High": "#f59e0b", "Severe": "#ef4444", "Critical": "#dc2626",
}

def _build_alert_html(score: int, label: str, threshold: int,
                      brief: str, flashpoints: list[dict], email: str) -> str:
    color   = LEVEL_COLORS.get(label, "#3b82f6")
    fp_rows = "".join(
        f"""<tr>
              <td style="padding:8px 0;border-bottom:1px solid #1e2a3a;color:#eef2ff;font-size:13px;">
                <b style="color:{LEVEL_COLORS.get('Severe','#ef4444')}">{fp['region']}</b>
                &nbsp;<span style="color:#4b5563;font-size:11px;">{' · '.join(fp.get('actors',[]))}</span><br>
                <span style="color:#8b95a8;font-size:12px;">{fp.get('insight','')}</span>
              </td>
            </tr>"""
        for fp in flashpoints[:3]
    )
    import base64
    token = base64.urlsafe_b64encode(email.encode()).decode()
    return f"""<!DOCTYPE html><html><body style="background:#080c14;color:#eef2ff;
font-family:'Helvetica Neue',Arial,sans-serif;margin:0;padding:32px 16px;">
<div style="max-width:520px;margin:0 auto;">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:28px;">
    <div style="width:38px;height:38px;background:linear-gradient(135deg,#ef4444,#b91c1c);
      border-radius:9px;display:flex;align-items:center;justify-content:center;
      font-size:20px;flex-shrink:0;">⚡</div>
    <div>
      <div style="font-size:17px;font-weight:700;letter-spacing:0.12em;">FLASHPOINT</div>
      <div style="font-size:9px;color:#4b5563;letter-spacing:0.22em;text-transform:uppercase;">
        Alert Triggered</div>
    </div>
  </div>

  <div style="background:#0d1220;border:1px solid {color}33;border-radius:12px;
    padding:24px;margin-bottom:16px;">
    <div style="font-size:9px;color:#4b5563;letter-spacing:0.22em;text-transform:uppercase;
      margin-bottom:10px;">Global Tension Index</div>
    <div style="font-size:72px;font-weight:700;color:{color};line-height:1;
      letter-spacing:-0.03em;">{score}</div>
    <div style="display:inline-block;background:{color}22;color:{color};
      border:1px solid {color}44;border-radius:999px;padding:4px 14px;
      font-size:11px;letter-spacing:0.14em;font-weight:700;margin-top:10px;">
      {label.upper()}</div>
    <div style="margin-top:12px;font-size:11px;color:#4b5563;">
      Your alert threshold: {threshold}/100</div>
  </div>

  <div style="background:#0d1220;border:1px solid rgba(255,255,255,0.06);
    border-radius:12px;padding:20px;margin-bottom:16px;">
    <div style="font-size:9px;color:#4b5563;letter-spacing:0.22em;text-transform:uppercase;
      margin-bottom:10px;">Signal Brief</div>
    <p style="font-size:13px;line-height:1.8;color:#8b95a8;margin:0;">{brief}</p>
  </div>

  {'<div style="background:#0d1220;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:20px;margin-bottom:16px;"><div style="font-size:9px;color:#4b5563;letter-spacing:0.22em;text-transform:uppercase;margin-bottom:12px;">Active Flashpoints</div><table style="width:100%;border-collapse:collapse;">' + fp_rows + '</table></div>' if fp_rows else ''}

  <div style="text-align:center;padding:16px 0;font-size:10px;color:#4b5563;">
    FLASHPOINT · Global Intelligence Monitor<br>
    <a href="{os.getenv('APP_BASE_URL', 'http://localhost:8000')}/api/alerts/unsubscribe?email={token}"
       style="color:#3b82f6;">Unsubscribe</a>
  </div>
</div></body></html>"""


def send_alert_email(email: str, threshold: int, score: int, label: str,
                     brief: str, flashpoints: list[dict]) -> bool:
    api_key = os.getenv("RESEND_API_KEY", "")
    if not api_key:
        print(f"[Flashpoint] Alert skipped (no RESEND_API_KEY): {email}")
        return False
    import base64 as _b64
    from_addr   = os.getenv("ALERT_FROM_EMAIL", "FLASHPOINT <alerts@flashpoint.io>")
    base_url    = os.getenv("APP_BASE_URL", "http://localhost:8000")
    token       = _b64.urlsafe_b64encode(email.encode()).decode()
    unsub_url   = f"{base_url}/api/alerts/unsubscribe?email={token}"
    plain_fps   = "\n".join(f"- {fp['region']}: {fp.get('insight','')}" for fp in flashpoints[:3])
    plain_text  = (
        f"FLASHPOINT Alert — Tension at {score}/100 ({label})\n\n"
        f"Your threshold: {threshold}/100\n\n"
        f"SIGNAL BRIEF\n{brief}\n\n"
        f"ACTIVE FLASHPOINTS\n{plain_fps}\n\n"
        f"View dashboard: {base_url}\n"
        f"Unsubscribe: {unsub_url}"
    )
    try:
        resp = http_req.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from":    from_addr,
                "to":      email,
                "subject": f"⚡ Tension at {score}/100 ({label}) — FLASHPOINT Alert",
                "html":    _build_alert_html(score, label, threshold, brief, flashpoints, email),
                "text":    plain_text,
                "headers": {
                    "List-Unsubscribe":      f"<{unsub_url}>",
                    "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
                },
            },
            timeout=10,
        )
        resp.raise_for_status()
        print(f"[Flashpoint] Alert sent → {email} (tension={score})")
        return True
    except Exception as e:
        print(f"[Flashpoint] Alert send error: {e}")
        return False


def check_and_send_alerts(score: int, label: str, brief: str, flashpoints: list[dict]) -> None:
    """Send alert emails to subscribers whose threshold has been crossed."""
    now_ts = datetime.now(timezone.utc)
    for sub in get_subscriptions():
        if score < sub["threshold"]:
            continue
        # Rate-limit: skip if alerted within the last 4 hours
        if sub["last_alerted"]:
            last = datetime.fromisoformat(sub["last_alerted"])
            if (now_ts - last).total_seconds() < 4 * 3600:
                continue
        if send_alert_email(sub["email"], sub["threshold"], score, label, brief, flashpoints):
            mark_alerted(sub["id"])


# ── Daily digest ────────────────────────────────────────────────────────────────

def _build_digest_html(score: int, label: str, brief: str,
                       flashpoints: list[dict], regional: list[dict],
                       date_str: str) -> str:
    color = LEVEL_COLORS.get(label, "#3b82f6")
    fp_rows = "".join(
        f"""<tr>
              <td style="padding:10px 0;border-bottom:1px solid #1e2a3a;">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                  <span style="color:{LEVEL_COLORS.get('Severe','#ef4444')};
                    font-weight:700;font-size:13px;">{fp['region']}</span>
                  <span style="background:{LEVEL_COLORS.get(fp.get('sentiment','Neutral'),'#4b5563')}22;
                    color:{LEVEL_COLORS.get(fp.get('sentiment','Neutral'),'#4b5563')};
                    border:1px solid {LEVEL_COLORS.get(fp.get('sentiment','Neutral'),'#4b5563')}44;
                    border-radius:999px;padding:2px 8px;font-size:10px;
                    letter-spacing:0.1em;">{fp.get('sentiment','').upper()}</span>
                </div>
                <div style="color:#8b95a8;font-size:12px;line-height:1.6;">
                  {fp.get('insight','')}
                </div>
                {'<a href="' + fp.get("url","#") + '" style="color:#3b82f6;font-size:11px;text-decoration:none;">→ Read source</a>' if fp.get("url","#") != "#" else ""}
              </td>
            </tr>"""
        for fp in flashpoints[:5]
    )
    reg_rows = "".join(
        f"""<tr>
              <td style="padding:6px 0;border-bottom:1px solid #1e2a3a;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                  <span style="color:#eef2ff;font-size:13px;">{r['region']}</span>
                  <div style="display:flex;align-items:center;gap:8px;">
                    <div style="width:80px;height:4px;background:#1e2a3a;border-radius:2px;">
                      <div style="width:{r['score']}%;height:100%;background:{LEVEL_COLORS.get(r['label'],'#3b82f6')};border-radius:2px;"></div>
                    </div>
                    <span style="color:{LEVEL_COLORS.get(r['label'],'#3b82f6')};
                      font-weight:700;font-size:13px;width:28px;text-align:right;">{r['score']}</span>
                  </div>
                </div>
              </td>
            </tr>"""
        for r in regional[:6]
    )
    base_url = os.getenv("APP_BASE_URL", "http://localhost:8000")
    return f"""<!DOCTYPE html><html><body style="background:#080c14;color:#eef2ff;
font-family:'Helvetica Neue',Arial,sans-serif;margin:0;padding:32px 16px;">
<div style="max-width:560px;margin:0 auto;">

  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
    <div style="width:38px;height:38px;background:linear-gradient(135deg,#ef4444,#b91c1c);
      border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:20px;">⚡</div>
    <div>
      <div style="font-size:17px;font-weight:700;letter-spacing:0.12em;">FLASHPOINT</div>
      <div style="font-size:9px;color:#4b5563;letter-spacing:0.22em;text-transform:uppercase;">
        Daily Intelligence Digest · {date_str}</div>
    </div>
  </div>

  <div style="height:1px;background:linear-gradient(90deg,#ef4444,transparent);
    margin-bottom:28px;"></div>

  <div style="background:#0d1220;border:1px solid {color}33;border-radius:12px;
    padding:24px;margin-bottom:16px;display:flex;align-items:center;gap:24px;">
    <div>
      <div style="font-size:9px;color:#4b5563;letter-spacing:0.22em;
        text-transform:uppercase;margin-bottom:6px;">Global Tension</div>
      <div style="font-size:64px;font-weight:700;color:{color};line-height:1;
        letter-spacing:-0.03em;">{score}</div>
      <div style="display:inline-block;background:{color}22;color:{color};
        border:1px solid {color}44;border-radius:999px;padding:3px 12px;
        font-size:11px;letter-spacing:0.14em;font-weight:700;margin-top:8px;">
        {label.upper()}</div>
    </div>
  </div>

  <div style="background:#0d1220;border:1px solid rgba(255,255,255,0.06);
    border-radius:12px;padding:20px;margin-bottom:16px;">
    <div style="font-size:9px;color:#4b5563;letter-spacing:0.22em;
      text-transform:uppercase;margin-bottom:10px;">Today's Signal Brief</div>
    <p style="font-size:13px;line-height:1.85;color:#8b95a8;margin:0;">{brief}</p>
  </div>

  {'<div style="background:#0d1220;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:20px;margin-bottom:16px;"><div style="font-size:9px;color:#4b5563;letter-spacing:0.22em;text-transform:uppercase;margin-bottom:12px;">Active Flashpoints</div><table style="width:100%;border-collapse:collapse;">' + fp_rows + '</table></div>' if fp_rows else ''}

  {'<div style="background:#0d1220;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:20px;margin-bottom:16px;"><div style="font-size:9px;color:#4b5563;letter-spacing:0.22em;text-transform:uppercase;margin-bottom:12px;">Regional Pulse</div><table style="width:100%;border-collapse:collapse;">' + reg_rows + '</table></div>' if reg_rows else ''}

  <div style="text-align:center;padding:20px 0 8px;">
    <a href="{base_url}" style="display:inline-block;background:#ef4444;color:#fff;
      text-decoration:none;padding:10px 28px;border-radius:8px;font-size:13px;
      font-weight:600;letter-spacing:0.06em;">VIEW LIVE DASHBOARD →</a>
  </div>

  <div style="text-align:center;padding:16px 0;font-size:10px;color:#4b5563;">
    FLASHPOINT · Global Intelligence Monitor<br>
    You're receiving this because you subscribed to daily digests.<br>
    <a href="{base_url}/api/alerts/unsubscribe?email={{token}}"
       style="color:#3b82f6;">Unsubscribe</a>
  </div>
</div></body></html>"""


def send_daily_digest(score: int, label: str, brief: str,
                      flashpoints: list[dict], regional: list[dict]) -> None:
    """Send the daily digest to all subscribers."""
    api_key = os.getenv("RESEND_API_KEY", "")
    if not api_key:
        print("[Flashpoint] Daily digest skipped (no RESEND_API_KEY)")
        return
    from_addr = os.getenv("ALERT_FROM_EMAIL", "FLASHPOINT <alerts@flashpoint.watch>")
    date_str  = datetime.now(timezone.utc).strftime("%B %d, %Y")
    subs      = get_subscriptions()
    if not subs:
        print("[Flashpoint] Daily digest: no subscribers")
        return
    import base64 as _b64
    base_url = os.getenv("APP_BASE_URL", "http://localhost:8000")
    sent = 0
    for sub in subs:
        token     = _b64.urlsafe_b64encode(sub["email"].encode()).decode()
        unsub_url = f"{base_url}/api/alerts/unsubscribe?email={token}"
        html      = _build_digest_html(score, label, brief, flashpoints, regional, date_str)
        html      = html.replace("{token}", token)
        fp_lines  = "\n".join(f"- {fp['region']}: {fp.get('insight','')}" for fp in flashpoints[:5])
        reg_lines = "\n".join(f"- {r['region']}: {r['score']}/100 ({r['label']})" for r in regional[:6])
        plain_text = (
            f"FLASHPOINT Daily Intelligence Digest — {date_str}\n\n"
            f"GLOBAL TENSION: {score}/100 ({label})\n\n"
            f"TODAY'S SIGNAL BRIEF\n{brief}\n\n"
            f"ACTIVE FLASHPOINTS\n{fp_lines}\n\n"
            f"REGIONAL PULSE\n{reg_lines}\n\n"
            f"View live dashboard: {base_url}\n"
            f"Unsubscribe: {unsub_url}"
        )
        try:
            resp = http_req.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={
                    "from":    from_addr,
                    "to":      sub["email"],
                    "subject": f"⚡ FLASHPOINT Daily Digest · {date_str} · Tension {score}/100",
                    "html":    html,
                    "text":    plain_text,
                    "headers": {
                        "List-Unsubscribe":      f"<{unsub_url}>",
                        "List-Unsubscribe-Post": "List-Unsubscribe=One-Click",
                    },
                },
                timeout=10,
            )
            resp.raise_for_status()
            sent += 1
        except Exception as e:
            print(f"[Flashpoint] Digest send error ({sub['email']}): {e}")
    print(f"[Flashpoint] Daily digest sent to {sent}/{len(subs)} subscribers")


# ── Metric helpers ──────────────────────────────────────────────────────────────

def _weighted_score(a: dict) -> float:
    return (a.get("severity", 1)
            * CATEGORY_WEIGHTS.get(a.get("category", "Other"), 1.0)
            * SENTIMENT_WEIGHTS.get(a.get("sentiment", "Neutral"), 1.0))


def _to_100(avg: float) -> int:
    # Linear scaling against the theoretical MAX_RAW (5 × Nuclear × Escalating = 13.0)
    # compresses real-world data into a narrow 30-60 band.  A mild power curve
    # (exponent < 1) redistributes the range so "genuinely heated" conditions
    # register in the 65-80 zone while still anchoring 0=0 and max=100.
    raw_pct = avg / MAX_RAW          # 0.0 → 1.0
    curved  = raw_pct ** 0.55        # concave up: amplifies mid-range values
    return max(0, min(100, round(curved * 100)))


def _label(score: int) -> str:
    if score > 85: return "Critical"
    if score > 70: return "Severe"
    if score > 50: return "High"
    if score > 30: return "Elevated"
    return "Stable"


def _is_analyzed(a: dict) -> bool:
    insight = a.get("insight", "")
    return bool(insight) and insight != "Analysis unavailable."


# ── Metric computation ──────────────────────────────────────────────────────────

def compute_tension_index(articles: list[dict]) -> tuple[int, str]:
    pool = [a for a in articles if _is_analyzed(a)] or articles
    if not pool:
        return 0, "Stable"
    scores = [_weighted_score(a) for a in pool]
    idx = _to_100(sum(scores) / len(scores))
    return idx, _label(idx)


def compute_regional_pulse(articles: list[dict]) -> list[dict]:
    buckets: dict[str, list] = {r: [] for r in ALL_REGIONS}
    for a in articles:
        if _is_analyzed(a) and a.get("region", "") in buckets:
            buckets[a["region"]].append(a)

    results = []
    for region, arts in buckets.items():
        score = 0
        if arts:
            score = _to_100(sum(_weighted_score(a) for a in arts) / len(arts))

        # Top 3 contributing articles for the tooltip
        top = sorted(arts, key=lambda x: -_weighted_score(x))[:3]
        sources = sorted({a.get("source", "") for a in arts if a.get("source")})

        results.append({
            "region":   REGION_DISPLAY.get(region, region),
            "score":    score,
            "label":    _label(score),
            "count":    len(arts),
            "sources":  sources,
            "articles": [
                {"title": a.get("title", ""), "url": a.get("url", "#")}
                for a in top
            ],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def compute_flashpoints(articles: list[dict]) -> list[dict]:
    candidates = [a for a in articles if _is_analyzed(a) and a.get("severity", 0) >= 2]
    candidates.sort(key=lambda x: (
        -x.get("severity", 0),
        SENTIMENT_ORDER.get(x.get("sentiment", "Neutral"), 2),
    ))
    seen, out = set(), []
    for a in candidates:
        r = a.get("region", "Unknown")
        if r not in seen:
            seen.add(r)
            out.append({
                "region":    REGION_DISPLAY.get(r, r),
                "actors":    a.get("actors", []),
                "insight":   a.get("insight", ""),
                "severity":  a.get("severity", 1),
                "sentiment": a.get("sentiment", "Neutral"),
                "category":  a.get("category", "Other"),
                "url":       a.get("url", "#"),
                "source":    a.get("source", ""),
            })
        if len(out) == 6:
            break
    return out


def compute_sentiment_breakdown(articles: list[dict]) -> dict:
    pool = [a for a in articles if _is_analyzed(a)]
    counts = {"Escalating": 0, "Uncertain": 0, "Neutral": 0, "De-escalating": 0}
    for a in pool:
        s = a.get("sentiment", "Neutral")
        if s in counts:
            counts[s] += 1
    return counts


def compute_category_breakdown(articles: list[dict]) -> dict:
    pool = [a for a in articles if _is_analyzed(a)]
    counts = {"Nuclear": 0, "Conflict": 0, "Terrorism": 0, "Economic": 0, "Diplomacy": 0}
    for a in pool:
        cat = a.get("category", "")
        if cat in counts:
            counts[cat] += 1
    return counts


def compute_category_articles(articles: list[dict]) -> dict:
    """Top 3 articles per category, sorted by severity, for the breakdown tooltip."""
    cats = list(CATEGORY_WEIGHTS.keys())  # Nuclear, Conflict, Terrorism, Economic, Diplomacy
    buckets: dict[str, list] = {cat: [] for cat in cats}
    for a in articles:
        if _is_analyzed(a):
            cat = a.get("category", "")
            if cat in buckets:
                buckets[cat].append(a)
    result = {}
    for cat, arts in buckets.items():
        top = sorted(arts, key=lambda x: -x.get("severity", 0))[:3]
        result[cat] = [{"title": a.get("title", ""), "url": a.get("url", "#")} for a in top]
    return result


# ── Data refresh ────────────────────────────────────────────────────────────────

async def refresh_data() -> dict:
    global _previous_tension

    loop     = asyncio.get_event_loop()
    articles = await loop.run_in_executor(None, get_all_news)
    enriched = await loop.run_in_executor(None, analyze_batch, articles, 20)
    brief_full = await loop.run_in_executor(
        None, generate_situation_report, enriched[:12]
    )

    tension_idx, tension_label = compute_tension_index(enriched)

    trend = "—"
    if _previous_tension is not None:
        diff  = tension_idx - _previous_tension
        trend = f"+{diff}" if diff >= 0 else str(diff)
    _previous_tension = tension_idx

    # Strip markdown → first real prose paragraph
    brief = ""
    if brief_full:
        clean = re.sub(r'^#{1,6}\s+',        '', brief_full, flags=re.MULTILINE)
        clean = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', clean)
        clean = re.sub(r'^[-*_]{3,}\s*$',    '', clean,     flags=re.MULTILINE)
        clean = re.sub(r'^\s*[-*+]\s+',      '', clean,     flags=re.MULTILINE)
        paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip() and len(p.strip()) > 40]
        prose = [p for p in paragraphs if "." in p]
        brief = prose[0] if prose else (paragraphs[0] if paragraphs else clean.strip()[:600])

    flashpoints        = compute_flashpoints(enriched)
    regional_pulse_data = compute_regional_pulse(enriched)
    regional_scores    = {r["region"]: r["score"] for r in regional_pulse_data}

    # Persist reading and check alerts (non-blocking — fire and forget)
    loop.run_in_executor(None, save_reading, tension_idx, tension_label, len(enriched), regional_scores)
    loop.run_in_executor(None, check_and_send_alerts, tension_idx, tension_label, brief, flashpoints)

    # Article feed — analyzed articles sorted by severity, top 30
    feed = sorted(
        [a for a in enriched if _is_analyzed(a)],
        key=lambda x: (-x.get("severity", 0),
                       SENTIMENT_ORDER.get(x.get("sentiment", "Neutral"), 2))
    )[:30]

    return {
        "tension_index":       tension_idx,
        "tension_label":       tension_label,
        "tension_trend":       trend,
        "flashpoints":         flashpoints,
        "regional_pulse":      regional_pulse_data,
        "sentiment_breakdown": compute_sentiment_breakdown(enriched),
        "category_breakdown":  compute_category_breakdown(enriched),
        "category_articles":   compute_category_articles(enriched),
        "brief":               brief,
        "last_updated":        datetime.now(timezone.utc).isoformat(),
        "total_events":        len(enriched),
        "articles": [
            {
                "title":     a.get("title", ""),
                "url":       a.get("url", "#"),
                "source":    a.get("source", ""),
                "region":    REGION_DISPLAY.get(a.get("region", ""), a.get("region", "")),
                "category":  a.get("category", "Other"),
                "severity":  a.get("severity", 1),
                "sentiment": a.get("sentiment", "Neutral"),
                "actors":    a.get("actors", []),
                "insight":   a.get("insight", ""),
                "published": a.get("published", ""),
            }
            for a in feed
        ],
    }


# ── Background cache refresh loop ───────────────────────────────────────────────

async def _background_refresh():
    while True:
        async with _cache_lock:
            try:
                _cache["data"] = await refresh_data()
                print(
                    f"[Flashpoint] Refreshed at "
                    f"{datetime.now(timezone.utc).strftime('%H:%M UTC')} "
                    f"— tension: {_cache['data']['tension_index']}"
                )
            except Exception as exc:
                print(f"[Flashpoint] Refresh error: {exc}")
        await asyncio.sleep(CACHE_TTL_SECONDS)


async def _daily_digest_loop():
    """Fire the daily digest at 07:00 UTC every day."""
    DIGEST_HOUR = 7  # UTC
    while True:
        now   = datetime.now(timezone.utc)
        next_ = now.replace(hour=DIGEST_HOUR, minute=0, second=0, microsecond=0)
        if now >= next_:
            next_ = next_.replace(day=next_.day + 1)
        wait_secs = (next_ - now).total_seconds()
        print(f"[Flashpoint] Daily digest scheduled in {wait_secs/3600:.1f}h "
              f"(next run: {next_.strftime('%Y-%m-%d %H:%M UTC')})")
        await asyncio.sleep(wait_secs)
        try:
            # Use cached data if fresh, else fetch
            data = _cache.get("data") or {}
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                None, send_daily_digest,
                data.get("tension_index", 0),
                data.get("tension_label", "Unknown"),
                data.get("brief", ""),
                data.get("flashpoints", []),
                data.get("regional_pulse", []),
            )
        except Exception as exc:
            print(f"[Flashpoint] Daily digest error: {exc}")


@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(_background_refresh())
    asyncio.create_task(_daily_digest_loop())


# ── Routes ───────────────────────────────────────────────────────────────────────

@app.get("/bimi-logo.svg")
async def serve_bimi_logo():
    """BIMI brand logo — square SVG required for Gmail sender avatar."""
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" rx="22" fill="#b91c1c"/>
  <text x="50" y="68" font-size="58" text-anchor="middle" font-family="Arial,sans-serif">⚡</text>
</svg>"""
    from fastapi.responses import Response
    return Response(content=svg, media_type="image/svg+xml",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTMLResponse(
        content=(BASE_DIR / "index.html").read_text(encoding="utf-8"), status_code=200
    )


@app.get("/intel", response_class=HTMLResponse)
async def serve_intel():
    return HTMLResponse(
        content=(BASE_DIR / "intel.html").read_text(encoding="utf-8"), status_code=200
    )


@app.get("/share", response_class=HTMLResponse)
async def serve_share(name: str = ""):
    """Serve shareable intel card with server-injected OG tags for link previews."""
    base_url = os.getenv("APP_BASE_URL", "http://localhost:8000")
    template = (BASE_DIR / "share.html").read_text(encoding="utf-8")

    if not name:
        return HTMLResponse(content=template, status_code=200)

    # Build OG metadata from cached data (no API call — instant)
    data     = _cache.get("data", {})
    articles = data.get("articles", [])
    name_lower = name.lower()
    matching = [
        a for a in articles
        if name_lower in a.get("title",   "").lower()
        or any(name_lower in act.lower() for act in a.get("actors", []))
        or name_lower in a.get("region",  "").lower()
        or name_lower in a.get("insight", "").lower()
    ]

    if matching:
        avg_sev  = sum(a.get("severity", 1) for a in matching) / len(matching)
        sev_round = round(avg_sev)
        sents    = [a.get("sentiment", "Neutral") for a in matching]
        cats     = [a.get("category", "Other") for a in matching]
        dominant_sent = max(set(sents), key=sents.count)
        dominant_cat  = max(set(cats), key=cats.count)
    else:
        avg_sev, sev_round = 0, 0
        dominant_sent, dominant_cat = "Neutral", "Other"

    og_title = f"{name} — {dominant_sent} · {dominant_cat} | FLASHPOINT"
    og_desc  = f"Severity {avg_sev:.1f}/5 across {len(matching)} articles. {dominant_sent} · {dominant_cat}."
    if matching and matching[0].get("insight"):
        og_desc = matching[0]["insight"]
    og_image = f"{base_url}/og-card.png?name={name}&sev={sev_round}&sent={dominant_sent}"
    og_url   = f"{base_url}/share?name={name}"

    html = (template
        .replace("{{OG_TITLE}}", og_title)
        .replace("{{OG_DESC}}",  og_desc)
        .replace("{{OG_IMAGE}}", og_image)
        .replace("{{OG_URL}}",   og_url))

    return HTMLResponse(content=html, status_code=200)


@app.get("/og-card.png")
async def serve_og_card(name: str = "FLASHPOINT", sev: int = 0, sent: str = "Neutral"):
    """Dynamic PNG used as og:image — renders as a visual card in link previews."""
    import io
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1200, 630
    BG     = (8, 12, 20)
    BAR_BG = (13, 18, 32)
    WHITE  = (238, 242, 255)
    MUTED  = (75, 85, 99)

    sev_colors = {
        0: (75,85,99), 1: (34,197,94), 2: (59,130,246),
        3: (245,158,11), 4: (239,68,68), 5: (220,38,38),
    }
    sent_colors = {
        'Escalating': (239,68,68), 'De-escalating': (34,197,94),
        'Uncertain': (245,158,11), 'Neutral': (59,130,246),
    }
    sc  = sev_colors.get(sev, (75,85,99))
    stc = sent_colors.get(sent, (59,130,246))

    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Use DejaVu (available on most Linux systems incl Railway)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        logo_font  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        sev_font   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        title_font = label_font = small_font = logo_font = sev_font = ImageFont.load_default()

    # Top accent bar
    draw.rectangle([0, 0, W, 5], fill=sc)

    # Logo area
    draw.rounded_rectangle([60, 44, 100, 84], radius=8, fill=(185, 28, 28))
    draw.text((112, 52), "FLASHPOINT", fill=WHITE, font=logo_font)
    draw.text((310, 58), "INTELLIGENCE CARD", fill=MUTED, font=small_font)

    # Divider
    draw.line([60, 110, 1140, 110], fill=(30, 42, 58), width=1)

    # Severity ring
    cx, cy, r = 120, 240, 56
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=sc, width=4)
    sev_text = f"{sev}.0"
    bb = draw.textbbox((0, 0), sev_text, font=sev_font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    draw.text((cx - tw//2, cy - th//2 - 4), sev_text, fill=sc, font=sev_font)

    # Actor name
    draw.text((210, 200), name, fill=WHITE, font=title_font)

    # Sentiment badge
    badge_text = sent.upper()
    bb = draw.textbbox((0, 0), badge_text, font=small_font)
    bw = bb[2] - bb[0] + 28
    bx, by = 210, 265
    badge_bg = (stc[0]//8, stc[1]//8, stc[2]//8)
    draw.rounded_rectangle([bx, by, bx+bw, by+26], radius=13, fill=badge_bg, outline=stc)
    draw.text((bx + 14, by + 5), badge_text, fill=stc, font=small_font)

    # Severity dots
    draw.text((60, 350), "SEVERITY", fill=MUTED, font=small_font)
    dot_x = 60
    for i in range(5):
        color = sc if i < sev else (40, 50, 65)
        draw.ellipse([dot_x, 380, dot_x+24, 404], fill=color)
        dot_x += 34

    # Bottom bar
    draw.rectangle([0, 540, W, H], fill=BAR_BG)
    draw.text((60, 575), f"flashpoint.watch/share?name={name}", fill=MUTED, font=small_font)
    rt = "REAL-TIME GEOPOLITICAL INTELLIGENCE"
    rb = draw.textbbox((0, 0), rt, font=small_font)
    draw.text((1140 - (rb[2]-rb[0]), 575), rt, fill=MUTED, font=small_font)

    # Export PNG
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)

    from fastapi.responses import Response
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=300"})


@app.get("/scenario")
async def serve_scenario():
    """Serve the Scenario Engine page."""
    with open(BASE_DIR / "scenario.html") as f:
        return HTMLResponse(f.read())


@app.get("/api/scenario/stream")
async def scenario_stream(actor: str, trigger: str):
    """SSE endpoint — streams scenario analysis for the given actor + trigger."""
    import json as _json
    import queue
    import threading
    from fastapi.responses import StreamingResponse as _StreamingResponse

    data         = _cache.get("data", {})
    all_articles = data.get("articles", [])
    actors_data  = data.get("actors", [])

    actor_lower = actor.lower()
    matching = [
        a for a in all_articles
        if actor_lower in [x.lower() for x in a.get("actors", [])]
        or actor_lower in a.get("title",  "").lower()
        or actor_lower in a.get("region", "").lower()
    ]
    actor_info = next(
        (a for a in actors_data if a.get("name", "").lower() == actor_lower), {}
    )
    severity  = actor_info.get("severity",  3)
    sentiment = actor_info.get("sentiment", "Uncertain")

    chunk_queue: queue.Queue = queue.Queue()

    def producer():
        try:
            for chunk in stream_scenario(actor, trigger, matching, severity, sentiment):
                chunk_queue.put(chunk)
        except Exception as e:
            chunk_queue.put({"error": str(e)})
        finally:
            chunk_queue.put(None)  # sentinel

    threading.Thread(target=producer, daemon=True).start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, chunk_queue.get)
            if item is None:
                yield "data: [DONE]\n\n"
                break
            if isinstance(item, dict) and "error" in item:
                yield f"data: {_json.dumps(item)}\n\n"
                yield "data: [DONE]\n\n"
                break
            yield f"data: {_json.dumps({'chunk': item})}\n\n"

    return _StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/pulse")
async def get_pulse():
    if "data" not in _cache:
        async with _cache_lock:
            if "data" not in _cache:
                _cache["data"] = await refresh_data()
    return JSONResponse(content=_cache["data"])


@app.get("/api/history")
async def get_history_endpoint(limit: int = 120):
    loop = asyncio.get_event_loop()
    readings = await loop.run_in_executor(None, get_history, min(limit, 500))
    return JSONResponse(content=readings)


@app.get("/api/history/regional")
async def get_regional_history_endpoint(region: str = "Middle East", limit: int = 48):
    loop = asyncio.get_event_loop()
    readings = await loop.run_in_executor(None, get_regional_history, region, min(limit, 200))
    return JSONResponse(content=readings)


# ── Alert subscription ───────────────────────────────────────────────────────────

class SubscribeRequest(BaseModel):
    email:     str
    threshold: int = 70


@app.post("/api/alerts/subscribe")
async def subscribe(req: SubscribeRequest):
    email = req.email.lower().strip()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(status_code=422, detail="Invalid email address.")
    if not (30 <= req.threshold <= 95):
        raise HTTPException(status_code=422, detail="Threshold must be between 30 and 95.")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, upsert_subscription, email, req.threshold)
    return JSONResponse(content={
        "ok":      True,
        "message": f"Subscribed! You'll be alerted when tension exceeds {req.threshold}/100.",
    })


@app.get("/api/country")
async def get_country_intel(name: str):
    """Return filtered articles + Claude summary for a country/actor."""
    data      = _cache.get("data", {})
    articles  = data.get("articles", [])

    name_lower = name.lower()
    matching = [
        a for a in articles
        if name_lower in a.get("title",   "").lower()
        or any(name_lower in act.lower() for act in a.get("actors", []))
        or name_lower in a.get("region",  "").lower()
        or name_lower in a.get("insight", "").lower()
    ]

    if matching:
        severities = [a.get("severity", 1)    for a in matching]
        sentiments = [a.get("sentiment", "Neutral") for a in matching]
        categories = [a.get("category",  "Other")   for a in matching]
        avg_severity       = sum(severities) / len(severities)
        dominant_sentiment = max(set(sentiments), key=sentiments.count)
        dominant_category  = max(set(categories), key=categories.count)
    else:
        avg_severity       = 0.0
        dominant_sentiment = "Neutral"
        dominant_category  = "Other"

    loop    = asyncio.get_event_loop()
    summary = await loop.run_in_executor(None, generate_country_brief, name, matching[:8])

    return JSONResponse(content={
        "name":               name,
        "article_count":      len(matching),
        "avg_severity":       round(avg_severity, 1),
        "dominant_sentiment": dominant_sentiment,
        "dominant_category":  dominant_category,
        "summary":            summary,
        "articles":           matching[:10],
    })


@app.get("/api/alerts/unsubscribe")
async def unsubscribe(email: str):
    """Decode base64 email token from unsubscribe link in alert email."""
    import base64
    try:
        decoded = base64.urlsafe_b64decode(email.encode()).decode()
    except Exception:
        decoded = email  # fallback: treat as plain email
    loop = asyncio.get_event_loop()
    removed = await loop.run_in_executor(None, delete_subscription, decoded)
    msg = "Unsubscribed successfully." if removed else "Email not found."
    return HTMLResponse(content=f"""<!DOCTYPE html><html><body
        style="background:#080c14;color:#eef2ff;font-family:Arial,sans-serif;
               display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
        <div style="text-align:center;">
          <div style="font-size:32px;margin-bottom:12px;">⚡</div>
          <div style="font-size:20px;font-weight:700;letter-spacing:0.1em;">FLASHPOINT</div>
          <p style="color:#8b95a8;margin-top:12px;">{msg}</p>
          <a href="/" style="color:#3b82f6;font-size:13px;">← Back to dashboard</a>
        </div></body></html>""")
