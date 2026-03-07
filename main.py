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

from analyzer import analyze_batch, generate_situation_report
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
CACHE_TTL_SECONDS      = 300  # 5 minutes

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

ALL_REGIONS = ["Europe", "Middle East", "Asia Pacific", "Africa", "Americas", "Conflict & War"]
REGION_DISPLAY = {"Conflict & War": "Global", "Asia Pacific": "Asia"}
SENTIMENT_ORDER = {"Escalating": 0, "Uncertain": 1, "Neutral": 2, "De-escalating": 3}


# ── SQLite helpers ──────────────────────────────────────────────────────────────

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS readings (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           TEXT    NOT NULL,
                score        INTEGER NOT NULL,
                label        TEXT    NOT NULL,
                total_events INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS subscriptions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                email        TEXT    NOT NULL UNIQUE,
                threshold    INTEGER NOT NULL DEFAULT 70,
                last_alerted TEXT,
                created_at   TEXT    NOT NULL
            );
        """)
        c.commit()


def save_reading(score: int, label: str, total: int) -> None:
    with sqlite3.connect(DB_PATH) as c:
        c.execute(
            "INSERT INTO readings (ts, score, label, total_events) VALUES (?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), score, label, total),
        )
        c.commit()


def get_history(limit: int = 120) -> list[dict]:
    """Return last `limit` readings oldest-first (for chart L→R time order)."""
    with sqlite3.connect(DB_PATH) as c:
        rows = c.execute(
            "SELECT ts, score, label FROM readings ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{"ts": r[0], "score": r[1], "label": r[2]} for r in reversed(rows)]


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
    <a href="http://localhost:8000/api/alerts/unsubscribe?email={token}"
       style="color:#3b82f6;">Unsubscribe</a>
  </div>
</div></body></html>"""


def send_alert_email(email: str, threshold: int, score: int, label: str,
                     brief: str, flashpoints: list[dict]) -> bool:
    api_key = os.getenv("RESEND_API_KEY", "")
    if not api_key:
        print(f"[Flashpoint] Alert skipped (no RESEND_API_KEY): {email}")
        return False
    from_addr = os.getenv("ALERT_FROM_EMAIL", "FLASHPOINT <alerts@flashpoint.io>")
    try:
        resp = http_req.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "from":    from_addr,
                "to":      email,
                "subject": f"⚡ Tension at {score}/100 ({label}) — FLASHPOINT Alert",
                "html":    _build_alert_html(score, label, threshold, brief, flashpoints, email),
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


# ── Metric helpers ──────────────────────────────────────────────────────────────

def _weighted_score(a: dict) -> float:
    return (a.get("severity", 1)
            * CATEGORY_WEIGHTS.get(a.get("category", "Other"), 1.0)
            * SENTIMENT_WEIGHTS.get(a.get("sentiment", "Neutral"), 1.0))


def _to_100(avg: float) -> int:
    return max(0, min(100, round((avg / MAX_RAW) * 100)))


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

    flashpoints = compute_flashpoints(enriched)

    # Persist reading and check alerts (non-blocking — fire and forget)
    loop.run_in_executor(None, save_reading, tension_idx, tension_label, len(enriched))
    loop.run_in_executor(None, check_and_send_alerts, tension_idx, tension_label, brief, flashpoints)

    return {
        "tension_index":       tension_idx,
        "tension_label":       tension_label,
        "tension_trend":       trend,
        "flashpoints":         flashpoints,
        "regional_pulse":      compute_regional_pulse(enriched),
        "sentiment_breakdown": compute_sentiment_breakdown(enriched),
        "brief":               brief,
        "last_updated":        datetime.now(timezone.utc).isoformat(),
        "total_events":        len(enriched),
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


@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(_background_refresh())


# ── Routes ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTMLResponse(
        content=(BASE_DIR / "index.html").read_text(encoding="utf-8"), status_code=200
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
