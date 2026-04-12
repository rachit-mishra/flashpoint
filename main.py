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
from sarvam import translate_to_hindi, text_to_speech
from receipts_data import RECEIPTS_SEED

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
CACHE_TTL_SECONDS      = 21600  # 6 hours (4 refreshes/day)

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

# South Asia theatre — actors and keywords for filtering
# Core SA countries — any mention = South Asia relevant
SOUTH_ASIA_CORE = {
    "india", "pakistan", "bangladesh", "sri lanka", "nepal", "bhutan", "maldives",
    "afghanistan", "kashmir", "ladakh",
}
# Keywords that only count as SA when paired with a core country or region=South Asia
SOUTH_ASIA_CONTEXT = {
    "china", "myanmar", "tibet", "modi", "isi", "bsf", "crpf",
}
# Specific SA terms — always count
SOUTH_ASIA_SPECIFIC = {
    "indian ocean", "south asia", "new delhi", "islamabad", "dhaka",
    "colombo", "kathmandu", "kabul", "mumbai", "karachi", "chennai",
    "kolkata", "hyderabad", "bengaluru", "pune", "arunachal",
    "siachen", "galwan", "doklam", "aksai chin", "balochistan",
    "waziristan", "pok", "gilgit", "rohingya", "loc escalat",
    "lac standoff", "line of control", "line of actual control",
}

SENTIMENT_ORDER = {"Escalating": 0, "Uncertain": 1, "Neutral": 2, "De-escalating": 3}

# ── Actor Network ────────────────────────────────────────────────────────────────
ACTOR_KEYWORDS: dict[str, list[str]] = {
    "India":        ["india", "indian", "modi", "new delhi"],
    "Pakistan":     ["pakistan", "pakistani", "islamabad"],
    "China":        ["china", "chinese", "beijing", "xi jinping", "pla", "ccp"],
    "USA":          ["united states", "america", "american", "washington", "pentagon"],
    "Russia":       ["russia", "russian", "moscow", "putin", "kremlin"],
    "Iran":         ["iran", "iranian", "tehran", "khamenei", "irgc"],
    "Israel":       ["israel", "israeli", "tel aviv", "netanyahu", "idf"],
    "Ukraine":      ["ukraine", "ukrainian", "kyiv", "zelensky"],
    "NATO":         ["nato", "north atlantic treaty"],
    "Saudi Arabia": ["saudi", "riyadh", "saudi arabia"],
    "Turkey":       ["turkey", "turkish", "ankara", "erdogan"],
    "North Korea":  ["north korea", "pyongyang", "kim jong"],
    "Japan":        ["japan", "japanese", "tokyo"],
    "Afghanistan":  ["afghanistan", "taliban", "kabul"],
    "Bangladesh":   ["bangladesh", "dhaka"],
    "Sri Lanka":    ["sri lanka", "colombo"],
    "Myanmar":      ["myanmar", "burma"],
    "Hamas":        ["hamas", "gaza"],
    "Hezbollah":    ["hezbollah"],
    "EU":           ["european union", "brussels"],
    "UN":           ["united nations", "un security council"],
}

ACTOR_REGIONS: dict[str, str] = {
    "India": "South Asia", "Pakistan": "South Asia", "Bangladesh": "South Asia",
    "Sri Lanka": "South Asia", "Afghanistan": "South Asia", "Myanmar": "South Asia",
    "China": "Asia Pacific", "Japan": "Asia Pacific", "North Korea": "Asia Pacific",
    "Russia": "Europe", "Ukraine": "Europe", "EU": "Europe", "NATO": "Europe",
    "USA": "Americas",
    "Iran": "Middle East", "Israel": "Middle East", "Hamas": "Middle East",
    "Hezbollah": "Middle East", "Saudi Arabia": "Middle East", "Turkey": "Middle East",
    "UN": "Global",
}

_CONFLICT_KW  = {"war","attack","strike","conflict","clash","troops","military",
                  "missile","bomb","kill","invasion","offensive","airstrike","shelling","fired"}
_ECONOMIC_KW  = {"trade","investment","oil","supply","export","tariff","sanction",
                  "economic","currency","debt","energy","finance"}
_ALLIANCE_KW  = {"ally","partner","cooperation","joint","exercise","alliance","strategic"}
_DIPLOMAT_KW  = {"talks","meeting","summit","agreement","deal","treaty","visit",
                  "diplomatic","negotiation","dialogue","ceasefire","accord"}


def _detect_actors(text: str) -> list[str]:
    tl = text.lower()
    return [actor for actor, kws in ACTOR_KEYWORDS.items() if any(kw in tl for kw in kws)]


def _edge_type(article: dict) -> str:
    cat  = article.get("category", "")
    text = (article.get("title", "") + " " + article.get("insight", "")).lower()
    words = set(text.split())
    if cat in ("Conflict", "Nuclear", "Terrorism") or words & _CONFLICT_KW:
        return "conflict"
    if words & _ALLIANCE_KW:
        return "alliance"
    if words & _DIPLOMAT_KW or cat == "Diplomacy":
        return "diplomatic"
    if words & _ECONOMIC_KW or cat == "Economic":
        return "economic"
    return "diplomatic"


# Known geopolitical relationships — always present regardless of article volume
BASELINE_EDGES: list[tuple[str, str, str]] = [
    # South Asia
    ("India",        "Pakistan",      "conflict"),
    ("India",        "China",         "conflict"),
    ("India",        "USA",           "alliance"),
    ("India",        "Russia",        "alliance"),
    ("India",        "Bangladesh",    "diplomatic"),
    ("India",        "Sri Lanka",     "alliance"),
    ("India",        "Afghanistan",   "diplomatic"),
    ("India",        "Myanmar",       "diplomatic"),
    ("Pakistan",     "China",         "alliance"),
    ("Pakistan",     "Afghanistan",   "conflict"),
    ("Pakistan",     "USA",           "diplomatic"),
    ("Pakistan",     "Saudi Arabia",  "alliance"),
    # Middle East
    ("Iran",         "Israel",        "conflict"),
    ("Iran",         "USA",           "conflict"),
    ("Iran",         "Saudi Arabia",  "conflict"),
    ("Iran",         "Hezbollah",     "alliance"),
    ("Iran",         "Hamas",         "alliance"),
    ("Iran",         "Russia",        "alliance"),
    ("Israel",       "USA",           "alliance"),
    ("Israel",       "Hamas",         "conflict"),
    ("Israel",       "Hezbollah",     "conflict"),
    ("Saudi Arabia", "USA",           "alliance"),
    ("Turkey",       "NATO",          "alliance"),
    ("Turkey",       "Russia",        "diplomatic"),
    # Europe
    ("Russia",       "Ukraine",       "conflict"),
    ("Russia",       "NATO",          "conflict"),
    ("Russia",       "USA",           "conflict"),
    ("Russia",       "China",         "alliance"),
    ("Ukraine",      "USA",           "alliance"),
    ("Ukraine",      "NATO",          "alliance"),
    ("Ukraine",      "EU",            "alliance"),
    ("NATO",         "EU",            "alliance"),
    # Asia Pacific
    ("China",        "USA",           "conflict"),
    ("China",        "Japan",         "conflict"),
    ("China",        "North Korea",   "alliance"),
    ("Japan",        "USA",           "alliance"),
    ("North Korea",  "USA",           "conflict"),
    # Global
    ("UN",           "Russia",        "diplomatic"),
    ("UN",           "USA",           "diplomatic"),
    ("UN",           "China",         "diplomatic"),
]


def compute_actor_network(articles: list[dict]) -> dict:
    """Build actor co-occurrence network, seeded with known geopolitical relationships."""
    from collections import defaultdict
    node_counts:   dict[str, int]   = defaultdict(int)
    node_severity: dict[str, list]  = defaultdict(list)
    edge_raw:      dict[tuple, dict] = {}

    # ── Seed baseline edges / nodes ──────────────────────────────────────────
    for a1, a2, etype in BASELINE_EDGES:
        for actor in (a1, a2):
            if actor not in node_counts:
                node_counts[actor]   = 0
                node_severity[actor] = [1]
        key = tuple(sorted([a1, a2]))
        if key not in edge_raw:
            edge_raw[key] = {"weight": 1, "types": {etype: 1}, "baseline": True}

    # ── Layer live co-occurrence on top ──────────────────────────────────────
    for a in articles:
        if not _is_analyzed(a):
            continue
        text   = a.get("title", "") + " " + a.get("insight", "")
        actors = _detect_actors(text)
        sev    = a.get("severity", 1)
        etype  = _edge_type(a)

        for actor in actors:
            node_counts[actor]   += 1
            node_severity[actor].append(sev)

        for i in range(len(actors)):
            for j in range(i + 1, len(actors)):
                key = tuple(sorted([actors[i], actors[j]]))
                if key not in edge_raw:
                    edge_raw[key] = {"weight": 0, "types": {}, "baseline": False}
                edge_raw[key]["weight"] += 1
                edge_raw[key]["types"][etype] = edge_raw[key]["types"].get(etype, 0) + 1

    nodes = [
        {
            "id":           actor,
            "region":       ACTOR_REGIONS.get(actor, "Global"),
            "count":        count,
            "avg_severity": round(sum(node_severity[actor]) / len(node_severity[actor]), 2),
            "baseline":     count == 0,   # True = no live articles, anchored by baseline only
        }
        for actor, count in node_counts.items()
    ]

    edges = [
        {
            "source":   k[0],
            "target":   k[1],
            "weight":   v["weight"],
            "type":     max(v["types"], key=v["types"].get),
            "baseline": v.get("baseline", False),
        }
        for k, v in edge_raw.items()
        if k[0] in node_counts and k[1] in node_counts
    ]

    return {"nodes": nodes, "edges": edges}


def _is_south_asia(article: dict) -> bool:
    """Check if an article is relevant to South Asia theatre."""
    # Articles tagged South Asia by the feed
    if article.get("region", "") == "South Asia":
        return True

    text = (
        (article.get("title", "") + " " +
         article.get("summary", "") + " " +
         article.get("insight", ""))
    ).lower()
    actors = [a.lower() for a in article.get("actors", [])]
    all_text = text + " " + " ".join(actors)

    # Direct match on core SA countries
    if any(kw in all_text for kw in SOUTH_ASIA_CORE):
        return True
    # Direct match on specific SA terms
    if any(kw in all_text for kw in SOUTH_ASIA_SPECIFIC):
        return True
    return False


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
    # Ensure SA articles get analyzed: take up to 10 SA + up to 20 non-SA
    sa_pool = [a for a in articles if a.get("region") == "South Asia"][:10]
    other_pool = [a for a in articles if a.get("region") != "South Asia"][:20]
    to_analyze = sa_pool + other_pool
    # Append remaining un-selected articles for fallback data
    analyzed_set = {id(a) for a in to_analyze}
    rest = [a for a in articles if id(a) not in analyzed_set]
    enriched = await loop.run_in_executor(None, analyze_batch, to_analyze + rest, len(to_analyze))
    # Global brief: use top 12 by severity (not position-biased by SA priority)
    brief_candidates = sorted(
        [a for a in enriched if _is_analyzed(a)],
        key=lambda x: -x.get("severity", 0)
    )[:12]
    brief_full = await loop.run_in_executor(
        None, generate_situation_report, brief_candidates
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

    # ── South Asia theatre data ──────────────────────────────────────────────
    sa_articles = [a for a in enriched if _is_south_asia(a)]
    sa_tension_idx, sa_tension_label = compute_tension_index(sa_articles) if sa_articles else (0, "Stable")

    # Generate SA-specific brief from only SA articles
    sa_brief = ""
    if sa_articles:
        sa_top = sorted(
            [a for a in sa_articles if _is_analyzed(a)],
            key=lambda x: -x.get("severity", 0)
        )[:10]
        if sa_top:
            try:
                sa_brief_full = await loop.run_in_executor(
                    None, generate_situation_report, sa_top
                )
                if sa_brief_full:
                    clean = re.sub(r'^#{1,6}\s+',        '', sa_brief_full, flags=re.MULTILINE)
                    clean = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', clean)
                    clean = re.sub(r'^[-*_]{3,}\s*$',    '', clean,     flags=re.MULTILINE)
                    clean = re.sub(r'^\s*[-*+]\s+',      '', clean,     flags=re.MULTILINE)
                    paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip() and len(p.strip()) > 40]
                    prose = [p for p in paragraphs if "." in p]
                    sa_brief = prose[0] if prose else (paragraphs[0] if paragraphs else clean.strip()[:600])
            except Exception as e:
                print(f"[Flashpoint] SA brief error: {e}")

    sa_feed = sorted(
        [a for a in sa_articles if _is_analyzed(a)],
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
        # ── South Asia theatre ───────────────────────────────────────────
        "sa_tension_index":       sa_tension_idx,
        "sa_tension_label":       sa_tension_label,
        "sa_brief":               sa_brief,
        "sa_total_events":        len(sa_articles),
        "sa_flashpoints":         compute_flashpoints(sa_articles),
        "sa_regional_pulse":      compute_regional_pulse(sa_articles),
        "sa_sentiment_breakdown": compute_sentiment_breakdown(sa_articles),
        "sa_category_breakdown":  compute_category_breakdown(sa_articles),
        "sa_category_articles":   compute_category_articles(sa_articles),
        "sa_articles": [
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
            for a in sa_feed
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
        # Auto-scan disabled — hallucinated cases with bad citations pollute the archive
        # Re-enable once citation verification is in place
        # try:
        #     await _scan_for_new_receipts()
        # except Exception as exc:
        #     print(f"[Receipts] Scan error in loop: {exc}")
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

# ── Sarkari Receipt — DB + scan ────────────────────────────────────────────────

_RECEIPTS_FIELDS = [
    "id","name","year","era","era_label","era_class","ministry","minister",
    "loss_crore","loss_display","is_monetary","loss_note","status_label",
    "status_class","status_note","category","cat_label","source","source_url","tweet",
]
_receipts_cache: dict = {}
_receipts_cache_ts: float = 0.0
RECEIPTS_CACHE_TTL = 3600  # 1 hour


def _init_receipts_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS receipts_cases (
                id TEXT PRIMARY KEY,
                name TEXT, year INTEGER, era TEXT, era_label TEXT, era_class TEXT,
                ministry TEXT, minister TEXT, loss_crore REAL DEFAULT 0,
                loss_display TEXT, is_monetary INTEGER DEFAULT 1, loss_note TEXT,
                status_label TEXT, status_class TEXT, status_note TEXT,
                category TEXT, cat_label TEXT, source TEXT, source_url TEXT,
                tweet TEXT, is_auto_detected INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def _seed_receipts():
    """Wipe the receipts table and rebuild entirely from seed on every deploy.
    All data lives in receipts_data.py — there is no user-generated content to preserve.
    This guarantees no hallucinated/stale auto-detected rows survive across deploys.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM receipts_cases")
        print("[Receipts] Cleared table — reseeding from receipts_data.py")
        for c in RECEIPTS_SEED:
            vals = [int(c[f]) if f == "is_monetary" else c.get(f) for f in _RECEIPTS_FIELDS]
            conn.execute(
                f"INSERT INTO receipts_cases ({','.join(_RECEIPTS_FIELDS)}) VALUES ({','.join(['?']*len(_RECEIPTS_FIELDS))})",
                vals
            )
        conn.commit()
        print(f"[Receipts] Seeded {len(RECEIPTS_SEED)} verified cases")
    _receipts_cache.clear()


def _load_receipts_from_db() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT *,is_auto_detected FROM receipts_cases ORDER BY year ASC"
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["is_monetary"] = bool(d.get("is_monetary", 1))
        d["is_auto_detected"] = bool(d.get("is_auto_detected", 0))
        result.append(d)
    return result


async def _scan_for_new_receipts():
    """Use Claude Haiku to scan cached news for new Indian scam/terror events."""
    import anthropic as _ant
    import json as _json
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return
    data = _cache.get("data", {})
    articles = data.get("articles", [])
    if not articles:
        return

    with sqlite3.connect(DB_PATH) as conn:
        existing_names = [r[0] for r in conn.execute("SELECT name FROM receipts_cases")]

    recent = [a for a in articles if a.get("title")][:60]
    if not recent:
        return

    articles_text = "\n\n".join(
        f"Title: {a.get('title','')}\nSummary: {a.get('insight','')}\nSource: {a.get('source','')}\nURL: {a.get('url','')}"
        for a in recent
    )
    existing_str = "; ".join(existing_names[:30])

    prompt = f"""You scan Indian news to find significant NEW accountability events for a public database.

Existing cases (do NOT duplicate): {existing_str}

News articles:
{articles_text}

Find any NEW events that are:
1. Indian corruption, financial fraud, resource misuse, defence irregularity, transparency failure, OR terror attack
2. Not already in the existing list
3. Significant: financial loss > ₹500 Cr, OR terror casualties > 3
4. Recent (reported in these articles)

Return a JSON array of new cases. Each object must have ALL these fields:
id, name, year, era (pre/post based on whether before or after 2014), era_label, era_class (eb-nda1/eb-nda2/eb-upa1/eb-upa2/eb-inc),
ministry, minister, loss_crore (number, 0 if non-monetary), loss_display, is_monetary (true/false),
loss_note (2-3 sentence factual description), status_label, status_class (s-ongoing/s-convicted/s-pending/s-acquitted/s-sc),
status_note, category (corruption/financial/resource/defence/transparency/terrorism), cat_label,
source (outlet name e.g. "The Hindu"), source_url (MUST be the direct article URL from the news item above — never use a homepage like thehindu.com or ndtv.com, always use the specific article URL provided),
tweet (pre-formatted tweet with #SarkariReceipt flashpoint.watch/receipts)

CRITICAL: source_url must be the full specific article URL from the news feed (e.g. https://thehindu.com/news/.../article123.ece), NOT the website homepage.

Return [] if nothing new found. Return ONLY valid JSON array, no markdown."""

    try:
        client = _ant.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = resp.content[0].text.strip()
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return
        new_cases = _json.loads(m.group())
        if not new_cases:
            return

        with sqlite3.connect(DB_PATH) as conn:
            existing_ids = {r[0] for r in conn.execute("SELECT id FROM receipts_cases")}
            inserted = 0
            for c in new_cases:
                if not c.get("id") or c["id"] in existing_ids:
                    continue
                vals = [int(c[f]) if f == "is_monetary" else c.get(f, "") for f in _RECEIPTS_FIELDS]
                conn.execute(
                    f"INSERT OR IGNORE INTO receipts_cases ({','.join(_RECEIPTS_FIELDS)}, is_auto_detected) "
                    f"VALUES ({','.join(['?']*len(_RECEIPTS_FIELDS))}, 1)",
                    vals
                )
                inserted += 1
            conn.commit()
        if inserted:
            _receipts_cache.clear()
            print(f"[Receipts] Auto-detected {inserted} new case(s) from news scan")
    except Exception as e:
        print(f"[Receipts] Scan error: {e}")


# initialise on startup
_init_receipts_db()
_seed_receipts()


@app.get("/api/receipts/cases")
async def get_receipts_cases():
    global _receipts_cache, _receipts_cache_ts
    import time as _time
    now = _time.time()
    if _receipts_cache and (now - _receipts_cache_ts) < RECEIPTS_CACHE_TTL:
        return JSONResponse(_receipts_cache)
    cases = _load_receipts_from_db()
    _receipts_cache = {"cases": cases, "total": len(cases),
                       "auto_detected": sum(1 for c in cases if c.get("is_auto_detected"))}
    _receipts_cache_ts = now
    return JSONResponse(_receipts_cache)


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


@app.get("/network")
async def serve_network():
    """Serve the Actor Network page."""
    with open(BASE_DIR / "network.html") as f:
        return HTMLResponse(f.read())


@app.get("/scenario")
async def serve_scenario():
    """Serve the Scenario Engine page."""
    with open(BASE_DIR / "scenario.html") as f:
        return HTMLResponse(f.read())


@app.get("/receipts")
async def serve_receipts():
    """Serve the Sarkari Receipt accountability archive page."""
    with open(BASE_DIR / "receipts.html") as f:
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


@app.get("/api/network")
async def get_network():
    """Return actor co-occurrence network derived from cached articles."""
    data = _cache.get("data", {})
    seen: set[str] = set()
    combined: list[dict] = []
    for a in data.get("articles", []) + data.get("sa_articles", []):
        url = a.get("url", "")
        if url not in seen:
            seen.add(url)
            combined.append(a)
    return JSONResponse(compute_actor_network(combined))


@app.get("/api/sarvam/status")
async def sarvam_status():
    """Debug: confirm SARVAM_API_KEY is present (never exposes the value)."""
    import os as _os
    key = _os.getenv("SARVAM_API_KEY", "")
    return JSONResponse({
        "key_present": bool(key),
        "key_length":  len(key),
        "key_prefix":  key[:4] + "…" if key else "",
    })


@app.get("/api/sarvam/hindi-brief")
async def get_hindi_brief():
    """Translate the cached South Asia brief to Hindi using Sarvam Mayura."""
    data     = _cache.get("data", {})
    sa_brief = data.get("sa_brief", "")
    if not sa_brief:
        return JSONResponse({"hindi_brief": "", "error": "No South Asia brief in cache yet."})
    loop        = asyncio.get_event_loop()
    hindi_brief = await loop.run_in_executor(None, translate_to_hindi, sa_brief)
    return JSONResponse({"hindi_brief": hindi_brief, "source_brief": sa_brief})


class TTSRequest(BaseModel):
    text:    str
    lang:    str = "hi-IN"
    speaker: str = "shubh"


@app.post("/api/sarvam/tts")
async def get_tts(body: TTSRequest):
    """Convert text to Hindi speech via Sarvam Bulbul v3. Returns base64 MP3."""
    import base64 as _b64
    if not body.text.strip():
        raise HTTPException(400, "text is required")
    loop        = asyncio.get_event_loop()
    audio_bytes = await loop.run_in_executor(
        None, text_to_speech, body.text, body.lang, body.speaker
    )
    if not audio_bytes:
        raise HTTPException(500, "TTS generation failed — check SARVAM_API_KEY.")
    return JSONResponse({"audio": _b64.b64encode(audio_bytes).decode()})


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
