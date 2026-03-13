"""
Uses Claude to analyze geopolitical articles:
- Classify event type (conflict, diplomacy, sanctions, election, etc.)
- Assign severity score (1-5)
- Extract key actors and countries
- Generate a 1-sentence insight
"""

import os
import json
import re
import time
import anthropic
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """You are a geopolitical intelligence analyst.
Given a news headline and summary, respond ONLY with a JSON object (no markdown, no explanation) with these fields:
{
  "category": one of ["Conflict", "Diplomacy", "Sanctions", "Election", "Protest", "Terrorism", "Economic", "Humanitarian", "Nuclear", "Cyber", "Other"],
  "severity": integer 1-5 where 1=minor, 3=significant, 5=critical,
  "actors": list of up to 3 key country or actor names (strings),
  "sentiment": one of ["Escalating", "De-escalating", "Neutral", "Uncertain"],
  "insight": one punchy sentence (max 20 words) summarizing geopolitical significance
}"""


def analyze_article(title: str, summary: str, _retries: int = 3) -> dict:
    """Send article to Claude for analysis. Returns enriched dict."""
    for attempt in range(_retries):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Headline: {title}\n\nSummary: {summary[:500]}"}
                ],
            )
            raw = message.content[0].text.strip()
            # Haiku sometimes wraps JSON in markdown code blocks — strip them
            if "```" in raw:
                raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
            # Extract first JSON object in case there's surrounding text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e} — raw: {message.content[0].text[:120]!r}")
            return _fallback()
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < _retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(f"[Flashpoint] API overloaded (529), retrying in {wait}s… (attempt {attempt+1}/{_retries})")
                time.sleep(wait)
            elif e.status_code in (400, 402, 403) and (
                "credit" in str(e).lower() or e.status_code in (402, 403)
            ):
                print(f"[Flashpoint] API credits exhausted ({e.status_code}) — skipping analysis")
                return _fallback()
            else:
                print(f"Claude API error: {e}")
                return _fallback()
        except Exception as e:
            print(f"Claude error: {e}")
            return _fallback()
    return _fallback()


def analyze_batch(articles: list[dict], max_articles: int = 15) -> list[dict]:
    """Analyze up to max_articles in parallel, fallback for the rest."""
    to_analyze = articles[:max_articles]
    rest       = articles[max_articles:]

    results = [_fallback()] * len(to_analyze)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(analyze_article, a["title"], a.get("summary", "")): i
            for i, a in enumerate(to_analyze)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[Flashpoint] Batch analysis error at index {idx}: {e}")

    enriched  = [{**a, **r} for a, r in zip(to_analyze, results)]
    enriched += [{**a, **_fallback()} for a in rest]
    return enriched


def generate_situation_report(articles: list[dict], _retries: int = 3) -> str:
    """Ask Claude to generate a brief global situation report from top headlines."""
    if not articles:
        return "No articles available for analysis."
    headlines = "\n".join(f"- {a['title']}" for a in articles[:12])
    for attempt in range(_retries):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Based on these current world news headlines, write a concise 3-paragraph "
                        f"geopolitical situation report highlighting the most significant trends, "
                        f"hotspots, and what to watch. Be analytical and direct.\n\nHeadlines:\n{headlines}"
                    )
                }],
            )
            return message.content[0].text.strip()
        except anthropic.APIStatusError as e:
            if e.status_code in (400, 402, 403) and (
                "credit" in str(e).lower() or e.status_code in (402, 403)
            ):
                print(f"[Flashpoint] Situation report: credits exhausted ({e.status_code}) — skipping")
                return "Intelligence brief unavailable — API credits exhausted. Top up at console.anthropic.com."
            elif e.status_code == 529 and attempt < _retries - 1:
                wait = 2 ** attempt
                print(f"[Flashpoint] Situation report overloaded (529), retrying in {wait}s… (attempt {attempt+1}/{_retries})")
                time.sleep(wait)
            else:
                print(f"Situation report API error: {e}")
                return "Intelligence brief temporarily unavailable — high API load. Refreshing shortly."
        except Exception as e:
            print(f"Situation report error: {e}")
            return "Intelligence brief temporarily unavailable. Refreshing shortly."
    return "Intelligence brief temporarily unavailable. Refreshing shortly."


def generate_country_brief(name: str, articles: list[dict], _retries: int = 2) -> str:
    """Generate a focused 2-sentence intelligence assessment for a country/actor."""
    if not articles:
        return f"No recent intelligence on {name} in the current news cycle."
    headlines = "\n".join(
        f"- {a['title']}" for a in articles[:8] if a.get("title")
    )
    if not headlines:
        return f"No recent intelligence on {name} in the current news cycle."
    for attempt in range(_retries):
        try:
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=180,
                system=(
                    "You are a geopolitical intelligence analyst writing for a real-time intelligence dashboard. "
                    "You will always receive real, current headlines to analyze — never ask for clarification or more data. "
                    "Respond ONLY with exactly 2 analytical sentences in present tense. "
                    "Sentence 1: what is happening now. Sentence 2: the key risk or trend to watch. "
                    "No preamble, no meta-commentary, no questions. Just the 2 sentences."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Current headlines about {name}:\n\n{headlines}\n\n"
                        f"Write your 2-sentence intelligence assessment now."
                    )
                }],
            )
            return message.content[0].text.strip()
        except anthropic.APIStatusError as e:
            if e.status_code in (400, 402, 403) and (
                "credit" in str(e).lower() or e.status_code in (402, 403)
            ):
                return "Intelligence summary unavailable — API credits exhausted."
            elif e.status_code == 529 and attempt < _retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "Intelligence summary temporarily unavailable."
        except Exception:
            return "Intelligence summary temporarily unavailable."
    return "Intelligence summary temporarily unavailable."


def _fallback() -> dict:
    return {
        "category": "Other",
        "severity": 1,
        "actors": [],
        "sentiment": "Neutral",
        "insight": "Analysis unavailable.",
    }
