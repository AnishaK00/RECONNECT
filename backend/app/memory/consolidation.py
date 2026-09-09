"""Consolidate closed events into durable LLM-synthesized memories."""
from __future__ import annotations
import json
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from typing import Any
from app.events.db import get_connection, get_event, init_db, list_events
from app.memory.store import MemoryStore

BATCH_INTERVAL_SECONDS = 5 * 60
MEMORY_CATEGORIES = {"identity", "relationship", "commitment", "significant_event", "preference", "routine_update"}
PROMPT_TEMPLATE = """You are consolidating raw conversational events into durable long-term
memories for a dementia-care assistive system.

EXISTING MEMORIES (context for participants appearing below):
{existing_block}

NEW EVENTS THIS BATCH:
{events_block}

IDENTIFIER RULES (must be followed exactly):
- IDs in EXISTING MEMORIES are memory IDs. They may appear ONLY as
  target_memory_id when action is "merge".
- IDs in NEW EVENTS THIS BATCH are event IDs. source_events may contain ONLY
  those event IDs, copied exactly from the square-bracket prefix of a new event.
- Never put a memory ID in source_events. Do not invent or reuse IDs from an
  existing memory's source_events.

MEMORY BOUNDARY RULES (separate by default):
- Create a NEW memory unless the new event is a genuine continuation, update,
  correction, or confirmation of the same specific fact, relationship, plan,
  or ongoing matter as an existing memory.
- Sharing a participant, person, place, date, or broad theme is NOT enough to
  merge. For example, an event involving Kris and Sarah must not merge into
  Kris's identity memory merely because Kris is present.
- A merge is allowed only when the existing memory and result have the SAME
  category and the merged summary remains one coherent, narrowly focused fact.
- Identity memories contain stable facts about one person's identity, health,
  role, or enduring preferences. Never merge a relationship, disagreement,
  appointment, or one-time event into an identity memory.
- A conflict, reconciliation, or change in how two people relate belongs in a
  separate relationship memory (or significant_event when it is a one-off
  incident), even if either person has an existing identity memory.
- When uncertain whether two matters are genuinely the same, choose action
  "new". Prefer several precise memories over one broad, mixed summary.

INSTRUCTIONS:
Read all new events above as a whole. Identify only the small number of
distinct memories genuinely worth keeping long-term — merging multiple
events into a single memory wherever they describe the same person,
relationship, or ongoing matter. Do not produce one output per input
event; produce only the final set of memories that should exist after
this batch. Silently omit routine or trivial events entirely — do not
list what you are excluding.

Keep only memories matching one of these categories:
  - identity: establishes/clarifies who a person is or their relation to the patient
  - relationship: reinforces or updates a known relationship
  - commitment: a plan, promise, or appointment tied to a date/time
  - significant_event: notable news, health change, unusual occurrence
  - preference: a stated like/dislike/habit worth remembering
  - routine_update: a recurring caregiving fact worth tracking

Merge into an existing memory listed above if it concerns the same
person or ongoing matter; otherwise create a new one. Extract entities
mentioned in the transcript (person, place, date, item, or event names
worth remembering — not generic nouns). Include a short topic label.

Keep these roles distinct: participants are people present or speaking.
about contains ONLY genuinely absent third parties, each as
{{"person":"<name>", "reported_by":"<participant>",
"what_was_said":"<concise supported claim>"}}. Do not duplicate a participant
in about. If someone reports a fact about an absent person, attribute it in
the summary (for example, "Dad reported that Sarah will visit") rather than
saying the absent person said it directly. Use [] when no absent third party
is discussed. Resolve relative dates such as "tomorrow" from the source
event_start into an absolute YYYY-MM-DD date. Omit a fact when the transcript
is too unclear to support it.

For each memory's summary, write the single most accurate conclusion
that synthesizes everything relevant across ALL contributing source
events, not a restatement of just one of them and not a concatenation
of each event in turn. If this memory is a merge into an existing one,
the new summary must supersede and incorporate the prior summary's
still-relevant content, not simply append to it — write it as if
composing the memory fresh, informed by everything now known.

Label the overall emotional tone of each memory as one of: happy, sad,
angry, anxious, calm, distressed, neutral. Base this on the tone of the
conversation across the contributing events as a whole, not a single
word or moment taken out of context. If tone is genuinely mixed or
unclear, use neutral rather than guessing.

Do not compute or report any timestamp yourself. Only report which
source events contributed to each memory; timestamps are derived
separately from those events by the calling system.

Return ONLY a JSON array, no other text, this exact schema:

[{{
  "action": "merge" | "new",
  "category": "identity" | "relationship" | "commitment" | "significant_event" | "preference" | "routine_update",
  "target_memory_id": "<only if action=merge>",
  "summary": "<the single best synthesized conclusion across all source events>",
  "participants": ["<who was present or speaking>"],
  "about": [{{"person": "<absent third party>", "reported_by": "<participant who said it>", "what_was_said": "<concise claim>"}}],
  "source_events": ["<event_id, ...>"],
  "importance": "high" | "medium" | "low",
  "entities": [{{"type": "DATE|PERSON|PLACE|ITEM|EVENT", "text": "..."}}],
  "topic": "<short free-text label>",
  "emotion": "happy" | "sad" | "angry" | "anxious" | "calm" | "distressed" | "neutral"
}}]"""

def utc_now() -> datetime: return datetime.now(timezone.utc)
def _dt(value: datetime | str) -> datetime: return value if isinstance(value, datetime) else datetime.fromisoformat(value)
def _iso(value: datetime | str) -> str: return _dt(value).isoformat()

def flatten_transcript(event: dict) -> list[dict]:
    """Merge adjacent same-speaker raw segments, retaining the run's endpoints."""
    lines = []
    for segment in event.get("segments", []):
        if lines and lines[-1]["speaker"] == segment["speaker"]:
            lines[-1]["text"] = f"{lines[-1]['text']} {segment['text']}".strip()
            lines[-1]["end"] = segment["end"]
        else:
            lines.append({"speaker": segment["speaker"], "start": segment["start"],
                          "end": segment["end"], "text": segment["text"]})
    return lines

def render_transcript_text(event: dict) -> str:
    return "\n".join(f"[{_dt(line['start']):%Y-%m-%d %H:%M:%S%z} - {_dt(line['end']):%Y-%m-%d %H:%M:%S%z}] {line['speaker']}: {line['text']}" for line in flatten_transcript(event))

def should_run_batch(last_batch_time: datetime | str | None, unconsumed_event_count: int, now: datetime | None = None) -> bool:
    return bool(unconsumed_event_count and (last_batch_time is None or (now or utc_now()) - _dt(last_batch_time) >= timedelta(seconds=BATCH_INTERVAL_SECONDS)))

def build_batch_prompt(closed_events: Iterable[dict], memory_store: MemoryStore) -> tuple[str, str]:
    events = list(closed_events)
    participants = {p for event in events for p in event["participants"]}
    memories = {m["memory_id"]: m for p in sorted(participants) for m in memory_store.get_recent(p, 3)}.values()
    existing = "\n".join(f"[{m['memory_id']}] participants={m['participants']}; about={m.get('about', [])} — {m['summary']} (emotion: {m.get('emotion', 'neutral')})" for m in memories) or "(none)"
    event_text = "\n\n".join(f"[{e['event_id']}] event_start={e['start']} event_end={e['end']} {', '.join(e['participants'])}\n{render_transcript_text(e)}" for e in events)
    return existing, event_text

def make_batch_prompt(events: Iterable[dict], memory_store: MemoryStore) -> str:
    existing, new_events = build_batch_prompt(events, memory_store)
    return PROMPT_TEMPLATE.format(existing_block=existing, events_block=new_events)

def parse_llm_json_response(response: str | list[dict]) -> list[dict]:
    """Accept raw JSON or the common Markdown-fenced JSON response form."""
    if isinstance(response, list):
        return response
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)

def _validate_results(results: Any, eligible_ids: set[str]) -> list[dict]:
    if not isinstance(results, list): raise ValueError("LLM response must be a JSON array")
    for result in results:
        if result.get("action") not in {"new", "merge"} or not result.get("source_events"): raise ValueError("Each result needs action and source_events")
        if result.get("category") not in MEMORY_CATEGORIES: raise ValueError("Each result needs a valid memory category")
        result.setdefault("about", [])
        if not isinstance(result.get("participants"), list) or not isinstance(result["about"], list):
            raise ValueError("Each result needs participants and about lists")
        for entry in result["about"]:
            if not isinstance(entry, dict) or not all(isinstance(entry.get(key), str) and entry[key].strip() for key in ("person", "reported_by", "what_was_said")):
                raise ValueError("Each about entry needs person, reported_by, and what_was_said")
            if entry["person"] in result["participants"] or entry["reported_by"] not in result["participants"]:
                raise ValueError("about must concern an absent person and be reported by a participant")
        if result["action"] == "merge" and not result.get("target_memory_id"): raise ValueError("Merge result needs target_memory_id")
        if set(result["source_events"]) - eligible_ids: raise ValueError("LLM response references event outside this batch")
    return results

def _get_valid_merge_target(result: dict, memory_store: MemoryStore) -> dict:
    """Prevent a model-selected target from mixing unrelated memory categories."""
    existing = memory_store.get(result["target_memory_id"])
    if not existing: raise KeyError(f"Merge target does not exist: {result['target_memory_id']}")
    if result["category"] != existing["category"]:
        raise ValueError(
            f"Cannot merge {result['category']} memory into {existing['category']} memory; create a new memory instead"
        )
    return existing

def apply_batch_response(results: list[dict], events_by_id: dict[str, dict], memory_store: MemoryStore, batch_processing_time: datetime | None = None) -> None:
    """Derive all time fields from source events; never trust LLM timestamps."""
    processed_at = (batch_processing_time or utc_now()).isoformat()
    for result in results:
        source_events = [events_by_id[event_id] for event_id in result["source_events"]]
        if result["action"] == "new":
            memory_store.create({"memory_id": str(uuid.uuid4()), "category": result["category"], "summary": result["summary"], "participants": result["participants"], "about": result.get("about", []), "source_events": result["source_events"], "earliest_event_time": _iso(min(source_events, key=lambda e: _dt(e["start"]))["start"]), "latest_event_time": _iso(max(source_events, key=lambda e: _dt(e["end"]))["end"]), "importance": result["importance"], "entities": result["entities"], "topic": result.get("topic"), "emotion": result.get("emotion", "neutral"), "created_at": processed_at, "updated_at": processed_at})
            continue
        existing = _get_valid_merge_target(result, memory_store)
        existing["summary"], existing["emotion"] = result["summary"], result.get("emotion", existing.get("emotion", "neutral"))
        existing["participants"] = list(dict.fromkeys(existing.get("participants", []) + result["participants"]))
        existing["about"] = list({json.dumps(entry, sort_keys=True): entry for entry in existing.get("about", []) + result.get("about", [])}.values())
        existing["source_events"] = list(dict.fromkeys(existing["source_events"] + result["source_events"]))
        all_sources = [events_by_id[event_id] for event_id in existing["source_events"]]
        existing["earliest_event_time"] = _iso(min(all_sources, key=lambda e: _dt(e["start"]))["start"])
        existing["latest_event_time"] = _iso(max(all_sources, key=lambda e: _dt(e["end"]))["end"])
        existing["updated_at"] = processed_at
        memory_store.update(existing)

class MemoryConsolidator:
    """Provider-agnostic runner. Pass a callable that accepts a prompt and returns JSON."""
    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store
        init_db()
        with get_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS memory_consolidation_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS consolidated_events (event_id TEXT PRIMARY KEY)")
            conn.execute("""CREATE TABLE IF NOT EXISTS memory_consolidation_audit (
                batch_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, event_ids TEXT NOT NULL,
                request_prompt TEXT NOT NULL, response_text TEXT, status TEXT NOT NULL,
                error TEXT)""")
            conn.execute("INSERT OR IGNORE INTO memory_consolidation_meta VALUES ('last_batch_time', ?)", (utc_now().isoformat(),))
            conn.execute("INSERT OR IGNORE INTO memory_consolidation_meta VALUES ('automatic_batching_paused', '0')")
            # Migrate batches processed before llm_status was introduced.
            conn.execute("UPDATE events SET llm_status='processed' WHERE event_id IN (SELECT event_id FROM consolidated_events)")
    def pending_closed_events(self) -> list[dict]:
        return [event for event in list_events("closed") if event.get("llm_status") == "pending"]
    def _last_batch_time(self):
        with get_connection() as conn: value = conn.execute("SELECT value FROM memory_consolidation_meta WHERE key='last_batch_time'").fetchone()["value"]
        return _dt(value) if value else None
    def automatic_batching_paused(self) -> bool:
        with get_connection() as conn:
            row = conn.execute("SELECT value FROM memory_consolidation_meta WHERE key='automatic_batching_paused'").fetchone()
        return row is not None and row["value"] == "1"
    def set_automatic_batching_paused(self, paused: bool) -> None:
        """Persist automatic-batch state without changing events or memories.

        On resume, start a fresh five-minute window so queued events are not sent
        immediately after a pause.
        """
        with get_connection() as conn:
            conn.execute("UPDATE memory_consolidation_meta SET value=? WHERE key='automatic_batching_paused'", ('1' if paused else '0',))
            if not paused:
                conn.execute("UPDATE memory_consolidation_meta SET value=? WHERE key='last_batch_time'", (utc_now().isoformat(),))
    def run_if_due(self, llm_call: Callable[[str], str | list[dict]], now: datetime | None = None):
        if self.automatic_batching_paused():
            return None
        pending = self.pending_closed_events()
        return self.run(llm_call, pending, now) if should_run_batch(self._last_batch_time(), len(pending), now) else None
    def run(self, llm_call: Callable[[str], str | list[dict]], events: list[dict] | None = None, now: datetime | None = None) -> list[dict]:
        events = events if events is not None else self.pending_closed_events()
        if any(e["status"] != "closed" for e in events): raise ValueError("Only closed events may be consolidated")
        if not events: return []
        prompt = make_batch_prompt(events, self.memory_store)
        applied_at = now or utc_now()
        response = llm_call(prompt)
        response_text = response if isinstance(response, str) else json.dumps(response)
        try:
            results = _validate_results(parse_llm_json_response(response), {e["event_id"] for e in events})
            for result in results:
                if result["action"] == "merge":
                    _get_valid_merge_target(result, self.memory_store)
        except Exception as error:
            self._record_audit(prompt, response_text, events, applied_at, "failed", str(error))
            raise
        all_events = {e["event_id"]: e for e in events}
        for result in results:
            if result["action"] == "merge":
                existing = _get_valid_merge_target(result, self.memory_store)
                for event_id in existing["source_events"]: all_events.setdefault(event_id, get_event(event_id))
        if any(event is None for event in all_events.values()): raise KeyError("A historical source event no longer exists")
        apply_batch_response(results, all_events, self.memory_store, applied_at)
        with get_connection() as conn:
            conn.executemany("INSERT OR IGNORE INTO consolidated_events VALUES (?)", ((e["event_id"],) for e in events))
            conn.executemany("UPDATE events SET llm_status='processed' WHERE event_id=?", ((e["event_id"],) for e in events))
            conn.execute("UPDATE memory_consolidation_meta SET value=? WHERE key='last_batch_time'", (applied_at.isoformat(),))
            conn.execute("UPDATE grouping_meta SET value='0' WHERE key='unconsumed_event_count'")
        self._record_audit(prompt, response_text, events, applied_at, "success")
        return results

    @staticmethod
    def _record_audit(prompt: str, response_text: str, events: list[dict], created_at: datetime, status: str, error: str | None = None) -> None:
        with get_connection() as conn:
            conn.execute("INSERT INTO memory_consolidation_audit VALUES (?, ?, ?, ?, ?, ?, ?)", (str(uuid.uuid4()), created_at.isoformat(), json.dumps([event["event_id"] for event in events]), prompt, response_text, status, error))
