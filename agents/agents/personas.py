"""
Persona sampling from german_personas.json.

The JSON file contains 5 246 German citizen survey records stored as Python
dict repr strings (i.e. serialised with str() rather than json.dumps()).
ast.literal_eval is used to parse them back into proper dicts on load.

Public API:
    sample_personas(n, llm) -> list[dict]
        Randomly draws n survey records, derives a German first name and a
        concise German persona description for each via the LLM, and returns
        a list of {"name": str, "persona": str} dicts ready for Agent
        construction.
"""

import ast
import json
import random
import re
from pathlib import Path

_POOL_PATH = Path(__file__).parent / "german_personas.json"


def _load_pool() -> list[dict]:
    """Load and parse the persona pool from disk.

    Entries are stored as Python dict repr strings, so each element is parsed
    with ast.literal_eval rather than relied upon as a native JSON object.
    """
    with open(_POOL_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [ast.literal_eval(entry) if isinstance(entry, str) else entry for entry in raw]


def _clean_value(value: str) -> str:
    """Strip leading numeric answer codes from survey values.

    Survey answers are prefixed with codes like '(2) ' or '(11) ' that encode
    the response scale position. These are not meaningful for the LLM prompt.
    """
    return re.sub(r"^\(\d+\)\s*", "", value).strip()


def _format_record(record: dict) -> str:
    """Format a survey record dict as a human-readable attribute block."""
    return "\n".join(f"- {q}: {_clean_value(str(a))}" for q, a in record.items())


def _expand(record: dict, llm) -> dict:
    """Derive a name and persona description from a survey record via the LLM.

    Args:
        record: A single survey record dict with German question/answer pairs.
        llm:    An instantiated LangChain LLM.

    Returns:
        {"name": str, "persona": str} — falls back to generic values if the
        LLM response cannot be parsed.
    """
    attributes = _format_record(record)

    raw = llm.invoke(
        f"Dir wird das Umfrageprofil einer deutschen Bürgerin / eines deutschen Bürgers vorgelegt:\n\n"
        f"{attributes}\n\n"
        f"Deine Aufgabe:\n"
        f"1. Erfinde einen passenden deutschen Vornamen für diese Person.\n"
        f"2. Verfasse eine 2-3-sätzige Personenbeschreibung auf Deutsch in der zweiten Person "
        f"(beginne mit 'Du bist ...'), die Hintergrund, Weltanschauung und Kommunikationsstil "
        f"der Person auf Basis des Profils widerspiegelt.\n\n"
        f"Antworte ausschließlich in diesem Format:\n"
        f"NAME: <Vorname>\n"
        f"PERSONA: <Beschreibung>"
    ).strip()

    name, persona = None, None
    for line in raw.splitlines():
        if line.upper().startswith("NAME:"):
            name = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PERSONA:"):
            persona = line.split(":", 1)[1].strip()

    return {
        "name": name or "Alex",
        "persona": persona or "Du bist eine deutsche Bürgerin / ein deutscher Bürger mit einer eigenen Meinung.",
    }


def sample_personas(n: int, llm) -> list[dict]:
    """Sample n personas at random and expand each with the LLM.

    Names are guaranteed to be unique across all sampled agents. If the LLM
    assigns a name that is already taken, a new record is drawn and expanded
    until a distinct name is produced.

    Args:
        n:   Number of personas to sample.
        llm: An instantiated LangChain LLM used to expand each record.

    Returns:
        List of n {"name": str, "persona": str} dicts with unique names.
    """
    pool = _load_pool()
    personas = []
    used_names: set[str] = set()

    # Draw from a shuffled copy so retries never repeat an already-used record
    remaining = random.sample(pool, len(pool))

    for record in remaining:
        if len(personas) == n:
            break
        p = _expand(record, llm)
        if p["name"].lower() in used_names:
            continue  # name collision — skip this record and try the next
        used_names.add(p["name"].lower())
        print(f"  Sampled persona: {p['name']}")
        personas.append(p)

    return personas
