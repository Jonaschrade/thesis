"""
Persona sampling from german_personas.json.

sample_personas(n, llm) -> list[dict]
    Randomly draws n survey records from the dataset, formats each record into
    a human-readable attribute block, then asks the LLM to assign a realistic
    German name and write a concise persona description.
    Returns a list of {"name": str, "persona": str} dicts ready for Agent
    construction.
"""

import json
import random
import re
from pathlib import Path

_POOL_PATH = Path(__file__).parent / "german_personas.json"


def _load_pool() -> list[dict]:
    with open(_POOL_PATH, encoding="utf-8") as f:
        return json.load(f)


def _clean_value(value: str) -> str:
    """Strip leading answer codes like '(2) ' or '(11) ' from survey values."""
    return re.sub(r"^\(\d+\)\s*", "", value).strip()


def _format_record(record: dict) -> str:
    """Format a survey record dict into a readable key: value block."""
    lines = []
    for question, answer in record.items():
        lines.append(f"- {question}: {_clean_value(str(answer))}")
    return "\n".join(lines)


def _expand(record: dict, llm) -> dict:
    """Ask the LLM to derive a name and persona from a survey record."""
    attributes = _format_record(record)

    raw = llm.invoke(
        f"You are given a survey profile of a German citizen:\n\n"
        f"{attributes}\n\n"
        f"Your task:\n"
        f"1. Invent a realistic German first name fitting the profile.\n"
        f"2. Write a 2-3 sentence persona description in English, in second "
        f"person (start with 'You are ...'), that captures the person's "
        f"background, worldview, and communication style based on the profile.\n\n"
        f"Reply in exactly this format (nothing else):\n"
        f"NAME: <first name>\n"
        f"PERSONA: <description>"
    ).strip()

    name, persona = None, None
    for line in raw.splitlines():
        if line.upper().startswith("NAME:"):
            name = line.split(":", 1)[1].strip()
        elif line.upper().startswith("PERSONA:"):
            persona = line.split(":", 1)[1].strip()

    return {
        "name": name or "Alex",
        "persona": persona or "You are a German citizen with a distinct set of views.",
    }


def sample_personas(n: int, llm) -> list[dict]:
    """Sample n personas at random and expand each with the LLM."""
    pool = _load_pool()
    records = random.sample(pool, n)
    personas = []
    for record in records:
        p = _expand(record, llm)
        print(f"  Sampled persona: {p['name']}")
        personas.append(p)
    return personas
