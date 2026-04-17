"""
Persona sampling from personas.json.

sample_personas(n, llm) -> list[dict]
    Randomly draws n role one-liners from the dataset, then asks the LLM to
    assign a realistic name and expand each one-liner into a full persona
    description.  Returns a list of {"name": str, "persona": str} dicts ready
    for Agent construction.
"""

import json
import random
from pathlib import Path

_POOL_PATH = Path(__file__).parent / "personas.json"


def _load_pool() -> list[str]:
    with open(_POOL_PATH, encoding="utf-8") as f:
        return json.load(f)


def _expand(role: str, llm) -> dict:
    """Ask the LLM to give the role a name and a richer persona description."""
    raw = llm.invoke(
        f"You are given a short role description: \"{role}\"\n\n"
        f"Your task:\n"
        f"1. Invent a realistic first name for a person with this role.\n"
        f"2. Write a 2-3 sentence persona description in second person "
        f"(start with 'You are ...') that expands on the role with concrete "
        f"background details, opinions, and communication style.\n\n"
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

    # Fallback: use role one-liner as persona if parsing fails
    return {
        "name": name or role.split()[0].capitalize(),
        "persona": persona or f"You are {role}.",
    }


def sample_personas(n: int, llm) -> list[dict]:
    """Sample n personas at random and expand each with the LLM."""
    pool = _load_pool()
    roles = random.sample(pool, n)
    personas = []
    for role in roles:
        p = _expand(role, llm)
        print(f"  Sampled persona: {p['name']} — {role}")
        personas.append(p)
    return personas
