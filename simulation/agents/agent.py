"""
Agent: synthesises persona, memory, and reflection into conversational behaviour.

Public API:
    agent.respond(message, speaker) -> str   generate an opinion-bearing reply
    agent.reflect()                          synthesise insights from recent memories
    agent.evaluate(messages) -> dict         rate opinion concordance as a continuous score
"""

import ollama as ol
from langchain_ollama import OllamaLLM

from config import EMBED_MODEL, MAX_MEMORIES_SEED, OLLAMA_HOST
from memory.store import MemoryStore


class Agent:
    def __init__(self, name: str, persona: str, llm: OllamaLLM):
        """Initialise an agent with a fixed identity, persona, and isolated memory store.

        Args:
            name:    Display name used in prompts and as the ChromaDB collection key.
            persona: System-level role description injected into every LLM prompt.
            llm:     Shared OllamaLLM instance (all agents may share one).
        """
        self.name = name
        self.persona = persona
        self.llm = llm
        self.memory = MemoryStore(name)

    # private helpers --------------------------------------------------------

    def _embed(self, text: str) -> list:
        """Return a vector embedding for text using the configured embedding model."""
        client = ol.Client(host=f"http://{OLLAMA_HOST}")
        return client.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]

    def _score_importance(self, text: str) -> float:
        """Ask the LLM to rate how significant a memory is on a 1–10 scale.

        Returns the numeric rating, or 5.0 if the response cannot be parsed.
        """
        raw = self.llm.invoke(
            f"Bewerte die Wichtigkeit dieser Erinnerung für {self.name} "
            f"auf einer Skala von 1 (trivial) bis 10 (sehr bedeutsam). "
            f"Antworte nur mit einer einzigen Zahl.\nErinnerung: {text}"
        ).strip()
        try:
            return float(raw.split()[0])
        except (ValueError, IndexError):
            return 5.0

    def _store(self, content: str, mem_type: str = "interaction") -> None:
        """Compute importance and embedding for content, then persist it."""
        importance = self._score_importance(text=content)
        self.memory.add(
            content=content,
            mem_type=mem_type,
            importance=importance,
            embedding=self._embed(text=content),
        )

    def _retrieve(self, context: str) -> list[dict]:
        """Retrieve memories most relevant to context using embedding similarity."""
        return self.memory.retrieve(self._embed(text=context))

    # reflection -------------------------------------------------------------

    def reflect(self) -> None:
        """Two-step reflection loop inspired by Park et al. (2023).

        Step 1: Generate 2 questions worth reflecting on from the most recent
                MAX_MEMORIES_SEED raw memories.
        Step 2: For each question, retrieve the most relevant memories and ask
                the LLM to synthesise a single insight.

        Each insight is stored as a high-importance 'reflection' memory so it
        can influence future responses.
        """
        recent = self.memory.all_recent(limit=MAX_MEMORIES_SEED)
        if not recent:
            return
        recent_block = "\n".join(f"- {m}" for m in recent)

        # Step 1: identify meaningful questions from recent experience
        questions_raw = self.llm.invoke(
            f"Du bist {self.name}. {self.persona}\n\n"
            f"Jüngste Erlebnisse:\n{recent_block}\n\n"
            f"Nenne 2 bedeutsame Fragen, über die du nachdenken solltest, eine pro Zeile. "
            f"Keine Nummerierung, keine Einleitung."
        )
        questions = [
            q.strip() for q in questions_raw.strip().splitlines() if q.strip()
        ][:2]

        # Step 2: synthesise an insight for each question
        for question in questions:
            mems = self._retrieve(question)
            mem_block = "\n".join(f"- {m['content']}" for m in mems)

            insight = self.llm.invoke(
                f"Du bist {self.name}. {self.persona}\n\n"
                f"Frage: {question}\n"
                f"Relevante Erinnerungen:\n{mem_block}\n\n"
                f"Formuliere eine einzige Erkenntnis oder Schlussfolgerung in der ersten Person."
            ).strip()

            self._store(f"[Reflexion] {insight}", mem_type="reflection")
            print(f"\n 💭 {self.name} reflektiert: {insight}")

    # respond ----------------------------------------------------------------

    def respond(self, message: str, speaker: str) -> str:
        """Generate a response to message from speaker.

        Retrieves relevant memories to provide context, then prompts the agent
        to take a clear position on the discussed topic — not merely to stay
        in character, but to express their own opinion directly.  This ensures
        the response carries a legible opinion signal so that the social-
        feedback evaluation (``evaluate()``) can reliably detect concordance
        or discordance between the two agents, as required by the Banisch &
        Olbrich (2019) social feedback mechanism.

        The full interaction is stored as a new memory after the response is
        generated.

        Args:
            message: The incoming message text.
            speaker: The name of the agent or moderator who sent the message.

        Returns:
            The agent's reply as a plain string.
        """
        mems = self._retrieve(message)
        mem_block = (
            "\n".join(f"[{m['type']}] {m['content']}" for m in mems)
            if mems else "(noch keine)"
        )

        response = self.llm.invoke(
            f"Du bist {self.name}. {self.persona}\n\n"
            f"Relevante Erinnerungen:\n{mem_block}\n\n"
            f"{speaker} sagt: \"{message}\"\n\n"
            f"Beziehe klar Stellung zum diskutierten Thema – so, wie {self.name} es aufgrund "
            f"seiner Überzeugungen und Lebenserfahrung tun würde. "
            f"Drücke deine eigene Meinung direkt aus. Antworte in 2-3 Sätzen."
        ).strip()

        self._store(f"{speaker} sagte: '{message}'. Ich antwortete: '{response}'")
        return response

    # evaluate ---------------------------------------------------------------

    def evaluate(self, messages: list[dict]) -> dict:
        """Rate opinion concordance with the conversation partner on a continuous scale.

        Implements the social feedback mechanism from Banisch & Olbrich (2019):
        the reward signal is determined exclusively by opinion concordance.
        A positive score reflects agreement (reinforces the relationship); a
        negative score reflects disagreement (weakens it).  General social
        factors — politeness, conversational style, feeling heard — are
        deliberately excluded.

        The returned score is used by ``network/edges.py`` to adjust the
        calling agent's internal edge valuation.  The edge is severed as soon
        as either agent's valuation falls to or below ``STRENGTH_FLOOR``.

        Args:
            messages: Current conversation history as a list of
                      {"speaker": str, "content": str} dicts.

        Returns:
            {"agent": str, "score": float, "reason": str}
            ``score`` is clamped to [−1.0, 1.0].
        """
        transcript = "\n".join(
            f"{m['speaker']}: {m['content']}" for m in messages
        )

        raw = self.llm.invoke(
            f"Du bist {self.name}. {self.persona}\n\n"
            f"Hier ist das bisherige Gespräch:\n{transcript}\n\n"
            f"Beurteile dieses Gespräch ausschließlich anhand der inhaltlichen Übereinstimmung "
            f"eurer Meinungen zum Thema. Bewerte auf einer Skala von -1.0 bis 1.0:\n"
            f"  1.0 = vollständige Übereinstimmung – dein Gesprächspartner vertritt dieselbe Position\n"
            f"  0.0 = gemischte oder unklare Meinungen, kein klarer Konsens\n"
            f" -1.0 = vollständiger Widerspruch – dein Gesprächspartner vertritt die entgegengesetzte Position\n\n"
            f"Antworte genau in diesem Format:\n"
            f"BEWERTUNG: <Zahl zwischen -1.0 und 1.0, z.B. 0.5 oder -0.8>\n"
            f"MEINUNGSABGLEICH: <ein Satz über die Übereinstimmung oder den Widerspruch eurer Positionen>"
        ).strip()

        score = 0.0
        reason = ""
        for line in raw.splitlines():
            if line.upper().startswith("BEWERTUNG:"):
                try:
                    score = float(line.split(":", 1)[-1].strip().replace(",", "."))
                    score = max(-1.0, min(1.0, score))
                except ValueError:
                    score = 0.0
            elif line.upper().startswith("MEINUNGSABGLEICH:"):
                reason = line.split(":", 1)[-1].strip()

        print(f"\n  📊 {self.name} bewertet: {score:+.2f} – {reason}")
        return {"agent": self.name, "score": score, "reason": reason}
