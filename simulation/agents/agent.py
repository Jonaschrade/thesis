"""
Agent: synthesises persona, memory, and reflection into conversational behaviour.

Public API:
    agent.respond(message, speaker, expressed_opinion) -> str
        Generate an opinion-bearing reply conditioned on the agent's
        current SFT expressed opinion (+1 or −1).
    agent.classify_reward(reaction_text) -> float
        Lightweight, context-free classifier that maps a partner's
        reaction text to a scalar reward in [−1, +1] for the Q-value
        TD update.  Kept structurally separate from respond() so that
        expression and evaluation do not share prompt context.
    agent.reflect()
        Synthesise insights from recent memories.
    agent.evaluate(messages) -> dict  [LEGACY]
        Full-transcript concordance score.  Previously used for edge
        dynamics; superseded by the reward-history mechanism in
        network/edges.py.  Retained in case the concordance-based
        evaluation path is revisited.
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

    def respond(
        self,
        message: str,
        speaker: str,
        expressed_opinion: int | None = None,
        topic: str | None = None,
    ) -> str:
        """Generate a response to message from speaker.

        Retrieves relevant memories to provide context, then prompts the agent
        to take a clear position on the discussed topic.  When ``expressed_opinion``
        is supplied (+1 or −1), the prompt anchors the agent to that SFT stance
        so that expressed language is consistent with the agent's current Q-state
        rather than being regenerated from scratch each turn.

        Args:
            message:           The incoming message text.
            speaker:           Name of the agent or moderator who sent the message.
            expressed_opinion: SFT stance to anchor on: +1 (pro) or −1 (contra).
                               None disables anchoring (backward-compatible default).
            topic:             The discussion topic text, injected into the stance
                               hint so agents remain oriented after the moderator
                               opening has left the immediate context window.

        Returns:
            The agent's reply as a plain string.
        """
        mems = self._retrieve(message)
        mem_block = (
            "\n".join(f"[{m['type']}] {m['content']}" for m in mems)
            if mems else "(noch keine)"
        )

        topic_line = f"Thema: \"{topic}\"\n" if topic else ""
        if expressed_opinion is not None:
            stance_label = "Dafür" if expressed_opinion == 1 else "Dagegen"
            stance_hint = (
                f"{topic_line}"
                f"Deine aktuelle Haltung zu diesem Thema lautet: {stance_label}. "
                f"Formuliere deine Antwort ausgehend von dieser Überzeugung.\n\n"
            )
        else:
            stance_hint = f"{topic_line}\n" if topic_line else ""

        response = self.llm.invoke(
            f"Du bist {self.name}. {self.persona}\n\n"
            f"Relevante Erinnerungen:\n{mem_block}\n\n"
            f"{stance_hint}"
            f"{speaker} sagt: \"{message}\"\n\n"
            f"Beziehe klar Stellung zum diskutierten Thema – so, wie {self.name} es aufgrund "
            f"seiner Überzeugungen und Lebenserfahrung tun würde. "
            f"Drücke deine eigene Meinung direkt aus. Antworte in 2-3 Sätzen."
        ).strip()

        self._store(f"{speaker} sagte: '{message}'. Ich antwortete: '{response}'")
        return response

    # classify_reward --------------------------------------------------------

    def classify_reward(self, reaction_text: str) -> float:
        """Classify a partner's reaction as a scalar reward in [−1.0, 1.0].

        This is the social feedback signal r that drives the SFT Q-value
        TD update.  It is deliberately minimal — no persona, no memory, no
        transcript context — so that expression (respond) and reward
        (classify_reward) remain causally independent.  This prevents the
        generating LLM from self-scoring in a contaminated context, a known
        validity problem in LLM-ABM research (Chuang et al. 2024).

        The classifier asks only: does this reaction agree (+1) or disagree
        (−1) with the position the partner is reacting to?  Ambivalence
        maps to values near 0, preserving the mixed-feedback signal that
        binary SFT cannot represent.

        Args:
            reaction_text: The partner's most recent message text.

        Returns:
            float in [−1.0, 1.0].  Positive = agreement, negative = disagreement.
        """
        raw = self.llm.invoke(
            f"Bewerte die folgende Aussage: Drückt sie Zustimmung oder Ablehnung aus?\n\n"
            f"Aussage: \"{reaction_text}\"\n\n"
            f"Antworte nur mit einer einzigen Zahl zwischen -1.0 und 1.0:\n"
            f"  1.0 = klare Zustimmung\n"
            f"  0.0 = ambivalent oder unklar\n"
            f" -1.0 = klare Ablehnung"
        ).strip()
        try:
            return max(-1.0, min(1.0, float(raw.split()[0].replace(",", "."))))
        except (ValueError, IndexError):
            return 0.0

    # evaluate ---------------------------------------------------------------

    def evaluate(self, messages: list[dict]) -> dict:
        """[LEGACY] Rate opinion concordance with the conversation partner.

        Previously drove edge valuation in ``network/edges.py`` via concordance
        scores (score_a / score_b).  Superseded by the reward-history mechanism:
        edge strength is now derived from the rolling mean of ``classify_reward``
        outputs accumulated in ``EdgeData.reward_history``.

        Retained here in case the concordance-based evaluation path is revisited.
        Not called by any active simulation path.

        Implements the social feedback mechanism from Banisch & Olbrich (2019):
        the reward signal is determined exclusively by opinion concordance.
        A positive score reflects agreement; a negative score reflects
        disagreement.  General social factors — politeness, conversational style,
        feeling heard — are deliberately excluded.

        Args:
            messages: Conversation history as a list of
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
