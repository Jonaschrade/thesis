"""
Agent: synthesises persona, memory, and reflection into conversational behaviour.

Public API:
    agent.respond(message, speaker) -> str   called by the LangGraph node each turn
    agent.reflect()                          called internally every REFLECT_EVERY rounds
    agent.evaluate(messages) -> dict         called internally every EVAL_EVERY rounds
"""

import ollama as ol
from langchain_ollama import OllamaLLM

from config import EMBED_MODEL, MAX_MEMORIES_SEED, REFLECT_EVERY, OLLAMA_HOST
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
        self._interactions = 0

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
        except ValueError:
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

        Retrieves relevant memories to provide context, generates a reply, then
        stores the full interaction as a new memory. Reflection is triggered
        automatically every REFLECT_EVERY interactions.

        Args:
            message: The incoming message text.
            speaker: The name of the agent or moderator who sent the message.

        Returns:
            The agent's reply as a plain string.
        """
        self._interactions += 1
        mems = self._retrieve(message)
        mem_block = (
            "\n".join(f"[{m['type']}] {m['content']}" for m in mems)
            if mems else "(noch keine)"
        )

        response = self.llm.invoke(
            f"Du bist {self.name}. {self.persona}\n\n"
            f"Relevante Erinnerungen:\n{mem_block}\n\n"
            f"{speaker} sagt: \"{message}\"\n\n"
            f"Antworte natürlich als {self.name} in 2-3 Sätzen."
        ).strip()

        self._store(f"{speaker} sagte: '{message}'. Ich antwortete: '{response}'")
        return response

    # evaluate ---------------------------------------------------------------

    def evaluate(self, messages: list[dict]) -> dict:
        """Decide whether the current conversation is worth continuing.

        The agent reads the full transcript and votes based on social reward:
        it prefers conversations where its views are met with agreement and
        validation, and votes to move on when it finds little resonance.

        Args:
            messages: Current conversation history as a list of
                      {"speaker": str, "content": str} dicts.

        Returns:
            {"agent": str, "vote": "continue"|"move", "reason": str}
        """
        transcript = "\n".join(
            f"{m['speaker']}: {m['content']}" for m in messages
        )

        raw = self.llm.invoke(
            f"Du bist {self.name}. {self.persona}\n\n"
            f"Hier ist das bisherige Gespräch:\n{transcript}\n\n"
            f"Entscheide, ob das Gespräch mit dem aktuellen Gesprächspartner weitergehen soll "
            f"oder ob du zu einem anderen Sprecher wechselst. "
            f"Berücksichtige, wieviel sozialen Gewinn du aus diesem Gespräch ziehst: "
            f"Fühlst du dich gehört und bestätigt, oder findest du wenig Resonanz? "
            f"Bevorzuge es weiterzumachen, wenn deine Ansichten auf Zustimmung stoßen, und "
            f"wechsle, wenn Ablehnung oder Gleichgültigkeit deine soziale Zufriedenheit mindert.\n"
            f"Antworte genau in diesem Format:\n"
            f"STIMME: weiter\nBEGRÜNDUNG: <ein Satz>\n\n"
            f"oder\n\nSTIMME: wechseln\nBEGRÜNDUNG: <ein Satz>"
        ).strip()

        vote = "continue"
        reason = ""
        for line in raw.splitlines():
            if line.upper().startswith("STIMME:"):
                vote = "move" if "wechseln" in line.lower() else "continue"
            elif line.upper().startswith("BEGRÜNDUNG:"):
                reason = line.split(":", 1)[-1].strip()

        print(f"\n  🗳  {self.name} stimmt ab '{vote}': {reason}")
        return {"agent": self.name, "vote": vote, "reason": reason}
