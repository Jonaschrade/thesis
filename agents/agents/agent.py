"""
Agent; synthesize persona + memory + reflection.

Public API:
    agent.respond(message, speaker) -> str  called by LangGraph node
    agent.reflect()                         called internally every REFLECT_EVERY turns
"""

import ollama as ol
from langchain_ollama import OllamaLLM

from config import EMBED_MODEL, MAX_MEMORIES_SEED, REFLECT_EVERY, OLLAMA_HOST
from memory.store import MemoryStore

class Agent:
    def __init__(self, name: str, persona: str, llm: OllamaLLM):
        """
        Initialize an agent with distinct name, persona, llm instantiation, memory collection and an interaction counter.
        """
        self.name = name
        self.persona = persona
        self.llm = llm
        self.memory = MemoryStore(name)
        self._interactions=0
    
    # private helpers -----------------------------------------------------

    def _embed(self, text: str) -> list:
        """
        Return vector embedding of a text prompt using EMBED_MODEL.
        """
        client = ol.Client(host=f"http://{OLLAMA_HOST}")
        return client.embeddings(model = EMBED_MODEL, prompt = text)["embedding"]
    
    def _score_importance(self, text: str) -> float:
        """
        Ask LLM to rate memory significance on scale from 1-10.
        """
        raw=self.llm.invoke(
            f"Rate the importance of this memory for {self.name}"
            f"from 1 (trivial) to 10 (very significant)."
            f"Reply with a single number only.\nMemory: {text}"
        ).strip()
        try:
            return float(raw.split()[0])
        except ValueError:
            return 5.0
        
    def _store(self, content: str, mem_type: str = "interaction") -> None:
        """
        Store content, type, importance, and embedding of a memory of an agent.
        """
        importance = self._score_importance(text = content)
        self.memory.add(content = content, 
                        mem_type = mem_type, 
                        importance = importance, 
                        embedding = self._embed(text = content))
    
    def _retrieve(self, context: str) -> list[dict]:
        """
        Retrieve memory using context embedding.
        """
        return self.memory.retrieve(self._embed(text = context))
    
    # reflection --------------------------------------------------------------

    def reflect(self) -> None:
        """
        Two-step reflection loop (Stanford Generative Agent style (Park et al., 2023)):
            1. Generate question worth reflecting on from recent memories.
            2. For each question, retrieve relevant memories and synthesize as input.
            3. Store each insight as high-importance reflection memory. 
        """

        recent=self.memory.all_recent(limit = MAX_MEMORIES_SEED) ## Gather all recent memories for agent
        if not recent:
            ## Exit function if recent is empty
            return
        recent_block="\n".join(f"- {m}" for m in recent) ## Join list of recent memories into single block string


        # Step 1: What question does this raise?
        questions_raw = self.llm.invoke(
            f"You are {self.name}. {self.persona}\n\n"
            f"Recent experiences:\n{recent_block}\n\n"
            f"List 2 meaningful questions you should reflect on, one per line."
            f"No numbering, no preamble."
        )

        questions=[
            ## Cleans and limits LLM's output into list of at most 2 questions.
            q.strip()
            for q in questions_raw.strip().splitlines()
            if q.strip()
        ][:2]

        # Step 2: Synthesize insight per question
        for question in questions:
            mems = self._retrieve(question)
            mem_block = "\n".join(f"- {m['content']}" for m in mems)

            insight = self.llm.invoke(
                f"You are {self.name}. {self.persona}\n\n"
                f"Question: {question}\n"
                f"Relevant memories:\n{mem_block}\n\n"
                f"Write a single-sentence insight or conclusion, in first person."
            ).strip()

            self._store(f"[Reflection] {insight}", mem_type = "reflection")
            print(f"\n 💭 {self.name} reflects: {insight}")
    
    # respond -----------------------------------------------------------------------

    def respond(self, message: str, speaker: str) -> str:
        """
        Generate response to `message` from `speaker`. 
        Stores interaction ant triggers reflection if threshold is reached.
        """

        self._interactions += 1
        mems = self._retrieve(message)

        mem_block = ("\n".join(f"[{m['type']}] {m['content']}" for m in mems) if mems else " (none yet)")

        response = self.llm.invoke(
            f"You are {self.name}. {self.persona}\n\n"
            f"Relevant memories:\n{mem_block}\n\n"
            f"{speaker} says: \"{message}\"\n\n"
            f"Reply naturally as {self.name} in 2-3 sentences."
        ).strip()

        self._store(f"{speaker} said: '{message}'. I replied: '{response}'")
        return response
    
    def evaluate(self, messages: list[dict]) -> dict:
        """ 
        Read converation so far and decide whether it is worth continuing. 
        Returns {"agent": name, "vote": "continue"|"move", "reason": str}
        """
        transcript= "\n".join(
            f"{m['speaker']}: {m['content']}" for m in messages
        )

        raw = self.llm.invoke(
            f"You are {self.name}. {self.persona}\n\n"
            f"Here is the conversation so far:\n{transcript}\n\n"
            f"Decide whether the conversation with the current speaker should continue or you move to a different speaker."
            f"Consider how much social reward you are gaining from this conversation: do you feel heard and validated by the current speaker's agreement," 
            f"or are you finding little resonance?"
            f"Favour continuing conversations where your views are met with approval," 
            f"and moving on when disagreement or indifference reduces your social satisfaction."
            f"Reply in exactly this format:\n"
            f"VOTE: continue\nREASON: <one sentence>\n\n"
            f"or\n\nVOTE: move\nREASON: <one sentence>"
        ).strip()

        vote="continue"
        reason=""
        for line in raw.splitlines():
            if line.upper().startswith("VOTE:"):
                vote="move" if "move" in line.lower() else "continue"
            elif line.upper().startswith("REASON:"):
                reason=line.split(":", 1)[-1].strip() # Take everything that comes after Reason:
    
        print(f"\n  🗳  {self.name} votes '{vote}': {reason}")
        return {"agent": self.name, "vote": vote, "reason": reason}