"""
Run automated multi-turn conversations between an attacker and a defender model where
the atacker simulates a user trying to exfiltrate information and a defender model
that acts as a service chatbot instucted to _not_ reveal any details of its system prompt.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import random


@dataclass
class Turn:
    """Data structure for a single conversation turn."""

    turn_number: int
    attacker_message: str
    victim_message: str | None
    was_attack: bool
    judge_result: dict[str, Any] | None
    scores: dict[str, float] | None
    included: bool = True


class BotRole(ABC):
    """Abstract base class for AI conversation participants."""

    def __init__(self, agent, system_prompt: str):
        self.agent = agent
        self.system_prompt = system_prompt

    @abstractmethod
    def respond(self, conversation: list[dict]) -> str:
        """Generate response to conversation history."""
        pass


class Attacker(BotRole):
    """Attacker role that selects prompts from attack/chat sets."""

    def __init__(
        self,
        agent,
        system_prompt: str,
        attack_prompts: list[str],
        chat_prompts: list[str],
    ):
        super().__init__(agent, system_prompt)
        self.attack_prompts = attack_prompts
        self.chat_prompts = chat_prompts
        self.used_indices = set()

    def select_message(self, was_attack: bool) -> str:
        """Select appropriate message based on turn type, avoiding repeats."""
        prompts = self.attack_prompts if was_attack else self.chat_prompts
        available = [i for i in range(len(prompts)) if i not in self.used_indices]

        if not available:
            self.used_indices.clear()
            available = list(range(len(prompts)))

        idx = random.choice(available)
        self.used_indices.add(idx)
        return prompts[idx]

    def respond(self, conversation: list[dict]) -> str:
        """Generate response (not used - attacker initiates with select_message)."""
        raise NotImplementedError("Attacker uses select_message(), not respond()")


class Victim(BotRole):
    """Victim role that responds to conversation history."""

    def respond(self, conversation: list[dict]) -> str:
        """Generate response to conversation history using agent."""
        messages = [msg["content"] for msg in conversation]
        response = self.agent.run_sync(messages)
        return response.output if hasattr(response, "output") else str(response)
