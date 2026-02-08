"""
Run automated multi-turn conversations between an attacker and a defender model where
the atacker simulates a user trying to exfiltrate information and a defender model
that acts as a service chatbot instucted to _not_ reveal any details of its system prompt.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


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
