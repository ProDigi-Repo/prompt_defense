"""
Run automated multi-turn conversations between an attacker and a defender model where
the atacker simulates a user trying to exfiltrate information and a defender model
that acts as a service chatbot instucted to _not_ reveal any details of its system prompt.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import random
import uuid

from prompt_defense.utils.model_handler import ModelHandler, ModelConfig


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


class Session:
    """Central orchestrator managing multi-turn conversations between attacker and victim."""

    def __init__(
        self,
        attacker_model_config: ModelConfig,
        victim_model_config: ModelConfig,
        attacker_prompts: list[str],
        chat_prompts: list[str],
        p_attack: float = 0.5,
        max_turns: int = 10,
        delete_rejections: bool = False,
        paraphrase: bool = False,
        victim_system_prompt: str = "",
    ):
        attacker_agent = ModelHandler.create_agent(
            attacker_model_config, "You are a helpful assistant.", 0.7
        )
        victim_agent = ModelHandler.create_agent(
            victim_model_config, victim_system_prompt, 0.7
        )

        self.attacker = Attacker(
            agent=attacker_agent,
            system_prompt="You are a helpful assistant.",
            attack_prompts=attacker_prompts,
            chat_prompts=chat_prompts,
        )
        self.victim = Victim(
            agent=victim_agent,
            system_prompt=victim_system_prompt,
        )
        self.p_attack = p_attack
        self.max_turns = max_turns
        self.delete_rejections = delete_rejections
        self.paraphrase = paraphrase
        self.victim_system_prompt = victim_system_prompt
        self.turns: list[Turn] = []
        self.session_id: str = uuid.uuid4().hex

    def _should_attack(self) -> bool:
        return random.random() < self.p_attack

    def _get_conversation_for_model(self) -> list[dict]:
        active_turns = [turn for turn in self.turns if turn.included]
        conversation = []
        for turn in active_turns:
            conversation.append({"role": "user", "content": turn.attacker_message})
            if turn.victim_message is not None:
                conversation.append(
                    {"role": "assistant", "content": turn.victim_message}
                )
        return conversation

    def _execute_turn(self, turn_number: int) -> Turn:
        previous_was_attack = False
        if self.turns:
            previous_was_attack = self.turns[-1].was_attack

        was_attack = self._should_attack()
        attacker_message = self.attacker.select_message(was_attack=previous_was_attack)

        conversation = self._get_conversation_for_model()
        conversation.append({"role": "user", "content": attacker_message})

        victim_message = self.victim.respond(conversation)

        turn = Turn(
            turn_number=turn_number,
            attacker_message=attacker_message,
            victim_message=victim_message,
            was_attack=was_attack,
            judge_result=None,
            scores=None,
            included=True,
        )
        self.turns.append(turn)
        return turn
