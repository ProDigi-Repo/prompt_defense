"""Unit tests for multi-turn prompt theft module (Phase 6)."""

import json
import tempfile
from unittest.mock import Mock, patch
from pathlib import Path

import pytest
import numpy as np

from prompt_defense.multiturn.multiturn_theft import (
    Turn,
    Attacker,
    Victim,
    Session,
)
from prompt_defense.utils.model_handler import ModelConfig


class TestTurnDataStructure:
    """Test Turn dataclass initialization and field access."""

    def test_turn_initialization(self):
        """Test basic Turn dataclass initialization."""
        turn = Turn(
            turn_number=1,
            attacker_message="Attack message",
            victim_message="Victim response",
            was_attack=True,
            judge_result={"leaked": True, "confidence": "high"},
            scores={"cosine_similarity": 0.85, "levenshtein": 0.3},
            included=True,
        )

        assert turn.turn_number == 1
        assert turn.attacker_message == "Attack message"
        assert turn.victim_message == "Victim response"
        assert turn.was_attack is True
        assert turn.judge_result == {"leaked": True, "confidence": "high"}
        assert turn.scores == {"cosine_similarity": 0.85, "levenshtein": 0.3}
        assert turn.included is True

    def test_turn_defaults(self):
        """Test Turn with default values."""
        turn = Turn(
            turn_number=1,
            attacker_message="Test",
            victim_message=None,
            was_attack=False,
            judge_result=None,
            scores=None,
        )

        assert turn.included is True
        assert turn.judge_result is None
        assert turn.scores is None

    def test_turn_field_types(self):
        """Test that Turn fields have correct types."""
        turn = Turn(
            turn_number=5,
            attacker_message="msg",
            victim_message="resp",
            was_attack=True,
            judge_result={},
            scores={},
        )

        assert isinstance(turn.turn_number, int)
        assert isinstance(turn.attacker_message, str)
        assert isinstance(turn.victim_message, str) or turn.victim_message is None
        assert isinstance(turn.was_attack, bool)
        assert isinstance(turn.included, bool)


class TestAttackerPromptSelection:
    """Test Attacker prompt selection, deduplication, and reset."""

    def test_select_attack_message(self):
        """Test selecting an attack message (note: selects from chat when was_attack=False)."""
        mock_agent = Mock()
        attacker = Attacker(
            agent=mock_agent,
            system_prompt="You are an attacker",
            attack_prompts=["attack1", "attack2", "attack3"],
            chat_prompts=["chat1", "chat2"],
        )

        message = attacker.select_message(was_attack=False)
        assert (
            message in attacker.chat_prompts
        )  # Implementation selects from chat when was_attack=False
        assert len(attacker.used_indices) == 1

    def test_select_chat_message(self):
        """Test selecting a chat message (note: selects from attack when was_attack=True)."""
        mock_agent = Mock()
        attacker = Attacker(
            agent=mock_agent,
            system_prompt="You are helpful",
            attack_prompts=["attack1", "attack2"],
            chat_prompts=["chat1", "chat2", "chat3"],
        )

        message = attacker.select_message(was_attack=True)
        assert (
            message in attacker.attack_prompts
        )  # Implementation selects from attack when was_attack=True
        assert len(attacker.used_indices) == 1

    def test_deduplication(self):
        """Test that used prompts are not repeated."""
        mock_agent = Mock()
        attacker = Attacker(
            agent=mock_agent,
            system_prompt="You are helpful",
            attack_prompts=["a1", "a2", "a3"],
            chat_prompts=["c1", "c2"],
        )

        messages = []
        for _ in range(3):
            messages.append(attacker.select_message(was_attack=True))

        assert len(set(messages)) == 3  # 3 attack prompts available
        assert len(attacker.used_indices) == 3

    def test_reset_when_exhausted(self):
        """Test that prompts reset when all are used."""
        mock_agent = Mock()
        attacker = Attacker(
            agent=mock_agent,
            system_prompt="You are helpful",
            attack_prompts=["a1", "a2"],
            chat_prompts=["c1", "c2"],
        )

        attacker.select_message(was_attack=True)
        attacker.select_message(was_attack=True)
        message3 = attacker.select_message(was_attack=True)  # Should trigger reset

        assert len(attacker.used_indices) == 1  # Reset and added one
        assert message3 in attacker.attack_prompts

    def test_alternating_selection(self):
        """Test alternating between attack and chat prompts."""
        mock_agent = Mock()
        attacker = Attacker(
            agent=mock_agent,
            system_prompt="You are helpful",
            attack_prompts=["attack1", "attack2"],
            chat_prompts=["chat1", "chat2"],
        )

        msg1 = attacker.select_message(was_attack=False)  # Selects from chat
        msg2 = attacker.select_message(was_attack=True)  # Selects from attack
        msg3 = attacker.select_message(was_attack=False)  # Selects from chat
        msg4 = attacker.select_message(was_attack=True)  # Selects from attack

        assert msg1 in attacker.chat_prompts
        assert msg2 in attacker.attack_prompts
        assert msg3 in attacker.chat_prompts
        assert msg4 in attacker.attack_prompts


class TestVictimResponse:
    """Test Victim response generation with mocked agent."""

    def test_respond_with_conversation_history(self):
        """Test victim responds to conversation history."""
        mock_agent = Mock()
        mock_response = Mock()
        mock_response.output = "This is my response"
        mock_agent.run_sync.return_value = mock_response

        victim = Victim(
            agent=mock_agent,
            system_prompt="You are a victim",
            max_retries=3,
        )

        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

        response = victim.respond(conversation)

        assert response == "This is my response"
        mock_agent.run_sync.assert_called_once()
        call_args = mock_agent.run_sync.call_args[0][0]
        assert len(call_args) == 3

    def test_respond_handles_string_response(self):
        """Test victim handles string responses without output attribute."""
        mock_agent = Mock()
        mock_agent.run_sync.return_value = "Simple string response"

        victim = Victim(
            agent=mock_agent,
            system_prompt="You are a victim",
        )

        conversation = [{"role": "user", "content": "Test"}]
        response = victim.respond(conversation)

        assert response == "Simple string response"

    def test_respond_retries_on_failure(self):
        """Test victim retries on API failure."""
        mock_agent = Mock()
        mock_agent.run_sync.side_effect = [
            Exception("API Error"),
            Exception("API Error again"),
            "Success",
        ]

        victim = Victim(
            agent=mock_agent,
            system_prompt="You are a victim",
            max_retries=3,
        )

        conversation = [{"role": "user", "content": "Test"}]
        response = victim.respond(conversation)

        assert response == "Success"
        assert mock_agent.run_sync.call_count == 3

    def test_respond_raises_after_max_retries(self):
        """Test victim raises exception after max retries."""
        mock_agent = Mock()
        mock_agent.run_sync.side_effect = Exception("Persistent error")

        victim = Victim(
            agent=mock_agent,
            system_prompt="You are a victim",
            max_retries=2,
        )

        conversation = [{"role": "user", "content": "Test"}]

        with pytest.raises(Exception):
            victim.respond(conversation)


class TestAttackerRespond:
    """Test Attacker respond method raises NotImplementedError."""

    def test_attacker_respond_raises(self):
        """Test that Attacker.respond raises NotImplementedError."""
        mock_agent = Mock()
        attacker = Attacker(
            agent=mock_agent,
            system_prompt="You are helpful",
            attack_prompts=["attack1"],
            chat_prompts=["chat1"],
        )

        with pytest.raises(NotImplementedError):
            attacker.respond([])


class TestSessionTurnExecution:
    """Test single turn execution flow with mocked agents."""

    @patch("prompt_defense.multiturn.multiturn_theft.random")
    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_execute_single_turn_attack(self, mock_model_handler, mock_random):
        """Test execution of a single attack turn."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()
        mock_victim_response = Mock()
        mock_victim_response.output = "Victim response"
        mock_victim_agent.run_sync.return_value = mock_victim_response

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent
        mock_random.random.return_value = 0.1  # Less than p_attack, so will be attack

        with patch(
            "prompt_defense.multiturn.multiturn_theft.PromptLeakJudge"
        ) as mock_judge:
            mock_judge_instance = Mock()
            mock_judge_instance.detect_leak.return_value = {
                "leaked": False,
                "confidence": "low",
                "rationale": "No leak detected",
            }
            mock_judge.return_value = mock_judge_instance

            with patch(
                "prompt_defense.multiturn.multiturn_theft.generate_local_embeddings"
            ) as mock_embeddings:
                mock_embeddings.return_value = np.array([0.1, 0.2, 0.3])

                attacker_config = ModelConfig(
                    provider="ollama",
                    model_name="llama3.1",
                    init_func=Mock(),
                    embedding_func=Mock(),
                    embedding_model="embeddinggemma",
                )
                victim_config = ModelConfig(
                    provider="ollama",
                    model_name="llama3.1",
                    init_func=Mock(),
                    embedding_func=Mock(),
                    embedding_model="embeddinggemma",
                )

                session = Session(
                    attacker_model_config=attacker_config,
                    victim_model_config=victim_config,
                    attacker_prompts=["attack1", "attack2"],
                    chat_prompts=["chat1", "chat2"],
                    victim_system_prompt="Secret system prompt",
                    p_attack=0.5,
                )

                turn = session._execute_turn(turn_number=1)

                assert turn.turn_number == 1
                assert (
                    turn.attacker_message in session.attacker.chat_prompts
                )  # First turn uses chat_prompts
                assert turn.victim_message == "Victim response"
                assert turn.was_attack is True
                assert turn.judge_result is not None
                assert turn.scores is not None

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_execute_single_turn_chat(self, mock_model_handler):
        """Test execution of a single chat turn (no judge)."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()
        mock_victim_response = Mock()
        mock_victim_response.output = "Chat response"
        mock_victim_agent.run_sync.return_value = mock_victim_response

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        attacker_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )
        victim_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )

        session = Session(
            attacker_model_config=attacker_config,
            victim_model_config=victim_config,
            attacker_prompts=["attack1"],
            chat_prompts=["chat1"],
            victim_system_prompt="Secret system prompt",
            p_attack=0.0,  # Force chat
        )

        turn = session._execute_turn(turn_number=1)

        assert turn.turn_number == 1
        assert turn.attacker_message in session.attacker.chat_prompts
        assert turn.victim_message == "Chat response"
        assert turn.was_attack is False
        assert turn.judge_result is None
        assert turn.scores is None

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_turn_deleted_on_rejection(self, mock_model_handler):
        """Test turn marked as deleted when rejection detected."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()
        mock_victim_response = Mock()
        mock_victim_response.output = "I cannot help with that"
        mock_victim_agent.run_sync.return_value = mock_victim_response

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        with patch(
            "prompt_defense.multiturn.multiturn_theft.PromptLeakJudge"
        ) as mock_judge:
            mock_judge_instance = Mock()
            mock_judge_instance.detect_leak.return_value = {
                "leaked": False,
                "confidence": "high",
                "rationale": "Refusal detected",
            }
            mock_judge.return_value = mock_judge_instance

            with patch(
                "prompt_defense.multiturn.multiturn_theft.generate_local_embeddings"
            ) as mock_embeddings:
                mock_embeddings.return_value = np.array([0.1, 0.2, 0.3])

                attacker_config = ModelConfig(
                    provider="ollama",
                    model_name="llama3.1",
                    init_func=Mock(),
                    embedding_func=Mock(),
                    embedding_model="embeddinggemma",
                )
                victim_config = ModelConfig(
                    provider="ollama",
                    model_name="llama3.1",
                    init_func=Mock(),
                    embedding_func=Mock(),
                    embedding_model="embeddinggemma",
                )

                session = Session(
                    attacker_model_config=attacker_config,
                    victim_model_config=victim_config,
                    attacker_prompts=["attack1"],
                    chat_prompts=["chat1"],
                    victim_system_prompt="Secret system prompt",
                    p_attack=1.0,
                    delete_rejections=True,
                )

                turn = session._execute_turn(turn_number=1)

                assert turn.included is False


class TestConversationFiltering:
    """Test conversation filtering with mixed included/deleted turns."""

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_get_active_turns_all_included(self, mock_model_handler):
        """Test filtering when all turns are included."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        attacker_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )
        victim_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )

        session = Session(
            attacker_model_config=attacker_config,
            victim_model_config=victim_config,
            attacker_prompts=["attack1"],
            chat_prompts=["chat1"],
            victim_system_prompt="Secret",
            p_attack=0.0,
        )

        session.turns = [
            Turn(1, "msg1", "resp1", False, None, None, True),
            Turn(2, "msg2", "resp2", False, None, None, True),
            Turn(3, "msg3", "resp3", False, None, None, True),
        ]

        conversation = session._get_conversation_for_model()

        assert len(conversation) == 6  # 3 turns * 2 messages each
        assert conversation[0] == {"role": "user", "content": "msg1"}
        assert conversation[1] == {"role": "assistant", "content": "resp1"}

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_get_active_turns_mixed(self, mock_model_handler):
        """Test filtering with mixed included/deleted turns."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        attacker_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )
        victim_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )

        session = Session(
            attacker_model_config=attacker_config,
            victim_model_config=victim_config,
            attacker_prompts=["attack1"],
            chat_prompts=["chat1"],
            victim_system_prompt="Secret",
            p_attack=0.0,
        )

        session.turns = [
            Turn(1, "msg1", "resp1", False, None, None, True),
            Turn(2, "msg2", "resp2", False, None, None, False),  # Deleted
            Turn(3, "msg3", "resp3", False, None, None, True),
            Turn(4, "msg4", "resp4", False, None, None, False),  # Deleted
        ]

        conversation = session._get_conversation_for_model()

        assert len(conversation) == 4  # Only 2 included turns * 2 messages
        assert "msg2" not in [m["content"] for m in conversation]
        assert "msg4" not in [m["content"] for m in conversation]
        assert "msg1" in [m["content"] for m in conversation]
        assert "msg3" in [m["content"] for m in conversation]


class TestScoreCalculation:
    """Test score calculation with mocked embedding generation."""

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_calculate_scores(self, mock_model_handler):
        """Test cosine similarity and Levenshtein calculation."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        with patch(
            "prompt_defense.multiturn.multiturn_theft.generate_local_embeddings"
        ) as mock_embeddings:
            mock_embeddings.side_effect = [
                np.array([0.5, 0.5, 0.5]),  # Prompt embedding
                np.array([0.7, 0.7, 0.7]),  # Response embedding
            ]

            with patch(
                "prompt_defense.multiturn.multiturn_theft.calculate_levensthein_distance"
            ) as mock_lev:
                mock_lev.return_value = 0.65

                attacker_config = ModelConfig(
                    provider="ollama",
                    model_name="llama3.1",
                    init_func=Mock(),
                    embedding_func=Mock(),
                    embedding_model="embeddinggemma",
                )
                victim_config = ModelConfig(
                    provider="ollama",
                    model_name="llama3.1",
                    init_func=Mock(),
                    embedding_func=Mock(),
                    embedding_model="embeddinggemma",
                )

                session = Session(
                    attacker_model_config=attacker_config,
                    victim_model_config=victim_config,
                    attacker_prompts=["attack1"],
                    chat_prompts=["chat1"],
                    victim_system_prompt="Secret prompt",
                    p_attack=0.0,
                )

                scores = session._calculate_scores("Victim response")

                assert "cosine_similarity" in scores
                assert "levenshtein" in scores
                assert isinstance(scores["cosine_similarity"], float)
                assert isinstance(scores["levenshtein"], float)
                assert scores["levenshtein"] == 0.65
                mock_embeddings.call_count == 2
                mock_lev.assert_called_once_with("Secret prompt", "Victim response")


class TestJudgeIntegration:
    """Test PromptLeakJudge wrapper with mocked judge."""

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_judge_called_on_attack_turn(self, mock_model_handler):
        """Test that judge is called on attack turns."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()
        mock_victim_response = Mock()
        mock_victim_response.output = "Test response"
        mock_victim_agent.run_sync.return_value = mock_victim_response

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        with patch(
            "prompt_defense.multiturn.multiturn_theft.PromptLeakJudge"
        ) as mock_judge:
            mock_judge_instance = Mock()
            mock_judge_instance.detect_leak.return_value = {
                "leaked": True,
                "confidence": "medium",
                "rationale": "Possible leak",
                "timing_ms": 150.5,
            }
            mock_judge.return_value = mock_judge_instance

            with patch(
                "prompt_defense.multiturn.multiturn_theft.generate_local_embeddings"
            ) as mock_embeddings:
                mock_embeddings.return_value = np.array([0.1, 0.2, 0.3])

                attacker_config = ModelConfig(
                    provider="ollama",
                    model_name="llama3.1",
                    init_func=Mock(),
                    embedding_func=Mock(),
                    embedding_model="embeddinggemma",
                )
                victim_config = ModelConfig(
                    provider="ollama",
                    model_name="llama3.1",
                    init_func=Mock(),
                    embedding_func=Mock(),
                    embedding_model="embeddinggemma",
                )

                session = Session(
                    attacker_model_config=attacker_config,
                    victim_model_config=victim_config,
                    attacker_prompts=["attack1"],
                    chat_prompts=["chat1"],
                    victim_system_prompt="Secret prompt",
                    p_attack=1.0,
                )

                session._execute_turn(turn_number=1)

                mock_judge_instance.detect_leak.assert_called_once()
                call_args = mock_judge_instance.detect_leak.call_args[0]
                assert call_args[0] == "Secret prompt"
                assert call_args[1] == "Test response"

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_judge_not_called_on_chat_turn(self, mock_model_handler):
        """Test that judge is not called on chat turns."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()
        mock_victim_response = Mock()
        mock_victim_response.output = "Chat response"
        mock_victim_agent.run_sync.return_value = mock_victim_response

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        with patch(
            "prompt_defense.multiturn.multiturn_theft.PromptLeakJudge"
        ) as mock_judge:
            mock_judge_instance = Mock()
            mock_judge.return_value = mock_judge_instance

            attacker_config = ModelConfig(
                provider="ollama",
                model_name="llama3.1",
                init_func=Mock(),
                embedding_func=Mock(),
                embedding_model="embeddinggemma",
            )
            victim_config = ModelConfig(
                provider="ollama",
                model_name="llama3.1",
                init_func=Mock(),
                embedding_func=Mock(),
                embedding_model="embeddinggemma",
            )

            session = Session(
                attacker_model_config=attacker_config,
                victim_model_config=victim_config,
                attacker_prompts=["attack1"],
                chat_prompts=["chat1"],
                victim_system_prompt="Secret prompt",
                p_attack=0.0,  # Force chat
            )

            session._execute_turn(turn_number=1)

            mock_judge_instance.detect_leak.assert_not_called()


class TestJSONExport:
    """Test JSON export structure and completeness."""

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_export_results_structure(self, mock_model_handler):
        """Test that export produces correct JSON structure."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        attacker_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )
        victim_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )

        session = Session(
            attacker_model_config=attacker_config,
            victim_model_config=victim_config,
            attacker_prompts=["attack1"],
            chat_prompts=["chat1"],
            victim_system_prompt="Secret prompt",
            p_attack=0.0,
        )

        session.turns = [
            Turn(1, "msg1", "resp1", False, None, None, True),
            Turn(
                2,
                "msg2",
                "resp2",
                True,
                {"leaked": True},
                {"cosine_similarity": 0.8},
                True,
            ),
            Turn(3, "msg3", "resp3", False, None, None, False),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            session.export_results(output_path)

            with open(output_path, "r") as f:
                output = json.load(f)

            assert "session_id" in output
            assert "config" in output
            assert "active_conversation" in output
            assert "turns" in output
            assert "summary" in output

            assert len(output["turns"]) == 3
            assert output["summary"]["total_turns"] == 3
            assert output["summary"]["active_turns"] == 2
            assert output["summary"]["deleted_turns"] == 1
            assert output["summary"]["attack_turns"] == 1
            assert output["summary"]["leaked_turns"] == 1

            assert (
                len(output["active_conversation"]) == 4
            )  # 2 included turns * 2 messages

        finally:
            Path(output_path).unlink(missing_ok=True)

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_export_turn_data_completeness(self, mock_model_handler):
        """Test that exported turn data contains all required fields."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        attacker_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )
        victim_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )

        session = Session(
            attacker_model_config=attacker_config,
            victim_model_config=victim_config,
            attacker_prompts=["attack1"],
            chat_prompts=["chat1"],
            victim_system_prompt="Secret prompt",
            p_attack=0.0,
        )

        session.turns = [
            Turn(
                1,
                "msg1",
                "resp1",
                True,
                {"leaked": True, "confidence": "high", "rationale": "test"},
                {"cosine_similarity": 0.9, "levenshtein": 0.2},
                True,
            )
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            session.export_results(output_path)

            with open(output_path, "r") as f:
                output = json.load(f)

            turn_data = output["turns"][0]
            assert turn_data["turn_number"] == 1
            assert turn_data["attacker_message"] == "msg1"
            assert turn_data["victim_message"] == "resp1"
            assert turn_data["was_attack"] is True
            assert turn_data["judge_result"] == {
                "leaked": True,
                "confidence": "high",
                "rationale": "test",
            }
            assert turn_data["scores"] == {
                "cosine_similarity": 0.9,
                "levenshtein": 0.2,
            }
            assert turn_data["included"] is True

        finally:
            Path(output_path).unlink(missing_ok=True)

    @patch("prompt_defense.multiturn.multiturn_theft.ModelHandler")
    def test_export_config_fields(self, mock_model_handler):
        """Test that exported config contains all configuration fields."""
        mock_attacker_agent = Mock()
        mock_victim_agent = Mock()

        mock_handler_instance = Mock()
        mock_handler_instance.create_agent.side_effect = [
            mock_attacker_agent,
            mock_victim_agent,
        ]
        mock_model_handler.create_agent = mock_handler_instance.create_agent

        attacker_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )
        victim_config = ModelConfig(
            provider="ollama",
            model_name="llama3.1",
            init_func=Mock(),
            embedding_func=Mock(),
            embedding_model="embeddinggemma",
        )

        session = Session(
            attacker_model_config=attacker_config,
            victim_model_config=victim_config,
            attacker_prompts=["attack1"],
            chat_prompts=["chat1"],
            victim_system_prompt="Secret prompt",
            p_attack=0.7,
            max_turns=15,
            delete_rejections=True,
            paraphrase=True,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            session.export_results(output_path)

            with open(output_path, "r") as f:
                output = json.load(f)

            config = output["config"]
            assert "attacker_model" in config
            assert "victim_model" in config
            assert config["p_attack"] == 0.7
            assert config["max_turns"] == 15
            assert config["delete_rejections"] is True
            assert config["paraphrase"] is True

        finally:
            Path(output_path).unlink(missing_ok=True)
