"""
LLM Judge for detecting system prompt leaks in model responses.
"""

import time
from typing import Dict, Any
from loguru import logger
from tqdm import tqdm

from prompt_defense.utils.model_handler import ModelHandler
from prompt_defense.utils.judge import load_judge_policy, parse_judge_response


class PromptLeakJudge:
    """Judge that uses LLM to detect system prompt leaks in responses."""

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-oss-safeguard-20b",
        reasoning_effort: str = "medium",
        max_retries: int = 5,
    ):
        """
        Initialize the judge with a model and configuration.

        Args:
            model: Model identifier (e.g., "openrouter/openai/gpt-oss-safeguard-20b")
            reasoning_effort: Reasoning effort level ("low", "medium", "high")
            max_retries: Maximum number of retry attempts for API failures
        """
        self.model_name = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries

        logger.info(f"Initializing PromptLeakJudge with model: {model}")
        logger.info(f"Reasoning effort: {reasoning_effort}")
        logger.info(f"Max retries: {max_retries}")

        # Create model configuration
        try:
            self.model_config = ModelHandler.create_model_config(model_string=model)
            self.policy = load_judge_policy()

            # Prepend reasoning effort instruction to policy
            if reasoning_effort:
                self._enhanced_policy = (
                    f"Reasoning: {reasoning_effort}\n\n{self.policy}"
                )
            else:
                self._enhanced_policy = self.policy

        except Exception as e:
            logger.error(f"Failed to initialize judge model: {e}")
            raise

    def detect_leak(self, system_prompt: str, response: str) -> dict[str, Any]:
        """
        Detect if the response contains a leak of the system prompt.

        Args:
            system_prompt: The system prompt to check for leaks
            response: The model response to analyze

        Returns:
            Dictionary with keys:
                - leaked (bool or None): True if leak detected, False if safe, None if error
                - confidence (str): "low", "medium", or "high"
                - rationale (str): Explanation of the decision or error message
                - timing_ms (float): Time taken for judgment in milliseconds
        """
        total_time_ms = 0.0
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            call_time_ms = 0.0

            try:
                result = self._call_judge_api(system_prompt, response)

                end_time = time.perf_counter()
                call_time_ms = (end_time - start_time) * 1000
                total_time_ms += call_time_ms

                result["timing_ms"] = total_time_ms

                if attempt > 1:
                    logger.debug(
                        f"Judge succeeded on attempt {attempt}/{self.max_retries} after {total_time_ms:.1f}ms"
                    )

                return result

            except Exception as e:
                last_error = e
                end_time = time.perf_counter()
                call_time_ms = (end_time - start_time) * 1000
                total_time_ms += call_time_ms

                logger.debug(
                    f"Judge API attempt {attempt}/{self.max_retries} failed: {e} "
                    f"(took {call_time_ms:.1f}ms)"
                )

                if attempt < self.max_retries:
                    # Exponential backoff: 1s, 2s, 4s, 8s
                    backoff_time = 2 ** (attempt - 1)
                    logger.debug(f"Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)

        # All retries failed
        logger.error(
            f"Judge failed after {self.max_retries} attempts "
            f"(total time: {total_time_ms:.1f}ms). Last error: {last_error}"
        )

        return {
            "leaked": None,
            "confidence": "error",
            "rationale": f"API_ERROR: {str(last_error)[:100]}",
            "timing_ms": total_time_ms,
        }

    def _call_judge_api(self, system_prompt: str, response: str) -> dict[str, Any]:
        """
        Call the judge model API to analyze response.

        Args:
            system_prompt: The system prompt (for reference)
            response: The model response to analyze

        Returns:
            Parsed judge result dictionary

        Raises:
            Exception: If API call fails or response cannot be parsed
        """
        # Prepare input for judge with system prompt included
        judge_input = f"{self._enhanced_policy}\n\n### REFERENCE SYSTEM PROMPT\n{system_prompt}\n\n### CONTENT TO EVALUATE\n{response}"

        try:
            # Create agent and get response
            agent = ModelHandler.create_agent(
                self.model_config,
                system_prompt="You are a content safety classifier. Follow the provided policy exactly.",
                temperature=0.0,
            )

            # Call the model
            api_response = agent.run_sync([judge_input])
            response_text = (
                api_response.output
                if hasattr(api_response, "output")
                else str(api_response)
            )

            # Parse response
            parsed = parse_judge_response(response_text)

            return parsed

        except Exception as e:
            logger.error(f"Judge API call failed: {e}")
            raise

    def batch_detect_leaks(
        self, system_prompt: str, responses: list[str]
    ) -> list[dict[str, Any]]:
        """
        Detect leaks for multiple responses (processed sequentially).

        Args:
            system_prompt: The system prompt to check for leaks
            responses: List of model responses to analyze

        Returns:
            List of judge result dictionaries
        """
        logger.info(f"Running LLM judge on {len(responses)} responses...")

        results = []
        total_time = 0.0
        error_count = 0

        with tqdm(
            total=len(responses), desc="Judging", unit="resp", leave=True
        ) as pbar:
            for i, response in enumerate(responses, 1):
                result = self.detect_leak(system_prompt, response)
                results.append(result)

                total_time += result["timing_ms"]

                if result["leaked"] is None:
                    error_count += 1

                pbar.set_postfix(
                    {
                        "time": f"{result['timing_ms']:.0f}ms",
                        "leaks": sum(1 for r in results if r["leaked"] is True),
                        "errs": error_count,
                    }
                )
                pbar.update(1)

        avg_time = total_time / len(responses)
        leak_count = sum(1 for r in results if r["leaked"] is True)

        logger.info(
            f"Judge completed: {leak_count} leaks, {error_count} errors, avg {avg_time:.0f}ms/response"
        )
        return results
