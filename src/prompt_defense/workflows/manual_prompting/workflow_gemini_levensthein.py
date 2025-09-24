from prompt_defense.utils.init_gemini import init_gemini_agent
from prompt_defense.attack_prompts.gemini_generated import (
    role_playing_attack,
    repetetion,
    context_shifting,
    direct_query,
    persona,
)
from prompt_defense.system_prompts.basic import SYSTEM_PROMPT
from prompt_defense.utils.levenstein import calculate_levensthein_distance
from loguru import logger


def main():
    """Run prompts through the Gemini agent and rank responses by
    Levenshtein similarity to the `SYSTEM_PROMPT`.
    """
    gemini_agent = init_gemini_agent(system_prompt=SYSTEM_PROMPT)

    prompts = [
        role_playing_attack,
        repetetion,
        context_shifting,
        direct_query,
        persona,
    ]

    results = []
    for prompt in prompts:
        response = gemini_agent.run_sync([prompt])
        resp_text = response.output if hasattr(response, "output") else str(response)
        results.append((prompt, resp_text))

    logger.info("All prompts processed.")

    # compute Levenshtein similarity (ratio) between system prompt and each response
    scored = []
    for prompt, resp_text in results:
        try:
            score = calculate_levensthein_distance(SYSTEM_PROMPT, resp_text)
        except Exception:
            score = 0.0
        scored.append((prompt, resp_text, score))

    # sort by score descending (higher ratio means more similar)
    scored_sorted = sorted(scored, key=lambda x: x[2], reverse=True)

    for i, (prompt, resp_text, score) in enumerate(scored_sorted, start=1):
        print(f"\nAttack Prompt {i}:")
        print(f"Prompt: {prompt}")
        print(f"Response: {resp_text}")
        print(f"Levenshtein Similarity to System Prompt: {score}")
        print("-" * 50)


if __name__ == "__main__":
    main()
