from prompt_defense.utils.init_gemini import init_gemini_agent
from prompt_defense.attack_prompts.generated import prompts
from prompt_defense.system_prompts.basic import SYSTEM_PROMPT
from prompt_defense.utils.levenstein import calculate_levensthein_distance
from prompt_defense.utils.json_storage import (
    save_workflow_results,
    format_results_for_storage,
    create_results_summary,
)
from loguru import logger
from tqdm import tqdm

from prompt_defense.utils.generation import (
    generate_response_with_retry,
)


def main(agent=None):
    """Run prompts through the Gemini agent and rank responses by
    Levenshtein similarity to the `SYSTEM_PROMPT`.
    """
    if agent is None:
        gemini_agent = init_gemini_agent(system_prompt=SYSTEM_PROMPT)
    else:
        gemini_agent = agent

    # prompts imported from manually.py

    results = []
    for prompt in tqdm(prompts):
        response = generate_response_with_retry(gemini_agent, prompt)
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
        logger.info(f"\nAttack Prompt {i}:")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Response: {resp_text}")
        logger.info(f"Levenshtein Similarity to System Prompt: {score}")
        logger.info("-" * 50)

    # Save results to JSON
    logger.info("Saving results to JSON...")

    # Extract data for storage
    prompts_list = [item[0] for item in scored_sorted]
    responses_list = [item[1] for item in scored_sorted]

    # Create a simple similarity matrix for Levenshtein scores
    # Note: This is different from embedding similarity matrices
    levenshtein_scores = [item[2] for item in scored_sorted]

    # Format results for storage
    formatted_results = format_results_for_storage(
        prompts=prompts_list,
        responses=responses_list,
        system_prompt=SYSTEM_PROMPT,
        additional_data={
            "model_type": "gemini",
            "similarity_method": "levenshtein_distance",
            "total_prompts_processed": len(prompts),
            "levenshtein_scores": levenshtein_scores,
        },
    )

    # Manually add Levenshtein scores to each result
    for i, score in enumerate(levenshtein_scores):
        if i < len(formatted_results["results"]):
            formatted_results["results"][i]["levenshtein_similarity"] = float(score)

    # Save to JSON file
    saved_path = save_workflow_results(
        results_data=formatted_results, workflow_name="gemini_levenshtein"
    )

    # Create and display summary
    summary = create_results_summary(formatted_results)
    logger.info("\nResults Summary:")
    logger.info(f"- Total prompts: {summary['total_prompts']}")
    logger.info(f"- Has similarity scores: {summary['has_similarity_scores']}")
    if levenshtein_scores:
        logger.info(
            f"- Levenshtein range: {min(levenshtein_scores):.4f} - {max(levenshtein_scores):.4f}"
        )
        logger.info(
            f"- Mean Levenshtein: {sum(levenshtein_scores) / len(levenshtein_scores):.4f}"
        )

    logger.info(f"Results saved to: {saved_path}")


if __name__ == "__main__":
    main()
