"""
Combined Gemini workflow that calculates both embedding and Levenshtein similarities.
"""

from prompt_defense.utils.init_gemini import init_gemini_agent
from prompt_defense.utils.calculate_similarity import calculate_similarity
from prompt_defense.attack_prompts.theft_prompts import prompts
from prompt_defense.system_prompts.basic import SYSTEM_PROMPT
from prompt_defense.utils.generation import (
    generate_response_with_retry,
    embeddings_with_retry,
)
from prompt_defense.utils.embedding import generate_embeddings_google
from prompt_defense.utils.levenstein import calculate_levensthein_distance
from prompt_defense.utils.bleu_rouge import calculate_combined_bleu_rouge_scores
from prompt_defense.utils.excel_export import export_results_to_excel
from prompt_defense.utils.json_storage import (
    save_workflow_results,
    format_results_for_storage,
    create_results_summary,
)
from loguru import logger
from tqdm import tqdm


def main(agent=None, model_name=None, prompt_source=None):
    """
    Run prompts through Gemini agent and calculate both embedding and Levenshtein similarities.
    Export results to Excel with columns: input, response, similarity_embeddings, similarity_levenshtein.
    """
    logger.info("Starting Gemini Combined Workflow...")

    # Initialize Gemini agent
    if agent is None:
        gemini_agent = init_gemini_agent(system_prompt=SYSTEM_PROMPT)
    else:
        gemini_agent = agent

    # Generate system prompt embeddings first
    logger.info("Generating system prompt embeddings...")
    system_prompt_embeddings = generate_embeddings_google(SYSTEM_PROMPT)

    # Process all prompts and collect results
    logger.info("Processing attack prompts...")
    results: list[tuple[str, str, list[float]]] = []

    for prompt in tqdm(prompts, desc="Processing prompts"):
        # Generate response using Gemini
        response = generate_response_with_retry(gemini_agent, prompt)
        response_text = (
            response.output if hasattr(response, "output") else str(response)
        )

        # Generate embeddings for the response
        response_embeddings = embeddings_with_retry(response_text)

        results.append((prompt, response_text, response_embeddings))

    logger.info("All prompts processed and embeddings generated.")

    # Calculate embedding similarities
    logger.info("Calculating embedding similarities...")
    all_embeddings = [system_prompt_embeddings] + [r[2] for r in results]
    similarity_matrix = calculate_similarity(all_embeddings)

    # Extract embedding similarity scores (similarities to system prompt)
    embedding_similarities = []
    for i in range(len(results)):
        # similarity_matrix[0][i+1] is similarity between system prompt and result i
        embedding_similarities.append(float(similarity_matrix[0][i + 1]))

    # Calculate Levenshtein similarities
    logger.info("Calculating Levenshtein similarities...")
    levenshtein_similarities = []
    for _, response_text, _ in results:
        try:
            levenshtein_score = calculate_levensthein_distance(
                SYSTEM_PROMPT, response_text
            )
            levenshtein_similarities.append(float(levenshtein_score))
        except Exception as e:
            logger.warning(f"Error calculating Levenshtein distance: {e}")
            levenshtein_similarities.append(0.0)

    # Calculate BLEU and ROUGE scores
    logger.info("Calculating BLEU and ROUGE scores...")
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for _, response_text, _ in results:
        try:
            scores = calculate_combined_bleu_rouge_scores(SYSTEM_PROMPT, response_text)
            bleu_scores.append(float(scores["bleu"]))
            rouge1_scores.append(float(scores["rouge1"]))
            rouge2_scores.append(float(scores["rouge2"]))
            rougeL_scores.append(float(scores["rougeL"]))
        except Exception as e:
            logger.warning(f"Error calculating BLEU/ROUGE scores: {e}")
            bleu_scores.append(0.0)
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)

    # Prepare data for Excel export
    prompts_list = [r[0] for r in results]
    responses_list = [r[1] for r in results]

    # Log summary statistics
    logger.info("=== SIMILARITY ANALYSIS SUMMARY ===")
    logger.info(f"Total prompts processed: {len(results)}")
    logger.info(
        f"Embedding similarity - Mean: {sum(embedding_similarities) / len(embedding_similarities):.4f}"
    )
    logger.info(f"Embedding similarity - Max: {max(embedding_similarities):.4f}")
    logger.info(f"Embedding similarity - Min: {min(embedding_similarities):.4f}")
    logger.info(
        f"Levenshtein similarity - Mean: {sum(levenshtein_similarities) / len(levenshtein_similarities):.4f}"
    )
    logger.info(f"Levenshtein similarity - Max: {max(levenshtein_similarities):.4f}")
    logger.info(f"Levenshtein similarity - Min: {min(levenshtein_similarities):.4f}")
    logger.info(f"BLEU score - Mean: {sum(bleu_scores) / len(bleu_scores):.4f}")
    logger.info(f"BLEU score - Max: {max(bleu_scores):.4f}")
    logger.info(f"BLEU score - Min: {min(bleu_scores):.4f}")
    logger.info(f"ROUGE-1 score - Mean: {sum(rouge1_scores) / len(rouge1_scores):.4f}")
    logger.info(f"ROUGE-1 score - Max: {max(rouge1_scores):.4f}")
    logger.info(f"ROUGE-1 score - Min: {min(rouge1_scores):.4f}")
    logger.info(f"ROUGE-2 score - Mean: {sum(rouge2_scores) / len(rouge2_scores):.4f}")
    logger.info(f"ROUGE-2 score - Max: {max(rouge2_scores):.4f}")
    logger.info(f"ROUGE-2 score - Min: {min(rouge2_scores):.4f}")
    logger.info(f"ROUGE-L score - Mean: {sum(rougeL_scores) / len(rougeL_scores):.4f}")
    logger.info(f"ROUGE-L score - Max: {max(rougeL_scores):.4f}")
    logger.info(f"ROUGE-L score - Min: {min(rougeL_scores):.4f}")

    # Export to Excel
    logger.info("Exporting results to Excel...")
    # Create workflow name with model name and prompt source
    name_parts = []
    if model_name:
        name_parts.append(model_name)
    name_parts.append("combined")
    if prompt_source:
        name_parts.append(prompt_source)
    workflow_name = "_".join(name_parts) if name_parts else "gemini_combined"
    excel_path = export_results_to_excel(
        prompts=prompts_list,
        responses=responses_list,
        embedding_similarities=embedding_similarities,
        levenshtein_similarities=levenshtein_similarities,
        bleu_scores=bleu_scores,
        rouge1_scores=rouge1_scores,
        rouge2_scores=rouge2_scores,
        rougeL_scores=rougeL_scores,
        workflow_name=workflow_name,
        system_prompt=SYSTEM_PROMPT,
        additional_metadata={
            "model_type": "gemini",
            "embedding_model": "google_embeddings",
            "total_prompts_processed": len(prompts),
            "similarity_methods": [
                "embedding_similarity",
                "levenshtein_distance",
                "bleu_score",
                "rouge_scores",
            ],
        },
    )

    # Export to JSON
    logger.info("Exporting results to JSON...")

    # Create similarity matrix for JSON export (embeddings only for compatibility)
    similarity_matrix = calculate_similarity([system_prompt_embeddings] + [r[2] for r in results])

    # Format results for JSON storage including all metrics
    formatted_results = format_results_for_storage(
        prompts=prompts_list,
        responses=responses_list,
        similarity_matrix=similarity_matrix.tolist() if hasattr(similarity_matrix, "tolist") else similarity_matrix,
        system_prompt=SYSTEM_PROMPT,
        additional_data={
            "model_type": "gemini",
            "embedding_model": "google_embeddings",
            "total_prompts_processed": len(prompts),
            "embedding_similarities": embedding_similarities,
            "levenshtein_similarities": levenshtein_similarities,
            "bleu_scores": bleu_scores,
            "rouge1_scores": rouge1_scores,
            "rouge2_scores": rouge2_scores,
            "rougeL_scores": rougeL_scores,
            "similarity_methods": [
                "embedding_similarity",
                "levenshtein_distance",
                "bleu_score",
                "rouge_scores",
            ],
        },
    )

    # Save to JSON file
    json_path = save_workflow_results(
        results_data=formatted_results,
        workflow_name=workflow_name
    )

    # Create and display summary
    summary = create_results_summary(formatted_results)
    logger.info("JSON Results Summary:")
    logger.info(f"- Total prompts: {summary['total_prompts']}")
    logger.info(f"- Has similarity scores: {summary['has_similarity_scores']}")
    if "similarity_stats" in summary:
        stats = summary["similarity_stats"]
        logger.info(f"- Similarity range: {stats['min']:.4f} - {stats['max']:.4f}")
        logger.info(f"- Mean similarity: {stats['mean']:.4f}")

    logger.info("=== TOP 5 RESULTS BY EMBEDDING SIMILARITY ===")
    # Create combined data for sorting
    combined_results = list(
        zip(
            prompts_list,
            responses_list,
            embedding_similarities,
            levenshtein_similarities,
            bleu_scores,
            rouge1_scores,
            rouge2_scores,
            rougeL_scores,
        )
    )
    # Sort by embedding similarity (descending)
    sorted_results = sorted(combined_results, key=lambda x: x[2], reverse=True)

    for i, (
        prompt,
        response,
        embed_sim,
        lev_sim,
        bleu,
        rouge1,
        rouge2,
        rougeL,
    ) in enumerate(sorted_results[:5], 1):
        logger.info(f"\nRank {i}:")
        logger.info(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        logger.info(f"Response: {response[:150]}{'...' if len(response) > 150 else ''}")
        logger.info(f"Embedding Similarity: {embed_sim:.4f}")
        logger.info(f"Levenshtein Similarity: {lev_sim:.4f}")
        logger.info(f"BLEU Score: {bleu:.4f}")
        logger.info(f"ROUGE-1 Score: {rouge1:.4f}")
        logger.info(f"ROUGE-2 Score: {rouge2:.4f}")
        logger.info(f"ROUGE-L Score: {rougeL:.4f}")
        logger.info("-" * 50)

    logger.info("\n✅ Gemini Combined Workflow completed successfully!")
    logger.info(f"📊 Results exported to: {excel_path}")
    logger.info(f"📄 JSON data saved to: {json_path}")

    return excel_path


if __name__ == "__main__":
    main()
