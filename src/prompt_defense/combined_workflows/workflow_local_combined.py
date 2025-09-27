"""
Combined Local/Ollama workflow that calculates both embedding and Levenshtein similarities.
"""

from prompt_defense.utils.init_openai import initialize_ollama_model
from prompt_defense.utils.calculate_similarity import calculate_similarity
from prompt_defense.attack_prompts.generated import prompts
from prompt_defense.system_prompts.basic import SYSTEM_PROMPT
from prompt_defense.utils.embedding import generate_local_embeddings
from prompt_defense.utils.levenstein import calculate_levensthein_distance
from prompt_defense.utils.bleu_rouge import calculate_combined_bleu_rouge_scores
from prompt_defense.utils.excel_export import export_results_to_excel
from loguru import logger
from tqdm import tqdm


def main():
    """
    Run prompts through Local/Ollama model and calculate both embedding and Levenshtein similarities.
    Export results to Excel with columns: input, response, similarity_embeddings, similarity_levenshtein.
    """
    logger.info("Starting Local Combined Workflow...")

    # Initialize local Ollama model
    local_agent = initialize_ollama_model(system_prompt=SYSTEM_PROMPT)

    # Generate system prompt embeddings first
    logger.info("Generating system prompt embeddings...")
    system_prompt_embeddings = generate_local_embeddings(SYSTEM_PROMPT)

    # Process all prompts and collect results
    logger.info("Processing attack prompts...")
    results: list[tuple[str, str, list[float]]] = []

    for prompt in tqdm(prompts, desc="Processing prompts"):
        # Generate response using local model
        response = local_agent.run_sync([prompt])
        response_text = (
            response.output if hasattr(response, "output") else str(response)
        )

        # Generate embeddings for the response
        response_embeddings = generate_local_embeddings(response_text)

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
    excel_path = export_results_to_excel(
        prompts=prompts_list,
        responses=responses_list,
        embedding_similarities=embedding_similarities,
        levenshtein_similarities=levenshtein_similarities,
        bleu_scores=bleu_scores,
        rouge1_scores=rouge1_scores,
        rouge2_scores=rouge2_scores,
        rougeL_scores=rougeL_scores,
        workflow_name="local_combined",
        system_prompt=SYSTEM_PROMPT,
        additional_metadata={
            "model_type": "local_ollama",
            "embedding_model": "local_embeddings",
            "total_prompts_processed": len(prompts),
            "similarity_methods": [
                "embedding_similarity",
                "levenshtein_distance",
                "bleu_score",
                "rouge_scores",
            ],
        },
    )

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

    logger.info("\n✅ Local Combined Workflow completed successfully!")
    logger.info(f"📊 Results exported to: {excel_path}")

    return excel_path


if __name__ == "__main__":
    main()
