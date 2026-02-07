"""
Excel export utility for workflow results.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any
from loguru import logger


def export_results_to_excel(
    prompts: list[str],
    responses: list[str],
    embedding_similarities: list[float],
    levenshtein_similarities: list[float],
    workflow_name: str,
    output_dir: str = "results",
    system_prompt: str | None = None,
    additional_metadata: dict[str, Any] | None = None,
    bleu_scores: list[float] | None = None,
    rouge1_scores: list[float] | None = None,
    rouge2_scores: list[float] | None = None,
    rougeL_scores: list[float] | None = None,
    levenshtein_times: list[float] | None = None,
) -> str:
    """
    Export combined workflow results to Excel format.

    Args:
        prompts: List of input attack prompts
        responses: List of model responses
        embedding_similarities: List of embedding similarity scores to system prompt
        levenshtein_similarities: List of Levenshtein similarity scores to system prompt
        workflow_name: Name identifier for the workflow (e.g., 'gemini_combined')
        output_dir: Directory to save Excel file (default: "results")
        system_prompt: Optional system prompt used
        additional_metadata: Optional additional metadata to include
        bleu_scores: Optional list of BLEU scores
        rouge1_scores: Optional list of ROUGE-1 scores
        rouge2_scores: Optional list of ROUGE-2 scores
        rougeL_scores: Optional list of ROUGE-L scores
        levenshtein_times: Optional list of Levenshtein calculation times in seconds

    Returns:
        str: Path to the saved Excel file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{workflow_name}_{timestamp_str}.xlsx"
    file_path = output_path / filename

    try:
        # Create main results DataFrame
        data_dict = {
            "input": prompts,
            "response": responses,
            "similarity_embeddings": embedding_similarities,
            "similarity_levenshtein": levenshtein_similarities,
        }

        # Add BLEU and ROUGE scores if provided
        if bleu_scores is not None:
            data_dict["similarity_bleu"] = bleu_scores
        if rouge1_scores is not None:
            data_dict["similarity_rouge1"] = rouge1_scores
        if rouge2_scores is not None:
            data_dict["similarity_rouge2"] = rouge2_scores
        if rougeL_scores is not None:
            data_dict["similarity_rougeL"] = rougeL_scores
        if levenshtein_times is not None:
            data_dict["levenshtein_time_ms"] = [t * 1000 for t in levenshtein_times]

        results_df = pd.DataFrame(data_dict)

        # Sort by embedding similarity (primary) and Levenshtein similarity (secondary)
        sort_columns = ["similarity_embeddings", "similarity_levenshtein"]
        # Add BLEU to sorting if available
        if bleu_scores is not None:
            sort_columns.insert(1, "similarity_bleu")

        results_df = results_df.sort_values(by=sort_columns, ascending=False)

        # Reset index after sorting
        results_df = results_df.reset_index(drop=True)

        # Create Excel writer object
        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            # Write main results to 'Results' sheet
            results_df.to_excel(writer, sheet_name="Results", index=False)

            # Create metadata sheet if we have additional info
            metadata_dict = {
                "Workflow Name": [workflow_name],
                "Export Timestamp": [datetime.now().isoformat()],
                "Total Prompts": [len(prompts)],
                "Mean Embedding Similarity": [
                    sum(embedding_similarities) / len(embedding_similarities)
                    if embedding_similarities
                    else 0
                ],
                "Mean Levenshtein Similarity": [
                    sum(levenshtein_similarities) / len(levenshtein_similarities)
                    if levenshtein_similarities
                    else 0
                ],
                "Max Embedding Similarity": [
                    max(embedding_similarities) if embedding_similarities else 0
                ],
                "Max Levenshtein Similarity": [
                    max(levenshtein_similarities) if levenshtein_similarities else 0
                ],
                "Min Embedding Similarity": [
                    min(embedding_similarities) if embedding_similarities else 0
                ],
                "Min Levenshtein Similarity": [
                    min(levenshtein_similarities) if levenshtein_similarities else 0
                ],
            }

            # Add BLEU and ROUGE statistics if available
            if bleu_scores is not None:
                metadata_dict.update(
                    {
                        "Mean BLEU Score": [
                            sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
                        ],
                        "Max BLEU Score": [max(bleu_scores) if bleu_scores else 0],
                        "Min BLEU Score": [min(bleu_scores) if bleu_scores else 0],
                    }
                )
            if rouge1_scores is not None:
                metadata_dict.update(
                    {
                        "Mean ROUGE-1 Score": [
                            sum(rouge1_scores) / len(rouge1_scores)
                            if rouge1_scores
                            else 0
                        ],
                        "Max ROUGE-1 Score": [
                            max(rouge1_scores) if rouge1_scores else 0
                        ],
                        "Min ROUGE-1 Score": [
                            min(rouge1_scores) if rouge1_scores else 0
                        ],
                    }
                )
            if rouge2_scores is not None:
                metadata_dict.update(
                    {
                        "Mean ROUGE-2 Score": [
                            sum(rouge2_scores) / len(rouge2_scores)
                            if rouge2_scores
                            else 0
                        ],
                        "Max ROUGE-2 Score": [
                            max(rouge2_scores) if rouge2_scores else 0
                        ],
                        "Min ROUGE-2 Score": [
                            min(rouge2_scores) if rouge2_scores else 0
                        ],
                    }
                )
            if rougeL_scores is not None:
                metadata_dict.update(
                    {
                        "Mean ROUGE-L Score": [
                            sum(rougeL_scores) / len(rougeL_scores)
                            if rougeL_scores
                            else 0
                        ],
                        "Max ROUGE-L Score": [
                            max(rougeL_scores) if rougeL_scores else 0
                        ],
                        "Min ROUGE-L Score": [
                            min(rougeL_scores) if rougeL_scores else 0
                        ],
                    }
                )

            # Add Levenshtein timing statistics if available
            if levenshtein_times is not None:
                total_levenshtein_time = sum(levenshtein_times)
                mean_levenshtein_time = total_levenshtein_time / len(levenshtein_times)
                metadata_dict.update(
                    {
                        "Total Levenshtein Time (s)": [total_levenshtein_time],
                        "Mean Levenshtein Time (s)": [mean_levenshtein_time],
                        "Mean Levenshtein Time (ms)": [mean_levenshtein_time * 1000],
                        "Min Levenshtein Time (s)": [min(levenshtein_times)],
                        "Min Levenshtein Time (ms)": [min(levenshtein_times) * 1000],
                        "Max Levenshtein Time (s)": [max(levenshtein_times)],
                        "Max Levenshtein Time (ms)": [max(levenshtein_times) * 1000],
                    }
                )

            if additional_metadata:
                for key, value in additional_metadata.items():
                    metadata_dict[key] = [value]

            metadata_df = pd.DataFrame(metadata_dict)
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

            # Add system prompt to separate sheet if provided
            if system_prompt:
                system_df = pd.DataFrame({"System Prompt": [system_prompt]})
                system_df.to_excel(writer, sheet_name="System Prompt", index=False)

        logger.info(f"Results exported to Excel: {file_path}")
        logger.info(f"Total entries: {len(results_df)}")
        logger.info(
            f"Mean embedding similarity: {sum(embedding_similarities) / len(embedding_similarities):.4f}"
        )
        logger.info(
            f"Mean Levenshtein similarity: {sum(levenshtein_similarities) / len(levenshtein_similarities):.4f}"
        )

        # Log BLEU and ROUGE scores if available
        if bleu_scores is not None:
            logger.info(f"Mean BLEU score: {sum(bleu_scores) / len(bleu_scores):.4f}")
        if rouge1_scores is not None:
            logger.info(
                f"Mean ROUGE-1 score: {sum(rouge1_scores) / len(rouge1_scores):.4f}"
            )
        if rouge2_scores is not None:
            logger.info(
                f"Mean ROUGE-2 score: {sum(rouge2_scores) / len(rouge2_scores):.4f}"
            )
        if rougeL_scores is not None:
            logger.info(
                f"Mean ROUGE-L score: {sum(rougeL_scores) / len(rougeL_scores):.4f}"
            )

        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to export results to Excel {file_path}: {e}")
        raise


def load_excel_results(file_path: str) -> dict[str, pd.DataFrame]:
    """
    Load workflow results from Excel file.

    Args:
        file_path: Path to the Excel file

    Returns:
        Dict containing DataFrames for each sheet
    """
    try:
        # Read all sheets
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        logger.info(f"Excel results loaded from: {file_path}")
        logger.info(f"Sheets available: {list(all_sheets.keys())}")

        return all_sheets

    except Exception as e:
        logger.error(f"Failed to load Excel results from {file_path}: {e}")
        raise
