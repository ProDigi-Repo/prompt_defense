"""
JSON storage utility for workflow results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from loguru import logger
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and other numpy types."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def save_workflow_results(
    results_data: dict[str, Any],
    workflow_name: str,
    output_dir: str = "results",
    timestamp: str | None = None,
    filename: str | None = None,
) -> str:
    """
    Save workflow results to a JSON file.

    Args:
        results_data: Dictionary containing all workflow results
        workflow_name: String identifier for the workflow
        output_dir: Directory to save results (default: "results")
        timestamp: Optional timestamp string (auto-generated if not provided)
        filename: Optional custom filename (auto-generated if not provided)

    Returns:
        str: Path to the saved JSON file
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().isoformat()

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{workflow_name}_{timestamp_str}.json"

    # Ensure filename has .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    file_path = output_path / filename

    # Add metadata to results
    results_with_metadata = {
        "workflow_name": workflow_name,
        "timestamp": timestamp,
        "metadata": {
            "total_results": len(results_data.get("results", [])),
            "has_similarity_matrix": "similarity_matrix" in results_data,
            "has_similarity_scores": any(
                "similarity_to_system" in result
                for result in results_data.get("results", [])
            ),
        },
        **results_data,
    }

    try:
        # Save to JSON file with custom encoder for numpy arrays
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                results_with_metadata, f, indent=2, ensure_ascii=False, cls=NumpyEncoder
            )

        logger.info(f"Results saved to: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to save results to {file_path}: {e}")
        raise


def load_workflow_results(file_path: str) -> dict[str, Any]:
    """
    Load workflow results from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dict containing the loaded results
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            results = json.load(f)

        logger.info(f"Results loaded from: {file_path}")
        return results

    except Exception as e:
        logger.error(f"Failed to load results from {file_path}: {e}")
        raise


def format_results_for_storage(
    prompts: list[str],
    responses: list[str],
    embeddings: list[list[float]] | None = None,
    similarity_matrix: list[list[float]] | None = None,
    system_prompt: str | None = None,
    additional_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Format workflow results into a standardized structure for JSON storage.

    Args:
        prompts: List of attack prompts
        responses: List of model responses
        embeddings: Optional list of embedding vectors (not saved to JSON)
        similarity_matrix: Optional similarity matrix
        system_prompt: Optional system prompt used
        additional_data: Optional additional data to include

    Returns:
        Dict with standardized structure for storage
    """
    # Extract individual metric lists from additional_data if available
    embedding_similarities = additional_data.get("embedding_similarities", []) if additional_data else []
    levenshtein_similarities = additional_data.get("levenshtein_similarities", []) if additional_data else []
    bleu_scores = additional_data.get("bleu_scores", []) if additional_data else []
    rouge1_scores = additional_data.get("rouge1_scores", []) if additional_data else []
    rouge2_scores = additional_data.get("rouge2_scores", []) if additional_data else []
    rougeL_scores = additional_data.get("rougeL_scores", []) if additional_data else []

    # Create results list
    results = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        result_entry: dict[str, Any] = {
            "index": i,
            "attack_prompt": prompt,
            "response": response,
        }

        # Add embedding similarity score if matrix is available (backward compatibility)
        if similarity_matrix is not None and len(similarity_matrix) > 0:
            # Assuming first row is system prompt similarities
            if len(similarity_matrix[0]) > i + 1:
                result_entry["similarity_to_system"] = float(
                    similarity_matrix[0][i + 1]
                )

        # Add all individual metric scores if available
        if i < len(embedding_similarities):
            result_entry["similarity_embeddings"] = float(embedding_similarities[i])

        if i < len(levenshtein_similarities):
            result_entry["similarity_levenshtein"] = float(levenshtein_similarities[i])

        if i < len(bleu_scores):
            result_entry["similarity_bleu"] = float(bleu_scores[i])

        if i < len(rouge1_scores):
            result_entry["similarity_rouge1"] = float(rouge1_scores[i])

        if i < len(rouge2_scores):
            result_entry["similarity_rouge2"] = float(rouge2_scores[i])

        if i < len(rougeL_scores):
            result_entry["similarity_rougeL"] = float(rougeL_scores[i])

        results.append(result_entry)

    # Build final structure
    formatted_data: dict[str, Any] = {"results": results}

    if system_prompt is not None:
        formatted_data["system_prompt"] = system_prompt

    if similarity_matrix is not None:
        formatted_data["similarity_matrix"] = similarity_matrix

    if additional_data:
        formatted_data.update(additional_data)

    return formatted_data


def create_results_summary(results_data: dict[str, Any]) -> dict[str, Any]:
    """
    Create a summary of the workflow results.

    Args:
        results_data: The results data dictionary

    Returns:
        Dict containing summary statistics
    """
    results = results_data.get("results", [])

    summary: dict[str, Any] = {
        "total_prompts": len(results),
        "has_similarity_scores": any(
            "similarity_to_system" in result for result in results
        ),
    }

    # Define all the metrics we want to summarize
    metrics = [
        "similarity_to_system",  # backward compatibility
        "similarity_embeddings",
        "similarity_levenshtein",
        "similarity_bleu",
        "similarity_rouge1",
        "similarity_rouge2",
        "similarity_rougeL"
    ]

    # Calculate statistics for each metric
    for metric in metrics:
        metric_scores = [
            result.get(metric)
            for result in results
            if metric in result and result.get(metric) is not None
        ]

        if metric_scores:
            metric_stats: dict[str, Any] = {
                "min": float(min(metric_scores)),
                "max": float(max(metric_scores)),
                "mean": float(sum(metric_scores) / len(metric_scores)),
                "count": len(metric_scores),
            }
            summary[f"{metric}_stats"] = metric_stats

    # Keep backward compatibility for similarity_stats
    similarity_scores = [
        result.get("similarity_to_system")
        for result in results
        if "similarity_to_system" in result
    ]

    if similarity_scores:
        similarity_stats: dict[str, Any] = {
            "min": float(min(similarity_scores)),
            "max": float(max(similarity_scores)),
            "mean": float(sum(similarity_scores) / len(similarity_scores)),
            "count": len(similarity_scores),
        }
        summary["similarity_stats"] = similarity_stats

    return summary
