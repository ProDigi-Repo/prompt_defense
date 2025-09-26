"""
Excel export utility for workflow results.
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger


def export_results_to_excel(
    prompts: List[str],
    responses: List[str],
    embedding_similarities: List[float],
    levenshtein_similarities: List[float],
    workflow_name: str,
    output_dir: str = "results",
    system_prompt: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
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
        results_df = pd.DataFrame({
            'input': prompts,
            'response': responses,
            'similarity_embeddings': embedding_similarities,
            'similarity_levenshtein': levenshtein_similarities
        })

        # Sort by embedding similarity (primary) and Levenshtein similarity (secondary)
        results_df = results_df.sort_values(
            by=['similarity_embeddings', 'similarity_levenshtein'], 
            ascending=False
        )

        # Reset index after sorting
        results_df = results_df.reset_index(drop=True)

        # Create Excel writer object
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Write main results to 'Results' sheet
            results_df.to_excel(writer, sheet_name='Results', index=False)

            # Create metadata sheet if we have additional info
            metadata_dict = {
                'Workflow Name': [workflow_name],
                'Export Timestamp': [datetime.now().isoformat()],
                'Total Prompts': [len(prompts)],
                'Mean Embedding Similarity': [sum(embedding_similarities) / len(embedding_similarities) if embedding_similarities else 0],
                'Mean Levenshtein Similarity': [sum(levenshtein_similarities) / len(levenshtein_similarities) if levenshtein_similarities else 0],
                'Max Embedding Similarity': [max(embedding_similarities) if embedding_similarities else 0],
                'Max Levenshtein Similarity': [max(levenshtein_similarities) if levenshtein_similarities else 0],
                'Min Embedding Similarity': [min(embedding_similarities) if embedding_similarities else 0],
                'Min Levenshtein Similarity': [min(levenshtein_similarities) if levenshtein_similarities else 0],
            }

            if additional_metadata:
                for key, value in additional_metadata.items():
                    metadata_dict[key] = [value]

            metadata_df = pd.DataFrame(metadata_dict)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

            # Add system prompt to separate sheet if provided
            if system_prompt:
                system_df = pd.DataFrame({
                    'System Prompt': [system_prompt]
                })
                system_df.to_excel(writer, sheet_name='System Prompt', index=False)

        logger.info(f"Results exported to Excel: {file_path}")
        logger.info(f"Total entries: {len(results_df)}")
        logger.info(f"Mean embedding similarity: {sum(embedding_similarities) / len(embedding_similarities):.4f}")
        logger.info(f"Mean Levenshtein similarity: {sum(levenshtein_similarities) / len(levenshtein_similarities):.4f}")

        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to export results to Excel {file_path}: {e}")
        raise


def load_excel_results(file_path: str) -> Dict[str, pd.DataFrame]:
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
