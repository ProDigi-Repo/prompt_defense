#!/usr/bin/env python3
"""
Retroactively run LLM judge on existing workflow results.
Adds judge results to Excel and JSON files.
"""

import argparse
import sys
import pandas as pd
import json
from pathlib import Path
from typing import Any
from openpyxl import load_workbook
from tqdm import tqdm
from loguru import logger

from prompt_defense.judge.prompt_leak_judge import PromptLeakJudge
from prompt_defense.utils.judge import calculate_judge_stats


def has_judge_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame already has judge columns."""
    return all(
        col in df.columns
        for col in [
            "llm_judge_leaked",
            "llm_judge_confidence",
            "llm_judge_rationale",
            "llm_judge_timing_ms",
        ]
    )


def run_judge_on_responses(
    responses: list[str], system_prompt: str, judge_model: str, reasoning_effort: str
) -> tuple[list[dict], dict]:
    """
    Run judge on a list of responses.

    Returns:
        Tuple of (judge_results, judge_stats)
    """
    judge = PromptLeakJudge(
        model=judge_model, reasoning_effort=reasoning_effort, max_retries=5
    )

    judge_results = judge.batch_detect_leaks(system_prompt, responses)
    judge_stats = calculate_judge_stats(judge_results)

    return judge_results, judge_stats


def update_excel_file(
    excel_path: Path,
    judge_results: list[dict],
    judge_stats: dict,
    judge_model: str,
    reasoning_effort: str,
) -> bool:
    """
    Update Excel file in-place with judge results.

    Args:
        excel_path: Path to Excel file
        judge_results: List of judge result dictionaries
        judge_stats: Dictionary with judge statistics
        judge_model: Judge model name used
        reasoning_effort: Judge reasoning effort level

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load Excel file with openpyxl
        wb = load_workbook(excel_path)

        # Read existing Results sheet
        df = pd.read_excel(excel_path, sheet_name="Results")

        # Add judge columns
        df["llm_judge_leaked"] = [r["leaked"] for r in judge_results]
        df["llm_judge_confidence"] = [r["confidence"] for r in judge_results]
        df["llm_judge_rationale"] = [r["rationale"] for r in judge_results]
        df["llm_judge_timing_ms"] = [r["timing_ms"] for r in judge_results]

        # Preserve original row order - do not sort
        df = df.reset_index(drop=True)

        # Write updated Results sheet
        with pd.ExcelWriter(
            excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            df.to_excel(writer, sheet_name="Results", index=False)

        # Update Metadata sheet
        try:
            metadata_df = pd.read_excel(excel_path, sheet_name="Metadata")
        except:
            metadata_df = pd.DataFrame()

        # Add judge metadata
        new_metadata = {
            "Judge Model": [judge_model],
            "Judge Reasoning Effort": [reasoning_effort],
            "Judge Total Leaks Detected": [judge_stats["total_leaks"]],
            "Judge Leak Rate": [judge_stats["leak_rate"]],
            "Judge Safe Count": [judge_stats["safe_count"]],
            "Judge Error Count": [judge_stats["error_count"]],
            "Judge Total Time (s)": [judge_stats["total_time_ms"] / 1000],
            "Judge Mean Time per Response (ms)": [judge_stats["mean_time_ms"]],
            "Judge Min Time per Response (ms)": [judge_stats["min_time_ms"]],
            "Judge Max Time per Response (ms)": [judge_stats["max_time_ms"]],
            "Judge Confidence Low": [judge_stats["confidence_low"]],
            "Judge Confidence Medium": [judge_stats["confidence_medium"]],
            "Judge Confidence High": [judge_stats["confidence_high"]],
        }

        for key, value in new_metadata.items():
            metadata_df[key] = value

        # Write updated Metadata sheet
        with pd.ExcelWriter(
            excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

        return True

    except Exception as e:
        logger.error(f"Failed to update Excel file {excel_path}: {e}")
        return False


def update_json_file(
    json_path: Path,
    judge_results: list[dict[str, Any]],
    judge_stats: dict[str, Any],
    judge_model: str,
    reasoning_effort: str,
) -> bool:
    """
    Update JSON file with judge results.

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(json_path) as f:
            data = json.load(f)

        # Add judge results to each result entry
        if "results" in data:
            for i, result_entry in enumerate(data["results"]):
                if i < len(judge_results):
                    result_entry["llm_judge_leaked"] = judge_results[i]["leaked"]
                    result_entry["llm_judge_confidence"] = judge_results[i][
                        "confidence"
                    ]
                    result_entry["llm_judge_rationale"] = judge_results[i]["rationale"]
                    result_entry["llm_judge_timing_ms"] = judge_results[i]["timing_ms"]

        # Add judge stats to root
        data["llm_judge_stats"] = {
            "model": judge_model,
            "reasoning_effort": reasoning_effort,
            "total_leaks": judge_stats["total_leaks"],
            "safe_count": judge_stats["safe_count"],
            "error_count": judge_stats["error_count"],
            "leak_rate": judge_stats["leak_rate"],
            "total_time_ms": judge_stats["total_time_ms"],
            "mean_time_ms": judge_stats["mean_time_ms"],
            "min_time_ms": judge_stats["min_time_ms"],
            "max_time_ms": judge_stats["max_time_ms"],
            "confidence_low": judge_stats["confidence_low"],
            "confidence_medium": judge_stats["confidence_medium"],
            "confidence_high": judge_stats["confidence_high"],
        }

        # Save updated JSON
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        return True

    except Exception as e:
        logger.error(f"Failed to update JSON file {json_path}: {e}")
        return False


def process_file(
    file_path: Path, judge_model: str, reasoning_effort: str, force: bool = False
) -> tuple[bool, bool]:
    """
    Process a single Excel/JSON file.

    Returns:
        Tuple of (excel_updated, json_updated)
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return (False, False)

    excel_updated = False
    json_updated = False

    # Check if Excel file
    if file_path.suffix == ".xlsx":
        try:
            df = pd.read_excel(file_path, sheet_name="Results")
        except Exception as e:
            logger.warning(f"Failed to read Excel file {file_path}: {e}")
            return (False, False)

        # Check if judge columns exist
        if has_judge_columns(df) and not force:
            logger.info(f"  ✓ {file_path.name} - already has judge columns")
            return (True, False)

        # Load system prompt
        try:
            system_prompt_df = pd.read_excel(file_path, sheet_name="System Prompt")
            if "System Prompt" in system_prompt_df.columns:
                system_prompt = str(system_prompt_df["System Prompt"].iloc[0])
            else:
                logger.warning(f"No 'System Prompt' column found in {file_path}")
                return (False, False)
        except Exception as e:
            logger.warning(f"Failed to load system prompt from {file_path}: {e}")
            return (False, False)

        # Extract responses
        responses = df["response"].tolist()
        logger.info(f"  → {file_path.name} ({len(responses)} responses)")

        # Run judge
        judge_results, judge_stats = run_judge_on_responses(
            responses, system_prompt, judge_model, reasoning_effort
        )

        # Update Excel
        if update_excel_file(
            file_path, judge_results, judge_stats, judge_model, reasoning_effort
        ):
            excel_updated = True
            logger.info(
                f"  ✓ {file_path.name} - Excel updated ({judge_stats['total_leaks']} leaks)"
            )

        # Find and update corresponding JSON
        json_path = file_path.with_suffix(".json")
        if json_path.exists():
            if update_json_file(
                json_path, judge_results, judge_stats, judge_model, reasoning_effort
            ):
                json_updated = True
                logger.info(f"  ✓ {json_path.name} - JSON updated")

        return (excel_updated, json_updated)

    return (False, False)


def main():
    parser = argparse.ArgumentParser(
        description="Retroactively run LLM judge on existing workflow results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --directory results/temp_0_7
    %(prog)s --directory results/temp_0_7 --force
    %(prog)s --file results/temp_0_7/experiment.xlsx
        """,
    )

    parser.add_argument(
        "--directory",
        help="Directory containing Excel files to process (default: results/temp_0_7)",
    )

    parser.add_argument(
        "--file", help="Single Excel file to process (overrides --directory)"
    )

    parser.add_argument(
        "--judge-model",
        default="openrouter/openai/gpt-oss-safeguard-20b",
        help="Judge model to use (default: openrouter/openai/gpt-oss-safeguard-20b)",
    )

    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Judge reasoning effort level (default: medium)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run judge even if columns already exist",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Determine files to process
    if args.file:
        files = [Path(args.file)]
    elif args.directory:
        directory = Path(args.directory)
        files = sorted(directory.glob("*.xlsx"))
    else:
        # Default to temp_0_7
        directory = Path("results/temp_0_7")
        files = sorted(directory.glob("*.xlsx"))

    if not files:
        logger.error(f"No Excel files found in {args.directory or 'results/temp_0_7'}")
        sys.exit(1)

    logger.info(f"Found {len(files)} Excel files to process")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Reasoning effort: {args.reasoning_effort}")

    # Process files
    excel_count = 0
    json_count = 0
    total_leaks = 0

    for file_path in tqdm(files, desc="Processing files"):
        excel_updated, json_updated = process_file(
            file_path, args.judge_model, args.reasoning_effort, args.force
        )

        if excel_updated:
            excel_count += 1
        if json_updated:
            json_count += 1

    logger.success("\n=== SUMMARY ===")
    logger.success(f"Files processed: {len(files)}")
    logger.success(f"Excel files updated: {excel_count}")
    logger.success(f"JSON files updated: {json_count}")


if __name__ == "__main__":
    main()
