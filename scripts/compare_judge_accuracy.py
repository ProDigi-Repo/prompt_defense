"""Compare LLM judge accuracy against human annotations."""

import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from loguru import logger


def find_annotation_pairs(human_dir: Path, judge_dir: Path) -> dict[str, Any]:
    """Find matching pairs of human annotation and LLM judge files.

    Args:
        human_dir: Directory containing human annotation Excel files
        judge_dir: Directory containing LLM judge Excel files

    Returns:
        Dict with:
            - pairs: List of dicts with keys: human_file, judge_file, model_name
            - unmatched_human: List of human files without matching judge file
            - unmatched_judge: List of judge files without matching human file
    """
    pairs = []
    unmatched_human = []

    # Find all human annotation files
    human_files = list(human_dir.glob("*.xlsx"))

    # Filter to only include theft-related files (skip chat prompts)
    human_files = [
        f
        for f in human_files
        if ("paraphrased_theft" in f.name or "theft_prompts" in f.name)
        and "combined_chat_prompts" not in f.name
    ]

    for human_file in human_files:
        # Try to find corresponding judge file with same name
        judge_file = judge_dir / human_file.name

        if judge_file.exists():
            # Extract model name from filename (before first underscore or 'combined')
            filename = human_file.stem
            model_name = (
                filename.split("_combined")[0]
                if "_combined" in filename
                else filename.split("_")[0]
            )

            pairs.append(
                {
                    "human_file": human_file,
                    "judge_file": judge_file,
                    "model_name": model_name,
                }
            )
        else:
            unmatched_human.append(human_file)

    # Find judge files without matching human files
    judge_files = list(judge_dir.glob("*.xlsx"))
    human_file_names = {f.name for f in human_files}
    unmatched_judge = [f for f in judge_files if f.name not in human_file_names]

    return {
        "pairs": pairs,
        "unmatched_human": unmatched_human,
        "unmatched_judge": unmatched_judge,
    }


def load_and_validate_pair(
    human_file: Path, judge_file: Path, judge_column: str = "judge_column"
) -> dict:
    """Load and validate a pair of human annotation and judge files.

    Args:
        human_file: Path to human annotation Excel file
        judge_file: Path to LLM judge Excel file

    Returns:
        Dict with:
            - human_labels: boolean array (True = leaked)
            - judge_labels: boolean array (True = leaked)
            - valid_rows: number of valid rows processed

    Raises:
        ValueError: If files have mismatched row counts or missing columns
    """
    # Load both files
    human_df = pd.read_excel(human_file)
    judge_df = pd.read_excel(judge_file)

    # Validate that both files have the 'input' column for alignment
    if "input" not in human_df.columns:
        raise ValueError(
            f"Human file missing 'input' column for row alignment: {human_file}"
        )
    if "input" not in judge_df.columns:
        raise ValueError(
            f"Judge file missing 'input' column for row alignment: {judge_file}"
        )

    # Verify rows are aligned by checking if inputs match
    if not human_df["input"].equals(judge_df["input"]):
        # Try to align by sorting both dataframes by input
        logger.warning(
            f"Row order mismatch detected for {human_file.name}, attempting to align by input..."
        )

        # Create a copy with index reset
        human_df = human_df.sort_values("input").reset_index(drop=True)
        judge_df = judge_df.sort_values("input").reset_index(drop=True)

        # Check if they match after sorting
        if not human_df["input"].equals(judge_df["input"]):
            # Count how many match
            matching_inputs = set(human_df["input"]) & set(judge_df["input"])
            raise ValueError(
                f"Cannot align files - inputs don't match even after sorting.\n"
                f"Human file: {human_file.name}\n"
                f"Judge file: {judge_file.name}\n"
                f"Human rows: {len(human_df)}, Judge rows: {len(judge_df)}\n"
                f"Matching inputs: {len(matching_inputs)}"
            )
        else:
            logger.info(f"Successfully aligned rows for {human_file.name} by sorting")

    # Validate required columns exist - try multiple possible column names (case-insensitive)
    human_column_names = [
        "Human Eval Prompt Leak",
        "Humal Eval Prompt Leak",
        "Human Eval Success Attack",
        "human eval success attack",
        "human_annotation",
        "human eval",
    ]
    human_column = None

    # Create a lowercase mapping of actual columns
    human_columns_lower = {col.lower(): col for col in human_df.columns}

    for col_name in human_column_names:
        if col_name.lower() in human_columns_lower:
            human_column = human_columns_lower[col_name.lower()]
            break

    if human_column is None:
        raise ValueError(
            f"Human file missing required column. Tried (case-insensitive): {human_column_names}. "
            f"Available columns: {list(human_df.columns)}"
        )

    if judge_column not in judge_df.columns:
        raise ValueError(f"Judge file missing '{judge_column}' column: {judge_file}")

    # Check row counts match
    if len(human_df) != len(judge_df):
        raise ValueError(
            f"Row count mismatch: human={len(human_df)}, judge={len(judge_df)} for {human_file.name}"
        )

    # Extract labels and convert to boolean
    # Human: "y" = True (leaked), "n" = False (not leaked)
    human_labels = (human_df[human_column].str.lower() == "y").values

    # Judge: already boolean True/False
    judge_labels = judge_df[judge_column].values.astype(bool)

    # Filter out rows where either is NaN
    valid_mask = ~(pd.isna(human_df[human_column]) | pd.isna(judge_df[judge_column]))
    human_labels = human_labels[valid_mask]
    judge_labels = judge_labels[valid_mask]

    return {
        "human_labels": human_labels,
        "judge_labels": judge_labels,
        "valid_rows": int(valid_mask.sum()),
    }


def calculate_metrics(human_labels: np.ndarray, judge_labels: np.ndarray) -> dict:
    """Calculate accuracy metrics comparing judge predictions to human labels.

    Args:
        human_labels: Ground truth labels (boolean array)
        judge_labels: Predicted labels from LLM judge (boolean array)

    Returns:
        Dict containing:
            - accuracy: Overall accuracy (0-1)
            - precision: Precision for leaked class (0-1)
            - recall: Recall for leaked class (0-1)
            - f1_score: F1 score for leaked class (0-1)
            - confusion_matrix: 2x2 array [[TN, FP], [FN, TP]]
            - true_positives: Count of TP
            - true_negatives: Count of TN
            - false_positives: Count of FP
            - false_negatives: Count of FN
    """
    # Calculate metrics
    acc = accuracy_score(human_labels, judge_labels)

    # Use zero_division=0 to handle edge cases
    prec = precision_score(human_labels, judge_labels, zero_division=0)
    rec = recall_score(human_labels, judge_labels, zero_division=0)
    f1 = f1_score(human_labels, judge_labels, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(human_labels, judge_labels)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": cm,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": len(human_labels),
    }


def process_all_pairs(
    human_dir: Path, judge_dir: Path, judge_column: str = "llm_judge_leaked"
) -> dict[str, Any]:
    """Process all matching file pairs and aggregate results.

    Args:
        human_dir: Directory containing human annotation files
        judge_dir: Directory containing LLM judge files

    Returns:
        Dict with:
            - per_model_metrics: Dict[model_name, metrics]
            - overall_metrics: Aggregated metrics across all models
            - file_pairs_processed: Number of file pairs processed
            - unmatched_human_count: Number of human files without matching judge file
            - unmatched_judge_count: Number of judge files without matching human file
            - unmatched_human_files: List of unmatched human file names
            - unmatched_judge_files: List of unmatched judge file names
    """
    result = find_annotation_pairs(human_dir, judge_dir)
    pairs = result["pairs"]
    unmatched_human = result["unmatched_human"]
    unmatched_judge = result["unmatched_judge"]

    if not pairs:
        raise ValueError(
            f"No matching file pairs found between {human_dir} and {judge_dir}"
        )

    logger.info(f"Found {len(pairs)} file pairs to process")

    # Log unmatched files information
    if unmatched_human:
        logger.warning(
            f"Found {len(unmatched_human)} human files without matching judge files"
        )
        for f in unmatched_human:
            logger.warning(f"  - {f.name}")

    if unmatched_judge:
        logger.warning(
            f"Found {len(unmatched_judge)} judge files without matching human files"
        )
        for f in unmatched_judge:
            logger.warning(f"  - {f.name}")

    per_model_metrics = {}
    all_human_labels = []
    all_judge_labels = []
    skipped_pairs = []

    for pair in pairs:
        model_name = pair["model_name"]
        logger.info(f"Processing {model_name}...")

        try:
            # Load and validate data
            data = load_and_validate_pair(
                pair["human_file"], pair["judge_file"], judge_column
            )

            # Calculate metrics for this model
            metrics = calculate_metrics(data["human_labels"], data["judge_labels"])
            metrics["valid_rows"] = data["valid_rows"]
            metrics["human_file"] = str(pair["human_file"].name)
            metrics["judge_file"] = str(pair["judge_file"].name)

            per_model_metrics[model_name] = metrics

            # Accumulate for overall metrics
            all_human_labels.extend(data["human_labels"])
            all_judge_labels.extend(data["judge_labels"])

            logger.info(
                f"  Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}"
            )

        except ValueError as e:
            # Skip files that can't be aligned or validated
            if (
                "Cannot align files" in str(e)
                or "missing required column" in str(e)
                or "Row count mismatch" in str(e)
            ):
                logger.warning(f"Skipping {model_name}: {e}")
                skipped_pairs.append(
                    {
                        "model_name": model_name,
                        "reason": str(e),
                        "human_file": str(pair["human_file"].name),
                        "judge_file": str(pair["judge_file"].name),
                    }
                )
                continue
            else:
                # Re-raise unexpected ValueError
                raise
        except Exception as e:
            logger.error(f"Error processing {model_name}: {e}")
            continue

    # Calculate overall metrics across all models
    overall_metrics = calculate_metrics(
        np.array(all_human_labels), np.array(all_judge_labels)
    )

    return {
        "per_model_metrics": per_model_metrics,
        "overall_metrics": overall_metrics,
        "file_pairs_processed": len(per_model_metrics),
        "skipped_pairs": skipped_pairs,
        "skipped_count": len(skipped_pairs),
        "unmatched_human_count": len(unmatched_human),
        "unmatched_judge_count": len(unmatched_judge),
        "unmatched_human_files": [f.name for f in unmatched_human],
        "unmatched_judge_files": [f.name for f in unmatched_judge],
    }


def format_metrics_table(metrics: dict[str, Any]) -> str:
    """Format metrics as a readable table.

    Args:
        metrics: Metrics dictionary

    Returns:
        Formatted string table
    """
    lines = []
    lines.append(f"{'Metric':<20} {'Value':<10}")
    lines.append("-" * 30)
    lines.append(f"{'Accuracy':<20} {metrics['accuracy']:>7.1%}")
    lines.append(f"{'Precision':<20} {metrics['precision']:>7.1%}")
    lines.append(f"{'Recall':<20} {metrics['recall']:>7.1%}")
    lines.append(f"{'F1 Score':<20} {metrics['f1_score']:>7.1%}")
    lines.append("-" * 30)
    lines.append(f"{'True Positives':<20} {metrics['true_positives']:>7}")
    lines.append(f"{'True Negatives':<20} {metrics['true_negatives']:>7}")
    lines.append(f"{'False Positives':<20} {metrics['false_positives']:>7}")
    lines.append(f"{'False Negatives':<20} {metrics['false_negatives']:>7}")
    lines.append(f"{'Total Samples':<20} {metrics['total_samples']:>7}")

    return "\n".join(lines)


def save_results(results: dict[str, Any], output_file: Path) -> None:
    """Save results to JSON file.

    Args:
        results: Results dictionary
        output_file: Output JSON file path
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}

    for key, value in results.items():
        if key == "per_model_metrics":
            serializable_results[key] = {}
            for model, metrics in value.items():
                serializable_results[key][model] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metrics.items()
                }
        elif key == "overall_metrics":
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value

    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare LLM judge accuracy against human annotations"
    )
    parser.add_argument(
        "--human-dir",
        type=Path,
        required=True,
        help="Directory containing human annotation Excel files",
    )
    parser.add_argument(
        "--judge-dir",
        type=Path,
        required=True,
        help="Directory containing LLM judge Excel files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/judge_accuracy_comparison.json"),
        help="Output JSON file path (default: results/judge_accuracy_comparison.json)",
    )
    parser.add_argument(
        "--judge-column",
        default="llm_judge_leaked",
        help="Column name for judge predictions (default: llm_judge_leaked)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show per-model detailed metrics"
    )

    args = parser.parse_args()

    # Validate directories exist
    if not args.human_dir.exists():
        logger.error(f"Human annotation directory not found: {args.human_dir}")
        return 1

    if not args.judge_dir.exists():
        logger.error(f"Judge directory not found: {args.judge_dir}")
        return 1

    # Process all pairs
    logger.info("Starting judge accuracy comparison...")
    logger.info(f"  Human annotations: {args.human_dir}")
    logger.info(f"  LLM judge results: {args.judge_dir}")

    results = process_all_pairs(args.human_dir, args.judge_dir, args.judge_column)

    # Print overall results
    print("\n" + "=" * 50)
    print("OVERALL METRICS (Across All Models)")
    print("=" * 50)
    print(format_metrics_table(results["overall_metrics"]))

    # Print per-model results if verbose
    if args.verbose:
        print("\n" + "=" * 50)
        print("PER-MODEL METRICS")
        print("=" * 50)
        for model_name, metrics in results["per_model_metrics"].items():
            print(f"\n{model_name}")
            print("-" * 30)
            print(format_metrics_table(metrics))

    # Print unmatched files summary
    print("\n" + "=" * 50)
    print("FILE MATCHING SUMMARY")
    print("=" * 50)
    print(f"Matched pairs: {results['file_pairs_processed']}")
    print(f"Unmatched human files: {results['unmatched_human_count']}")
    print(f"Unmatched judge files: {results['unmatched_judge_count']}")

    # Show skipped pairs if any
    if results.get("skipped_count", 0) > 0:
        print(
            f"\n⚠ Skipped {results['skipped_count']} file pair(s) due to alignment/validation issues"
        )
        if args.verbose or results["skipped_count"] > 0:
            for skipped in results["skipped_pairs"]:
                print(f"  - {skipped['model_name']}: {skipped['reason'][:100]}...")

    # Show unmatched file names if verbose or if there are any unmatched files
    if (
        args.verbose
        or results["unmatched_human_count"] > 0
        or results["unmatched_judge_count"] > 0
    ):
        if results["unmatched_human_files"]:
            print("\nUnmatched human files:")
            for filename in results["unmatched_human_files"]:
                print(f"  - {filename}")

        if results["unmatched_judge_files"]:
            print("\nUnmatched judge files:")
            for filename in results["unmatched_judge_files"]:
                print(f"  - {filename}")

    # Save to file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)

    print(f"\n✓ Processed {results['file_pairs_processed']} file pairs")
    print(f"✓ Results saved to {args.output}")

    # Display final summary with combined accuracy
    print("\n" + "=" * 50)
    print("FINAL SUMMARY - COMBINED ACCURACY")
    print("=" * 50)
    print(f"Overall Accuracy: {results['overall_metrics']['accuracy']:>7.1%}")
    print(f"Overall Precision: {results['overall_metrics']['precision']:>6.1%}")
    print(f"Overall Recall: {results['overall_metrics']['recall']:>9.1%}")
    print(f"Overall F1 Score: {results['overall_metrics']['f1_score']:>7.1%}")
    print(f"Total Samples: {results['overall_metrics']['total_samples']:>11}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    exit(main())
