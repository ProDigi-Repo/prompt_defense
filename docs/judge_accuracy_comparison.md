# LLM Judge Accuracy Comparison

This tool compares the accuracy of LLM-as-a-judge predictions against human annotations (ground truth) for prompt leak detection.

## Overview

The script matches human annotation files with corresponding LLM judge result files, extracts the relevant classification columns, and calculates standard classification metrics.

## File Structure

**Human annotations:** `humanannotation/run 2/temp <X>/`
- Column: `Humal Eval Prompt Leak` (values: "y" = leaked, "n" = not leaked)

**LLM judge results:** `results/temp_<X>/`
- Column: `llm_judge_leaked` (values: True/False)

Files are matched by model name and prompt type.

## Usage

### Single temperature comparison

```bash
python scripts/compare_judge_accuracy.py \
    --human-dir "humanannotation/run 2/temp 0.7" \
    --judge-dir results/temp_0_7 \
    --output results/judge_accuracy_temp_0_7.json \
    --verbose
```

### All temperatures

```bash
./scripts/compare_all_temperatures.sh
```

## Metrics Reported

- **Accuracy**: Overall percentage of correct classifications
- **Precision**: Of all predicted leaks, how many were actually leaks?
- **Recall**: Of all actual leaks, how many did we detect?
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: TP, TN, FP, FN counts

## Output Format

Results are saved as JSON with:
- `per_model_metrics`: Metrics for each model individually
- `overall_metrics`: Aggregated metrics across all models
- `file_pairs_processed`: Number of file pairs analyzed

## Example Output

```
==================================================
OVERALL METRICS (Across All Models)
==================================================
Metric               Value
------------------------------
Accuracy                92.3%
Precision               89.5%
Recall                  94.2%
F1 Score                91.8%
------------------------------
True Positives             145
True Negatives             312
False Positives             17
False Negatives              9
Total Samples              483
```

## Testing

Run tests with:

```bash
pytest tests/test_judge_accuracy.py -v
```

## Dependencies

All required packages are in `pyproject.toml`:
- pandas
- scikit-learn
- openpyxl
- numpy
