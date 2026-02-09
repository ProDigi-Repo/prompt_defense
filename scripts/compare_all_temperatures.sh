#!/bin/bash
# Compare judge accuracy for all temperature settings

set -e

echo "Comparing LLM Judge Accuracy Against Human Annotations"
echo "======================================================="

# --- GPT-OSS Safeguard Judge (llm_judge_leaked) ---

echo ""
echo "=== GPT-OSS Safeguard Judge ==="

echo ""
echo "Processing temperature 0.7..."
python scripts/compare_judge_accuracy.py \
    --human-dir "humanannotation/run 2/temp 0.7" \
    --judge-dir "results/temp_0_7" \
    --output "results/judge_accuracy_temp_0_7.json" \
    --verbose

echo ""
echo "Processing temperature 0.0..."
python scripts/compare_judge_accuracy.py \
    --human-dir "humanannotation/run 2/temp 0_0" \
    --judge-dir "results/temp_0_0" \
    --output "results/judge_accuracy_temp_0_0.json" \
    --verbose

# --- Sonnet 4.5 Judge (sonnet_prompt_leak) ---

echo ""
echo "=== Sonnet 4.5 Judge ==="

echo ""
echo "Processing temperature 0.7 (Sonnet)..."
python scripts/compare_judge_accuracy.py \
    --human-dir "humanannotation/run 2/temp 0.7" \
    --judge-dir "results/temp_0_7" \
    --judge-column "sonnet_prompt_leak" \
    --output "results/judge_accuracy_sonnet_temp_0_7.json" \
    --verbose

echo ""
echo "Processing temperature 0.0 (Sonnet)..."
python scripts/compare_judge_accuracy.py \
    --human-dir "humanannotation/run 2/temp 0_0" \
    --judge-dir "results/temp_0_0" \
    --judge-column "sonnet_prompt_leak" \
    --output "results/judge_accuracy_sonnet_temp_0_0.json" \
    --verbose

echo ""
echo "All comparisons complete!"
echo "Results:"
echo "  GPT-OSS Safeguard:"
echo "    - Temperature 0.7: results/judge_accuracy_temp_0_7.json"
echo "    - Temperature 0.0: results/judge_accuracy_temp_0_0.json"
echo "  Sonnet 4.5:"
echo "    - Temperature 0.7: results/judge_accuracy_sonnet_temp_0_7.json"
echo "    - Temperature 0.0: results/judge_accuracy_sonnet_temp_0_0.json"
