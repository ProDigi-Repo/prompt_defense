#!/bin/bash
# Retroactively run LLM judge on existing workflow results

set -e

# Configuration
JUDGE_MODEL="openrouter/openai/gpt-oss-safeguard-20b"
REASONING_EFFORT="medium"
FORCE="false"
VERBOSE="false"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE="true"
            shift
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        --model)
            JUDGE_MODEL="$2"
            shift 2
            ;;
        --reasoning-effort)
            REASONING_EFFORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="uv run python src/prompt_defense/retro_judge.py --judge-model $JUDGE_MODEL --reasoning-effort $REASONING_EFFORT"

if [ "$FORCE" = "true" ]; then
    CMD="$CMD --force"
fi

if [ "$VERBOSE" = "true" ]; then
    CMD="$CMD --verbose"
fi

# Process temp directories
echo "=== Processing results/temp_0_0 ==="
(cd /home/mfm/code/prompt_defense && $CMD --directory results/temp_0_0)

echo ""
echo "=== Processing results/temp_0_7 ==="
(cd /home/mfm/code/prompt_defense && $CMD --directory results/temp_0_7)

echo ""
echo "=== Processing results/temp_0 ==="
(cd /home/mfm/code/prompt_defense && $CMD --directory results/temp_0)

echo ""
echo "✅ Retro-judge processing complete!"
