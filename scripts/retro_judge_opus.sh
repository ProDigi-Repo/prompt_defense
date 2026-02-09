#!/bin/bash
# Run Sonnet 4.5 judge on existing workflow results (adds sonnet_prompt_leak column)

set -e

JUDGE_MODEL="openrouter/anthropic/claude-sonnet-4-5"
REASONING_EFFORT="medium"
WORKERS="10"
FORCE="false"
VERBOSE="false"

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
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CMD="uv run python src/prompt_defense/retro_judge_opus.py --judge-model $JUDGE_MODEL --reasoning-effort $REASONING_EFFORT --workers $WORKERS"

if [ "$FORCE" = "true" ]; then
    CMD="$CMD --force"
fi

if [ "$VERBOSE" = "true" ]; then
    CMD="$CMD --verbose"
fi

echo "=== Processing results/temp_0_0 ==="
( $CMD --directory results/temp_0_0)

echo ""
echo "=== Processing results/temp_0_7 ==="
( $CMD --directory results/temp_0_7)

echo ""
echo "Sonnet retro-judge complete!"
