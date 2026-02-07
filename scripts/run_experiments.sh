#!/bin/bash
MODELS=(
    "openai/gpt-3.5-turbo-instruct"
    "openai/gpt-5-chat"
    "openai/gpt-oss-120b"
    "mistralai/mistral-medium-3.1"
    "qwen/qwen3-235b-a22b-thinking-2507"
    "anthropic/claude-sonnet-4"
    "google/gemini-2.5-flash"
)

for MODEL in "${MODELS[@]}"; do
    echo "Running experiments for model: $MODEL"
    (cd ../ && uv run python src/prompt_defense/run_workflow.py --model openrouter/$MODEL --prompt-source theft_prompts --temperature 0.0 --verbose)
    (cd ../ && uv run python src/prompt_defense/run_workflow.py --model openrouter/$MODEL --prompt-source chat_prompts --temperature 0.0 --verbose)
    (cd ../ && uv run python src/prompt_defense/run_workflow.py --model openrouter/$MODEL --prompt-source paraphrased_theft_prompts --temperature 0.0 --verbose)
done
