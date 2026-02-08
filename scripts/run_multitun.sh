#!/bin/bash
MODELS=(
    "openai/gpt-3.5-turbo-instruct"
    "openai/gpt-5-chat"
    "openai/gpt-oss-120b"
    "mistralai/mistral-medium-3.1"
    "qwen/qwen3-235b-a22b-thinking-2507"
    "anthropic/claude-sonnet-4"
    "google/gemini-2.5-flash"
    "cohere/command-a"
)

for MODEL in "${MODELS[@]}"; do
    echo "Running experiments for model: $MODEL"
    (cd ../ && uv run src/prompt_defense/run_multiturn.py --attacker culip/gpt-oss-120b-heretic --victim openrouter/$MODEL --embedding=culip/qwen3-embedding-0.6b-2 --p-attack 0.75 --max-turns 5 --victim-temperature 0.0)
    (cd ../ && uv run src/prompt_defense/run_multiturn.py --attacker culip/gpt-oss-120b-heretic --victim openrouter/$MODEL --embedding=culip/qwen3-embedding-0.6b-2 --p-attack 0.75 --max-turns 5 --victim-temperature 0.0)
done
