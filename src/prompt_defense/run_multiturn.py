#!/usr/bin/env python3
"""
CLI interface for running multi-turn prompt theft conversations between attacker and victim models.

Usage examples:
    python run_multiturn.py --attacker ollama/llama3.1 --victim openrouter/gpt-4
    python run_multiturn.py --attacker google/gemini-2.0-flash --victim ollama/llama3.1 --p-attack 0.7
"""

import argparse
import sys
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

from prompt_defense.multiturn.multiturn_theft import Session
from prompt_defense.utils.model_handler import ModelHandler
from prompt_defense.system_prompts.basic import SYSTEM_PROMPT


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run multi-turn prompt theft conversation"
    )
    parser.add_argument(
        "--attacker", required=True, help="Attacker model (e.g., ollama/llama3.1)"
    )
    parser.add_argument(
        "--victim", required=True, help="Victim model (e.g., openrouter/gpt-4)"
    )
    parser.add_argument(
        "--p-attack",
        type=float,
        default=0.5,
        help="Attack probability [0,1] (default: 0.5)",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="culip/qwen-embedding-0.6b-2",
        help="Maximum conversation turns (default: 10)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns (default: 10)",
    )
    parser.add_argument(
        "--delete-rejections",
        action="store_true",
        help="Exclude refusals from exports/history",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    if not 0 <= args.p_attack <= 1:
        logger.error("p-attack must be between 0 and 1")
        sys.exit(1)

    try:
        attacker_config = ModelHandler.create_model_config(
            args.attacker, embedding_model=args.embedding
        )
        victim_config = ModelHandler.create_model_config(
            args.victim, embedding_model=args.embedding
        )

        session = Session(
            attacker_model_config=attacker_config,
            victim_model_config=victim_config,
            p_attack=args.p_attack,
            max_turns=args.max_turns,
            delete_rejections=args.delete_rejections,
            victim_system_prompt=SYSTEM_PROMPT,
        )

        session.run()

        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        existing_files = list(results_dir.glob("multiturn_*.json"))
        existing_numbers = []
        for f in existing_files:
            try:
                num = int(f.stem.split("_")[1])
                existing_numbers.append(num)
            except (ValueError, IndexError):
                pass

        next_num = 1
        while next_num in existing_numbers:
            next_num += 1

        output_path = results_dir / f"multiturn_{next_num:02d}.json"
        session.export_results(str(output_path))
        logger.success(f"Results exported to {output_path}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
