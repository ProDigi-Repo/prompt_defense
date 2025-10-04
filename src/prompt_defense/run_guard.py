#!/usr/bin/env python3
"""
CLI interface for running {Qwen, Llama}-Guard 
"""

import argparse
import sys
import tqdm
import json
import datetime

from loguru import logger

from .guard.llama import LlamaGuard
from .guard.qwen import QwenGuard
from .attack_prompts.theft_prompts import prompts

def main():
    parser = argparse.ArgumentParser(
        description="Run prompt guard on attack prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --model qwen
    %(prog)s --model llama
"""
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=["llama", "qwen"]
        help="The guard model class to use."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # Create guard class and start executing
    logger.info("Initializing model...")
    guard = None
    if args.model == "qwen":
        guard = QwenGuard()
    elif args.model == "llama":
        guard = LlamaGuard()

    results = {}

    for prompt in tqdm(prompts, desc="Processing prompts"):
        decision, prob = guard.detect_jailbreak(prompt)
        results.append([
            {
                "prompt": prompt,
                "malicions": decision,
                "prob": prob
            }
        ])

    logger.info("Storing results as JSON...")

    timestamp = datetime.now().isoformat()
    with open(f"results/{args.model}_guard_{timestamp}.json", "w") as f:
        f.write(json.dumps({
            "model": args.model,
            "has_probs": guard.has_probs(),
            "results": results
        }))

    logger.info("Done!")


if __name__ == "__main__":
    main()