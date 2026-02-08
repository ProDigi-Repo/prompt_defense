#!/usr/bin/env python3
"""
CLI interface for running prompt defense workflows with different models and metrics.

Usage examples:
    python run_workflow.py --model google/gemini-2.0-flash
    python run_workflow.py --model ollama/llama3.1
    python run_workflow.py --model openrouter/x-ai/grok-4-fast:free --embedding-model google/gemini-embedding-001
"""

import argparse
import os
import sys
import importlib
from pathlib import Path

from loguru import logger

from prompt_defense.utils.model_handler import ModelHandler
from prompt_defense.system_prompts.basic import SYSTEM_PROMPT
import re
import nltk

nltk.download("punkt_tab")


def discover_prompt_sources() -> list[str]:
    """
    Dynamically discover available prompt sources from the attack_prompts directory.

    Returns:
        List of available prompt source names (without .py extension)
    """
    # Get the path to the attack_prompts directory
    attack_prompts_dir = Path(__file__).parent / "attack_prompts"

    prompt_sources = []

    # Find all .py files in the attack_prompts directory
    for file_path in attack_prompts_dir.glob("*.py"):
        # Skip __init__.py and any files starting with underscore
        if file_path.name.startswith("_"):
            continue

        # Remove .py extension to get the module name
        module_name = file_path.stem

        # Try to import and check if it has a 'prompts' attribute
        try:
            module = importlib.import_module(
                f"prompt_defense.attack_prompts.{module_name}"
            )
            if hasattr(module, "prompts"):
                prompt_sources.append(module_name)
        except ImportError:
            # Skip files that can't be imported
            continue

    return sorted(prompt_sources)


def load_prompts(source: str) -> list[str]:
    """
    Dynamically load prompts from specified source.

    Args:
        source: Name of prompt source module (generated, selected, manually, gemini_generated)

    Returns:
        List of attack prompts

    Raises:
        ImportError: If prompt source module cannot be imported
        AttributeError: If prompts attribute is not found in module
    """
    try:
        module = importlib.import_module(f"prompt_defense.attack_prompts.{source}")
        prompts = getattr(module, "prompts")
        logger.info(f"Loaded {len(prompts)} prompts from {source}")
        return prompts
    except ImportError as e:
        raise ImportError(f"Could not import prompt source '{source}': {e}")
    except AttributeError as e:
        raise AttributeError(
            f"Prompt source '{source}' does not have 'prompts' attribute: {e}"
        )


def validate_environment(provider: str) -> None:
    """
    Validate required environment variables for the provider.

    Args:
        provider: Model provider name

    Raises:
        EnvironmentError: If required environment variables are missing
    """
    required_vars = {
        "google": ["GEMINI_API_KEY"],
        "ollama": [],  # Local model, no API key required
        "openrouter": ["OPENROUTER_API_KEY"],
    }

    for var in required_vars.get(provider, []):
        if not os.getenv(var):
            raise OSError(
                f"Required environment variable {var} is not set for provider '{provider}'"
            )


def patch_workflow_prompts(workflow_module, prompts_list: list[str]):
    """
    Replace the prompts in the workflow module with our custom prompts.

    Args:
        workflow_module: The imported workflow module
        prompts_list: List of prompts to use

    Returns:
        Original prompts for restoration
    """
    original_prompts = getattr(workflow_module, "prompts", None)
    workflow_module.prompts = prompts_list
    return original_prompts


def restore_workflow_prompts(workflow_module, original_prompts):
    """
    Restore the original prompts in the workflow module.

    Args:
        workflow_module: The workflow module to restore
        original_prompts: Original prompts to restore
    """
    if original_prompts is not None:
        workflow_module.prompts = original_prompts


def patch_workflow_embedding_function(workflow_module, embedding_func):
    """
    Replace the embedding function in the workflow module.

    Args:
        workflow_module: The imported workflow module
        embedding_func: Embedding function to use

    Returns:
        Original embedding functions for restoration
    """
    original_funcs = {}

    # Store and replace embedding functions based on workflow type
    if hasattr(workflow_module, "generate_local_embeddings"):
        original_funcs["generate_local_embeddings"] = (
            workflow_module.generate_local_embeddings
        )
        workflow_module.generate_local_embeddings = embedding_func

    if hasattr(workflow_module, "generate_embeddings_google"):
        original_funcs["generate_embeddings_google"] = (
            workflow_module.generate_embeddings_google
        )
        workflow_module.generate_embeddings_google = embedding_func

    if hasattr(workflow_module, "embeddings_with_retry"):
        original_funcs["embeddings_with_retry"] = workflow_module.embeddings_with_retry
        workflow_module.embeddings_with_retry = embedding_func

    return original_funcs


def restore_workflow_embedding_functions(workflow_module, original_funcs):
    """
    Restore the original embedding functions in the workflow module.

    Args:
        workflow_module: The workflow module to restore
        original_funcs: Original functions to restore
    """
    for func_name, original_func in original_funcs.items():
        setattr(workflow_module, func_name, original_func)


def extract_model_name_for_filename(model_string: str) -> str:
    """
    Extract and sanitize model name from full model string for use in filenames.

    Args:
        model_string: Full model string like "openrouter/x-ai/grok-4-fast:free"

    Returns:
        Sanitized model name like "grok_4_fast_free"
    """
    # Extract everything after the last slash
    model_name = model_string.split("/")[-1]

    # Replace special characters (spaces, dashes, colons, dots) with underscores
    sanitized_name = re.sub(r"[-:\s.]+", "_", model_name)

    # Remove any trailing underscores
    sanitized_name = sanitized_name.strip("_")

    return sanitized_name


def run_workflow(
    model_config,
    prompts_list: list[str],
    prompt_source: str,
    temperature: float = 0.7,
    enable_judge: bool = False,
    judge_model: str | None = None,
    judge_reasoning_effort: str | None = None,
) -> str:
    """
    Run the appropriate combined workflow using the model configuration.

    Args:
        model_config: ModelConfig with all model details
        prompts_list: List of attack prompts
        prompt_source: Name of the prompt source (for filename)
        temperature: Model temperature
        enable_judge: Enable LLM judge for prompt leak detection
        judge_model: Judge model to use
        judge_reasoning_effort: Judge reasoning effort level

    Returns:
        Path to saved results file
    """
    # Determine which workflow module to import
    if model_config.provider == "google":
        workflow_name = "workflow_gemini_combined"
    else:  # ollama or openrouter (both use local workflows)
        workflow_name = "workflow_local_combined"

    logger.info(f"Running {workflow_name} with {model_config.provider} provider")

    # Extract model name for filename
    model_name_for_file = extract_model_name_for_filename(model_config.model_name)
    logger.info(f"Using model name for output files: {model_name_for_file}")

    try:
        # Import the workflow module
        workflow_module = importlib.import_module(
            f"prompt_defense.combined_workflows.{workflow_name}"
        )

        # Create the agent
        agent = ModelHandler.create_agent(model_config, SYSTEM_PROMPT, temperature)
        logger.info(
            f"Initialized {model_config.provider} agent with model: {model_config.model_name}"
        )

        # For combined workflows, patch embedding function based on provider
        original_embedding_funcs = {}
        logger.info(f"Using embedding model: {model_config.embedding_model}")
        embedding_func = ModelHandler.get_embedding_function(model_config)
        original_embedding_funcs = patch_workflow_embedding_function(
            workflow_module, embedding_func
        )

        # Replace prompts in the workflow module
        original_prompts = patch_workflow_prompts(workflow_module, prompts_list)

        # Run the workflow with our agent, model name, and prompt source
        logger.info(f"Starting {workflow_name}...")
        result_path = workflow_module.main(
            agent=agent,
            model_name=model_name_for_file,
            prompt_source=prompt_source,
            enable_judge=enable_judge,
            judge_model=judge_model,
            judge_reasoning_effort=judge_reasoning_effort,
        )

        # Return the result path from the workflow
        return result_path

    except ImportError as e:
        raise ImportError(f"Could not import workflow '{workflow_name}': {e}")
    except Exception as e:
        raise Exception(f"Error running workflow '{workflow_name}': {e}")
    finally:
        # Restore the original module state
        if "workflow_module" in locals():
            if "original_prompts" in locals():
                restore_workflow_prompts(workflow_module, original_prompts)
            if "original_embedding_funcs" in locals():
                restore_workflow_embedding_functions(
                    workflow_module, original_embedding_funcs
                )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run prompt defense workflows with different models and metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model google/gemini-2.0-flash
  %(prog)s --model ollama/llama3.1
  %(prog)s --model openrouter/x-ai/grok-4-fast:free
  %(prog)s --model openrouter/gpt-4 --embedding-model google/gemini-embedding-001
        """,
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name with provider prefix (e.g., google/gemini-2.0-flash, ollama/llama3.1, openrouter/gpt-4)",
    )

    parser.add_argument(
        "--embedding-model",
        help="Embedding model to use (e.g., google/gemini-embedding-001, ollama/embeddinggemma, nomic-ai/nomic-embed-text-v1.5). "
        "Defaults to provider-specific models: google→gemini-embedding-001, ollama→embeddinggemma, openrouter→nomic-ai/nomic-embed-text-v1.5.",
    )

    # Discover available prompt sources dynamically
    available_prompt_sources = discover_prompt_sources()
    default_prompt_source = (
        "generated"
        if "generated" in available_prompt_sources
        else available_prompt_sources[0]
    )

    parser.add_argument(
        "--prompt-source",
        default=default_prompt_source,
        choices=available_prompt_sources,
        help=f"Attack prompt source to use. Available: {', '.join(available_prompt_sources)} (default: {default_prompt_source})",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Model temperature (default: 0.7)",
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--enable-judge",
        action="store_true",
        help="Enable LLM judge for prompt leak detection",
    )

    parser.add_argument(
        "--judge-model",
        default="openrouter/openai/gpt-oss-safeguard-20b",
        help="Judge model to use for leak detection (default: openrouter/openai/gpt-oss-safeguard-20b)",
    )

    parser.add_argument(
        "--judge-reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Judge reasoning effort level (default: medium)",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    try:
        # Create model configuration
        model_config = ModelHandler.create_model_config(
            model_string=args.model, embedding_model=args.embedding_model
        )

        logger.info(
            f"Using provider: {model_config.provider}, model: {model_config.model_name}"
        )
        logger.info(f"Using embedding model: {model_config.embedding_model}")

        # Validate environment
        validate_environment(model_config.provider)

        # Load prompts
        prompts_list = load_prompts(args.prompt_source)

        # Ensure output directory exists
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Run the workflow
        saved_path = run_workflow(
            model_config=model_config,
            prompts_list=prompts_list,
            prompt_source=args.prompt_source,
            temperature=args.temperature,
            enable_judge=args.enable_judge,
            judge_model=args.judge_model,
            judge_reasoning_effort=args.judge_reasoning_effort,
        )

        logger.success(
            f"Workflow completed successfully. Results saved to: {saved_path}"
        )

    except (ValueError, ImportError, AttributeError, OSError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
