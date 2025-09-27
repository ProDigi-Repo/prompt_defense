#!/usr/bin/env python3
"""
CLI interface for running prompt defense workflows with different models and metrics.

Usage examples:
    python run_workflow.py --model google/gemini-2.0-flash --metric embeddings
    python run_workflow.py --model ollama/llama3.1 --metric levenstein
    python run_workflow.py --model openrouter/gpt-4 --metric embeddings --embedding-model google/gemini-embedding-001
"""

import argparse
import os
import sys
import importlib
from pathlib import Path
from typing import List

from loguru import logger

from prompt_defense.utils.model_handler import ModelHandler
from prompt_defense.system_prompts.basic import SYSTEM_PROMPT


def load_prompts(source: str) -> List[str]:
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
        raise AttributeError(f"Prompt source '{source}' does not have 'prompts' attribute: {e}")


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
        "openrouter": ["OPENROUTER_API_KEY"]
    }

    for var in required_vars.get(provider, []):
        if not os.getenv(var):
            raise EnvironmentError(f"Required environment variable {var} is not set for provider '{provider}'")


def patch_workflow_prompts(workflow_module, prompts_list: List[str]):
    """
    Replace the prompts in the workflow module with our custom prompts.

    Args:
        workflow_module: The imported workflow module
        prompts_list: List of prompts to use

    Returns:
        Original prompts for restoration
    """
    original_prompts = getattr(workflow_module, 'prompts', None)
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
    if hasattr(workflow_module, 'generate_local_embeddings'):
        original_funcs['generate_local_embeddings'] = workflow_module.generate_local_embeddings
        workflow_module.generate_local_embeddings = embedding_func

    if hasattr(workflow_module, 'generate_embeddings_google'):
        original_funcs['generate_embeddings_google'] = workflow_module.generate_embeddings_google
        workflow_module.generate_embeddings_google = embedding_func

    if hasattr(workflow_module, 'embeddings_with_retry'):
        original_funcs['embeddings_with_retry'] = workflow_module.embeddings_with_retry
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


def run_workflow(
    model_config,
    metric: str,
    prompts_list: List[str],
    temperature: float = 0.7
) -> str:
    """
    Run the appropriate workflow using the model configuration.

    Args:
        model_config: ModelConfig with all model details
        metric: Metric type (embeddings, levenstein)
        prompts_list: List of attack prompts
        temperature: Model temperature

    Returns:
        Path to saved results file
    """
    # Determine which workflow module to import
    if model_config.provider == "google":
        if metric == "embeddings":
            workflow_name = "workflow_gemini_embeddings"
        else:  # levenstein
            workflow_name = "workflow_gemini_levensthein"
    else:  # ollama or openrouter (both use local workflows)
        if metric == "embeddings":
            workflow_name = "workflow_local_embeddings"
        else:  # levenstein
            workflow_name = "workflow_local_levensthein"

    logger.info(f"Running {workflow_name} with {model_config.provider} provider")

    try:
        # Import the workflow module
        workflow_module = importlib.import_module(f"prompt_defense.workflows.{workflow_name}")

        # Create the agent
        agent = ModelHandler.create_agent(model_config, SYSTEM_PROMPT, temperature)
        logger.info(f"Initialized {model_config.provider} agent with model: {model_config.model_name}")

        # For embedding workflows, also patch the embedding function
        original_embedding_funcs = {}
        if metric == "embeddings":
            logger.info(f"Using embedding model: {model_config.embedding_model}")
            embedding_func = ModelHandler.get_embedding_function(model_config)
            original_embedding_funcs = patch_workflow_embedding_function(workflow_module, embedding_func)

        # Replace prompts in the workflow module
        original_prompts = patch_workflow_prompts(workflow_module, prompts_list)

        # Run the workflow with our agent
        logger.info(f"Starting {workflow_name}...")
        workflow_module.main(agent=agent)

        # The workflow saves results internally, so we need to determine the saved path
        # This is a simplification - in reality we'd need to capture the return value
        # from the workflow if it returns the path
        results_dir = Path("results")
        saved_files = list(results_dir.glob(f"*{workflow_name.split('_')[1]}*.json"))
        if saved_files:
            # Get the most recently created file
            latest_file = max(saved_files, key=lambda p: p.stat().st_mtime)
            return str(latest_file)
        else:
            return f"results/{workflow_name}_results.json"

    except ImportError as e:
        raise ImportError(f"Could not import workflow '{workflow_name}': {e}")
    except Exception as e:
        raise Exception(f"Error running workflow '{workflow_name}': {e}")
    finally:
        # Restore the original module state
        if 'workflow_module' in locals():
            if 'original_prompts' in locals():
                restore_workflow_prompts(workflow_module, original_prompts)
            if 'original_embedding_funcs' in locals():
                restore_workflow_embedding_functions(workflow_module, original_embedding_funcs)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run prompt defense workflows with different models and metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model google/gemini-2.0-flash --metric embeddings
  %(prog)s --model ollama/llama3.1 --metric levenstein
  %(prog)s --model openrouter/x-ai/grok-4-fast:free --metric embeddings
  %(prog)s --model openrouter/gpt-4 --metric embeddings --embedding-model google/gemini-embedding-001
        """
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name with provider prefix (e.g., google/gemini-2.0-flash, ollama/llama3.1, openrouter/gpt-4)"
    )

    parser.add_argument(
        "--metric",
        required=True,
        choices=["embeddings", "levenstein"],
        help="Similarity metric to use"
    )

    parser.add_argument(
        "--embedding-model",
        help="Embedding model to use (e.g., google/gemini-embedding-001, ollama/embeddinggemma, nomic-ai/nomic-embed-text-v1.5). "
             "Defaults to provider-specific models: google→gemini-embedding-001, ollama→embeddinggemma, openrouter→nomic-ai/nomic-embed-text-v1.5."
    )

    parser.add_argument(
        "--prompt-source",
        default="generated",
        choices=["generated", "selected", "manually", "gemini_generated"],
        help="Attack prompt source to use (default: generated)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Model temperature (default: 0.7)"
    )

    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)"
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

    try:
        # Create model configuration
        model_config = ModelHandler.create_model_config(
            model_string=args.model,
            embedding_model=args.embedding_model
        )

        logger.info(f"Using provider: {model_config.provider}, model: {model_config.model_name}")
        if args.metric == "embeddings":
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
            metric=args.metric,
            prompts_list=prompts_list,
            temperature=args.temperature
        )

        logger.success(f"Workflow completed successfully. Results saved to: {saved_path}")

    except (ValueError, ImportError, AttributeError, EnvironmentError) as e:
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