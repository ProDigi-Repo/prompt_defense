"""
ModelHandler for managing both main models and embedding models across different providers.
"""

from typing import Callable, Tuple, Optional
from dataclasses import dataclass

from prompt_defense.utils.init_gemini import init_gemini_agent
from prompt_defense.utils.init_openai import initialize_ollama_model, initialize_openrouter_model
from prompt_defense.utils.embedding import generate_local_embeddings, generate_embeddings_google, generate_sentence_transformer_embeddings


@dataclass
class ModelConfig:
    """Configuration for a model setup including main model and embedding model."""
    provider: str
    model_name: str
    init_func: Callable
    embedding_func: Callable
    embedding_model: str


class ModelHandler:
    """Handles model initialization and embedding function selection."""

    @staticmethod
    def parse_model_string(model_string: str) -> Tuple[str, str]:
        """
        Parse model prefix and return (provider, model_name).

        Args:
            model_string: Model name with provider prefix (e.g., "google/gemini-2.0-flash")

        Returns:
            Tuple of (provider, model_name)

        Raises:
            ValueError: If model prefix is not recognized
        """
        if "/" not in model_string:
            raise ValueError(f"Model string must include provider prefix: {model_string}")

        provider, model_name = model_string.split("/", 1)

        if provider not in ["google", "ollama", "openrouter"]:
            raise ValueError(f"Unsupported provider: {provider}")

        return provider, model_name

    @staticmethod
    def get_init_function(provider: str) -> Callable:
        """Get the appropriate initialization function for a provider."""
        if provider == "google":
            return init_gemini_agent
        elif provider == "ollama":
            return initialize_ollama_model
        elif provider == "openrouter":
            return initialize_openrouter_model
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def get_embedding_config(
        main_provider: str,
        embedding_model: Optional[str] = None
    ) -> Tuple[Callable, str]:
        """
        Get the appropriate embedding function and model based on the main provider and embedding model.

        Args:
            main_provider: Provider of the main model (google, ollama, openrouter)
            embedding_model: Optional specific embedding model to use

        Returns:
            Tuple of (embedding_function, embedding_model_name)

        Raises:
            ValueError: If configuration is invalid
        """
        if embedding_model is not None:
            # Parse the embedding model to determine which function to use
            if embedding_model.startswith("google/") or embedding_model == "gemini-embedding-001":
                # Strip google/ prefix if present
                model_name = embedding_model.replace("google/", "")
                return generate_embeddings_google, model_name
            elif embedding_model.startswith("ollama/"):
                # Strip ollama/ prefix
                model_name = embedding_model.replace("ollama/", "")
                return lambda text: generate_local_embeddings(text, model=model_name), model_name
            elif embedding_model.startswith("sentence-transformers/") or embedding_model in [
                "nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5"
            ]:
                # Handle SentenceTransformer models
                model_name = embedding_model.replace("sentence-transformers/", "")
                # Map common short names to full model names
                if model_name == "nomic-embed-text-v1.5":
                    model_name = "nomic-ai/nomic-embed-text-v1.5"
                return lambda text: generate_sentence_transformer_embeddings(text, model_name=model_name), model_name
            else:
                # Assume it's an ollama model name without prefix
                return lambda text: generate_local_embeddings(text, model=embedding_model), embedding_model

        # Default embedding model selection based on main provider
        if main_provider == "google":
            return generate_embeddings_google, "gemini-embedding-001"
        elif main_provider == "ollama":
            return generate_local_embeddings, "embeddinggemma"  # Default ollama embedding model
        elif main_provider == "openrouter":
            # Default to SentenceTransformer for OpenRouter since we can't use OpenRouter for embeddings
            return lambda text: generate_sentence_transformer_embeddings(text, model_name="nomic-ai/nomic-embed-text-v1.5"), "nomic-ai/nomic-embed-text-v1.5"
        else:
            raise ValueError(f"Unsupported provider for embeddings: {main_provider}")

    @classmethod
    def create_model_config(
        cls,
        model_string: str,
        embedding_model: Optional[str] = None
    ) -> ModelConfig:
        """
        Create a complete model configuration.

        Args:
            model_string: Main model with provider prefix
            embedding_model: Optional embedding model specification

        Returns:
            ModelConfig with all necessary functions and model names

        Raises:
            ValueError: If configuration is invalid
        """
        # Parse main model
        provider, model_name = cls.parse_model_string(model_string)
        init_func = cls.get_init_function(provider)

        # Get embedding configuration
        embedding_func, embedding_model_name = cls.get_embedding_config(provider, embedding_model)

        return ModelConfig(
            provider=provider,
            model_name=model_name,
            init_func=init_func,
            embedding_func=embedding_func,
            embedding_model=embedding_model_name
        )

    @staticmethod
    def create_agent(config: ModelConfig, system_prompt: str, temperature: float = 0.7):
        """
        Create an agent using the model configuration.

        Args:
            config: ModelConfig with initialization details
            system_prompt: System prompt for the agent
            temperature: Model temperature

        Returns:
            Initialized agent
        """
        if config.provider == "google":
            return config.init_func(
                system_prompt=system_prompt,
                temperature=temperature,
                model_name=config.model_name
            )
        else:
            return config.init_func(
                system_prompt=system_prompt,
                temperature=temperature,
                model_name=config.model_name
            )

    @staticmethod
    def get_embedding_function(config: ModelConfig) -> Callable[[str], list]:
        """
        Get the embedding function from the model configuration.

        Args:
            config: ModelConfig with embedding function

        Returns:
            Function that takes text and returns embeddings
        """
        return config.embedding_func