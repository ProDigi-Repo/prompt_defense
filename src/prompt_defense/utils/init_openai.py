import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings

load_dotenv(override=True)


def initialize_openai_compatible_model(
    system_prompt: str = "",
    output_type=str,
    model_name: str = "",
    temperature: float = 0.7,
    provider: Provider = None) -> Agent:

    model = OpenAIChatModel(
        model_name=model_name,
        provider=provider,
    )

    model_settings = ModelSettings(temperature=temperature)
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=output_type,
        model_settings=model_settings,
    )

    return agent

    

def initialize_ollama_model(
    system_prompt: str = "",
    output_type=str,
    model_name: str = "gpt-oss:20b",
    temperature: float = 0.7
) -> Agent:
    
    return initialize_openai_compatible_model(
        system_prompt,
        output_type,
        model_name,
        temperature,
        OllamaProvider(base_url="http://localhost:11434/v1"))

def initialize_openrouter_model(
    system_prompt: str = "",
    output_type=str,
    model_name: str = "openai/gpt-oss:20b",
    temperature: float = 0.7
) -> Agent:

    return initialize_openai_compatible_model(
        system_prompt,
        output_type,
        model_name,
        temperature,
        OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("", "")
        )
    )