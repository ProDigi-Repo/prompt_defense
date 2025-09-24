from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

load_dotenv(override=True)


def initialize_ollama_model(
    system_prompt: str = "",
    output_type=str,
    model_name: str = "gpt-oss:20b",
    temperature: float = 0.7,
) -> Agent:
    ollama_model = OpenAIModel(
        model_name=model_name,
        provider=OllamaProvider(base_url="http://localhost:11434/v1"),
    )

    model_settings = ModelSettings(temperature=temperature)
    agent = Agent(
        model=ollama_model,
        system_prompt=system_prompt,
        output_type=output_type,
        model_settings=model_settings,
    )

    return agent
