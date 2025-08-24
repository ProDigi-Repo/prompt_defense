"""Utils for initialization of Gemini."""

from dotenv import load_dotenv
from pydantic_ai import Agent
import os
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai.models.google import GoogleModelSettings


load_dotenv(override=True)


def init_gemini_agent(
    system_prompt: str = "",
    thinking_budget: int = 200,
    temperature: float = 1,
    model_name: str = "gemini-2.5-flash",
):
    provider = GoogleProvider(api_key=os.getenv("GEMINI_API_KEY"))

    settings = GoogleModelSettings(
        temperature=temperature,
        google_thinking_config={"thinking_budget": 2048},
        # google_safety_settings=[
        #     {
        #         'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        #         'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        #     }
        # ]
    )
    model = GoogleModel(model_name=model_name, provider=provider, settings=settings)
    agent = Agent(model=model)

    return agent
