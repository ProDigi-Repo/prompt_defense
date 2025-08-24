from prompt_defense.utils.init_gemini import init_gemini_agent
from prompt_defense.utils.init_openai import initialize_ollama_model
from loguru import logger


def main():
    gemini = init_gemini_agent()

    result = gemini.run_sync(user_prompt="what is a stone")

    logger.info(f"Gemini response: {result}")

    ollama_agent = initialize_ollama_model(
        model_name="gpt-oss:20b", temperature=0.7, output_type=str
    )

    result = ollama_agent.run_sync(user_prompt="what is a stone")

    logger.info(f"Ollama response: {result}")


if __name__ == "__main__":
    main()
