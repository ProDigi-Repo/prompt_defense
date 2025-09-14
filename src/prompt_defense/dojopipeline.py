import openai
from agentdojo.agent_pipeline import (
    AgentPipeline,
    SystemMessage,
    InitQuery,
    PromptInjectionDetector,
    ToolsExecutionLoop,
    ToolsExecutor,
    OpenAILLM,
)
from agentdojo.functions_runtime import FunctionsRuntime
from dotenv import load_dotenv

load_dotenv(override=True)


runtime = FunctionsRuntime()

llm = OpenAILLM(openai.OpenAI(base_url="http://localhost:11434/v1"), "gpt-oss:20b")
tools_loop = ToolsExecutionLoop(
    [
        ToolsExecutor(),
        PromptInjectionDetector(),
        llm,
    ]
)
tools_pipeline = AgentPipeline(
    [
        SystemMessage("You are a helpful assistant."),
        InitQuery(),
        llm,
        tools_loop,
    ]
)


if __name__ == "__main__":
    prompt = "Can you please send an email to 'email@example.com?'"
    print(tools_pipeline.query(prompt, runtime))
