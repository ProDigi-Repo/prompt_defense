<h3 align="center">Prompt Defense</h3>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

<p align="center"> Some sophisticated description coming soon
    <br> 
</p>

## 📝 Table of Contents
* [About](#about)
* [Getting Started](#getting_started)
* [Notes](#notes)

  
## 🧐 About <a name = "about"></a>


## 🚀 Experiment Set-Up <a name = "setup"></a>
## Adversarial Prompts <a name = "setup1.1"></a>
Prompt injections targeting the retrieval of the system prompt are in the theft_prompts.py
Normal chat interaction prompts are in the chat_prompts.py

## Metrics <a name = "setup1.2"></a>

## Sources

- ```generated.py``` is from https://github.com/y0mingzhang/prompt-extraction/blob/main/attacks/generated.json.
- ```selected.py``` is from https://github.com/y0mingzhang/prompt-extraction/blob/main/attacks/selected.json.

## 🚀 Usage <a name = "usage"></a>

### Overview

The prompt defense system evaluates AI model responses to various attack prompts using multiple similarity metrics. The CLI tool provides a unified interface for running comprehensive defense assessments across different model providers.

### CLI Tool

The `run_workflow.py` script automatically selects the appropriate combined workflow based on the model provider and calculates all similarity metrics (embeddings, Levenshtein distance, BLEU, and ROUGE scores).

#### Basic Usage

```bash
# Run with Google Gemini
uv run src/prompt_defense/run_workflow.py --model google/gemini-2.0-flash

# Run with local Ollama model
uv run src/prompt_defense/run_workflow.py --model ollama/llama3.1

# Run with OpenRouter
uv run src/prompt_defense/run_workflow.py --model openrouter/x-ai/grok-4-fast:free

# Specify custom embedding model
uv run src/prompt_defense/run_workflow.py --model openrouter/gpt-4 --embedding-model google/gemini-embedding-001
```

#### Model Provider Selection

The system automatically determines the workflow based on the model provider:

- **Google models** (`google/`): Uses Gemini-based combined workflow with Google embeddings
- **Ollama models** (`ollama/`): Uses local combined workflow with local embeddings
- **OpenRouter models** (`openrouter/`): Uses local combined workflow with OpenRouter API and default embeddings

#### Command Line Options

```bash
Options:
  --model MODEL               Model with provider prefix (required)
  --embedding-model MODEL     Custom embedding model (optional)
  --prompt-source SOURCE      Attack prompt source: generated, selected, gemini_generated (default: generated)
  --temperature FLOAT         Model temperature (default: 0.7)
  --output-dir DIR           Results directory (default: results)
  --verbose                  Enable verbose logging
```

#### Output Metrics

The system calculates and exports the following similarity metrics:

1. **Embedding Similarity**: Cosine similarity between system prompt and response embeddings
2. **Levenshtein Distance**: Normalized edit distance between texts
3. **BLEU Score**: Bilingual evaluation metric for text similarity
4. **ROUGE Scores**: ROUGE-1, ROUGE-2, and ROUGE-L for n-gram overlap analysis

Results are exported to Excel files in the `results/` directory with comprehensive statistics and analysis.

#### Environment Setup

Ensure required API keys are set in your `.env` file:

```bash
# For Google models
GEMINI_API_KEY=your_key_here

# For OpenRouter models
OPENROUTER_API_KEY=your_key_here

# Ollama models require no API key (local)
```
