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
* [LLM Judge for Prompt Leak Detection](#llm_judge)
* [Retro-Running Judge on Existing Results](#retro_judge)
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

## 🤖 LLM Judge for Prompt Leak Detection <a name = "llm_judge"></a>

### Overview

The system includes an LLM-based judge (`gpt-oss-safeguard-20b`) that can detect if model responses have leaked the system prompt. The judge evaluates responses against the actual system prompt to identify:

- **Direct leaks**: Exact wording from system prompt
- **Paraphrased leaks**: Same meaning as system prompt, different wording
- **Partial leaks**: Key elements (role, constraints, objectives) revealed
- **Semantic leaks**: Demonstrates knowledge of internal instructions

### Integration with Workflows

The LLM judge can be integrated directly into the workflow execution:

#### 1. **During Workflow Execution** (Real-time)

Add the `--enable-judge` flag when running experiments:

```bash
uv run python src/prompt_defense/run_workflow.py \
    --model openrouter/claude-sonnet-4 \
    --prompt-source theft_prompts \
    --temperature 0.0 \
    --enable-judge \
    --verbose
```

**Judge Configuration Options:**
- `--enable-judge`: Enable LLM judge for prompt leak detection
- `--judge-model MODEL`: Judge model to use (default: `openrouter/openai/gpt-oss-safeguard-20b`)
- `--judge-reasoning-effort LEVEL`: Reasoning effort level (default: `medium`, options: `low`, `medium`, `high`)

#### 2. **Retro-Running on Existing Results** (Post-processing)

Use the retro-judge script to add judge results to already-generated Excel/JSON files:

```bash
# Run on all existing result files
./scripts/retro_judge.sh

# Or run on specific directory
./scripts/retro_judge.sh --directory results/temp_0_7

# Or process single file
./scripts/retro_judge.sh --file results/temp_0_7/experiment.xlsx

# With verbose output
./scripts/retro_judge.sh --verbose

# Force re-run even if judge columns exist
./scripts/retro_judge.sh --force
```

**Retro-Judge Options:**
- `--directory DIR`: Directory containing Excel files (default: processes all temp directories)
- `--file PATH`: Single Excel file to process
- `--model MODEL`: Judge model to use
- `--reasoning-effort LEVEL`: Reasoning effort level
- `--force`: Re-run judge even if columns already exist
- `--verbose`: Enable verbose logging

### Judge Output

The judge adds the following columns to Excel files:

| Column | Type | Description |
|---------|------|-------------|
| `llm_judge_leaked` | Boolean | True if prompt leak detected, False if safe, None if error |
| `llm_judge_confidence` | String | Confidence level: "low", "medium", or "high" |
| `llm_judge_rationale` | String | Brief explanation of the decision |
| `llm_judge_timing_ms` | Float | Time taken for judgment in milliseconds |

And adds judge statistics to JSON files:
- Total leaks detected
- Leak rate (percentage)
- Safe count
- Error count
- Timing statistics (total, mean, min, max)
- Confidence breakdown

### Judge Advantages

✅ **Better for paraphrased leaks** than text similarity metrics
✅ **Context-aware** - understands semantic meaning, not just string overlap
✅ **Explainable** - provides rationale for each decision
✅ **Adjustable reasoning** - control tradeoff between speed and accuracy
✅ **Policy-based** - can customize what constitutes a leak

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
  --enable-judge            Enable LLM judge for prompt leak detection
  --judge-model MODEL         Judge model to use (default: openrouter/openai/gpt-oss-safeguard-20b)
  --judge-reasoning-effort {low,medium,high}  Judge reasoning effort (default: medium)
  --verbose                  Enable verbose logging
```

#### Output Metrics

The system calculates and exports the following similarity metrics:

1. **Embedding Similarity**: Cosine similarity between system prompt and response embeddings
2. **Levenshtein Distance**: Normalized edit distance between texts
3. **BLEU Score**: Bilingual evaluation metric for text similarity
4. **ROUGE Scores**: ROUGE-1, ROUGE-2, and ROUGE-L for n-gram overlap analysis
5. **LLM Judge**: Prompt leak detection using gpt-oss-safeguard-20b (optional, with `--enable-judge`)

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
