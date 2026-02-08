# Coding Guidelines for Prompt Defense System

This document provides coding guidelines, build commands, and testing instructions for AI agents working on the prompt defense repository.

## Build System

The project uses `uv_build` as defined in `pyproject.toml`.

### Installation

```bash
# Install dependencies (creates virtual environment automatically)
uv pip install -e .
```

### Linting & Formatting

The project uses pre-commit hooks with ruff:

```bash
# Install pre-commit hooks
pre-commit install

# Run linter
ruff check src/

# Format code
ruff format src/

# Check specific file
ruff check src/prompt_defense/run_workflow.py
```

### Type Checking

The project does not currently have explicit type checking configured. To add mypy:

```toml
# Add to pyproject.toml:
[tool.mypy]
python_version = "3.10"
warn_return_any = True
warn_unused_configs = True

[[tool.mypy.overrides]]
module = "prompt_defense.*"
ignore_missing_imports = True
```

## Code Style Guidelines

### Imports

**Follow these conventions:**

1. **Project imports first** (absolute from project root):
   ```python
   from prompt_defense.judge.prompt_leak_judge import PromptLeakJudge
   from prompt_defense.utils.judge import calculate_judge_stats
   ```

2. **Third-party imports next**:
   ```python
   import pandas as pd
   from loguru import logger
   from tqdm import tqdm
   ```

3. **Standard library last**:
   ```python
   import time
   import json
   ```

4. **Avoid circular imports** - don't import from subdirectories that import from you
5. **Use `from module import Class`** - don't use `import module; module.Class`

### Type Hints

**Use Python 3.10+ type hints consistently:**

```python
# Function signatures
def run_workflow(
    model_config: ModelConfig,
    prompts_list: list[str],
    prompt_source: str,
    temperature: float = 0.7,
    enable_judge: bool = False,
    judge_model: str | None = None,
    judge_reasoning_effort: str | None = None
) -> str:
    """Docstring."""
    pass

# Return types
def detect_leak(
    self, system_prompt: str, response: str
) -> dict[str, Any]:
    """Return dict with keys: leaked (bool | None), confidence (str), rationale (str), timing_ms (float)."""
    pass

# Class annotations
class PromptLeakJudge:
    def __init__(
        self,
        model: str = "openrouter/openai/gpt-oss-safeguard-20b",
        reasoning_effort: str = "medium",
        max_retries: int = 5
    ) -> None:
        pass
```

### Naming Conventions

**Follow PEP 8:**

- **Functions and variables**: `snake_case`
  ```python
  def calculate_judge_stats(judge_results: List[Dict[str, Any]]) -> Dict[str, Any]:
      total_leaks = len(leaked_responses)
  ```

- **Classes**: `PascalCase`
  ```python
  class PromptLeakJudge:
      class ModelHandler:
  ```

- **Constants**: `UPPER_SNAKE_CASE`
  ```python
  JUDGE_MODEL = "openrouter/openai/gpt-oss-safeguard-20b"
  SYSTEM_PROMPT = "..."
  ```

- **Private methods/attributes**: `_leading_underscore`
  ```python
  def _call_judge_api(self, ...):
      self._enhanced_policy = ...
  ```

### Formatting

**Line length**: Maximum 120 characters (ruff default)

**String formatting**:
- Use f-strings for simple cases: `f"{variable} value"`
- Use `.format()` for complex cases: `"{}/{}"`
- Use logging instead of print statements

### Docstrings

**Use Google style docstrings:**

```python
def function_name(param1, param2):
    """One-line summary.

    Extended description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
    pass
```

### Error Handling

**Follow these patterns:**

1. **Specific exceptions first, generic last**:
   ```python
   except FileNotFoundError as e:
       logger.error(f"File not found: {file_path}")
       sys.exit(1)
   except Exception as e:
       logger.error(f"Unexpected error: {e}")
       sys.exit(1)
   ```

2. **Log errors with context**:
   ```python
   logger.error(f"Failed to load Excel file {file_path}: {e}")
   logger.debug(f"Response was: {response[:200]}...")
   ```

3. **Never swallow exceptions silently**:
   ```python
   # BAD
   try:
       risky_operation()
   except:
       pass  # Silent failure!

   # GOOD
   try:
       risky_operation()
   except Exception as e:
       logger.warning(f"Operation failed: {e}")
       return None  # Explicit handling
   ```

4. **Use context managers**:
   ```python
   # GOOD
   with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
       df.to_excel(writer, sheet_name="Results")

   # BAD
   writer = pd.ExcelWriter(file_path, engine="openpyxl")
   df.to_excel(writer, sheet_name="Results")
   writer.close()
   ```

### Testing

**Run specific tests:**

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_judge.py

# Run with verbose output
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_judge.py::test_detect_leak

# Run with coverage
uv run pytest tests/ --cov=src/prompt_defense
```

**Running a single workflow file for testing:**

```bash
# Test judge module directly
uv run python -c "
from src.prompt_defense.judge.prompt_leak_judge import PromptLeakJudge
from src.prompt_defense.system_prompts.basic import SYSTEM_PROMPT

judge = PromptLeakJudge()
result = judge.detect_leak(SYSTEM_PROMPT, 'test response')
print(f'Result: {result}')
"

# Test workflow with specific prompt source
uv run python src/prompt_defense/run_workflow.py \
    --model ollama/llama3.1 \
    --prompt-source chat_prompts \
    --temperature 0.0 \
    --enable-judge \
    --verbose 2>&1 | head -100

# Test retro-judge on single file
uv run python src/prompt_defense/retro_judge.py \
    --file results/temp_0_7/experiment.xlsx \
    --verbose 2>&1 | tail -50
```

### Common Patterns

**Working with Excel files:**

```python
# Load Excel
df = pd.read_excel(file_path, sheet_name="Results")

# Load all sheets
xl = pd.ExcelFile(file_path)
for sheet_name in xl.sheet_names:
    df = pd.read_excel(xl, sheet_name=sheet_name)

# Update in-place (preserve formatting)
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name="Results", index=False)

# Use openpyxl for more control
from openpyxl import load_workbook
wb = load_workbook(file_path)
ws = wb["Results"]
ws.append_row([new_data])
wb.save(file_path)
```

**Working with JSON:**

```python
# Load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Save JSON
with open(json_path, 'w') as f:
    json.dump(data, f, indent=2)

# Handle numpy types in JSON
from numpy import integer as np_integer
json.dumps(data, cls=NumpyEncoder, indent=2)
```

**Progress bars:**

```python
from tqdm import tqdm

# Basic progress
for item in tqdm(items, desc="Processing"):
    process(item)

# Progress with metrics
with tqdm(total=len(items), desc="Processing") as pbar:
    for item in items:
        result = process(item)
        pbar.update(1)
        pbar.set_postfix({"time": f"{result['time']:.0f}ms"})
```

### Logging

**Use loguru (already configured):**

```python
from loguru import logger

# Different log levels
logger.debug("Detailed debug info")  # Only in verbose mode
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message")
logger.success("Success message")  # Green text

# Configure logging
if args.verbose:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
```

### Environment Variables

**Required environment variables:**

```bash
# For Google models
export GEMINI_API_KEY=your_key_here

# For OpenRouter models
export OPENROUTER_API_KEY=your_key_here

# Load from .env file
load-dotenv
```

### File Organization

**Project structure:**

```
prompt_defense/
├── src/prompt_defense/
│   ├── attack_prompts/          # Attack prompt definitions
│   ├── combined_workflows/    # Workflow implementations
│   ├── guard/                  # Guard models (separate)
│   ├── judge/                  # LLM judge implementation
│   ├── system_prompts/        # System prompt definitions
│   ├── utils/                  # Utility modules
│   │   ├── model_handler.py   # Model initialization
│   │   ├── excel_export.py    # Excel I/O
│   │   ├── json_storage.py    # JSON I/O
│   │   ├── judge.py           # Judge utilities
│   │   ├── embedding.py       # Embedding generation
│   │   └── ...
│   ├── run_workflow.py          # Main CLI
│   └── run_guard.py            # Guard CLI
├── scripts/
│   └── run_experiments.sh      # Batch experiment runner
├── results/                     # Output directory
└── pyproject.toml              # Project config
```

### Important Notes for AI Agents

1. **Always preserve imports order**: Project imports first, then third-party, then standard library
2. **Never modify test files** unless specifically asked to fix failing tests
3. **Check existing code patterns** before introducing new patterns
4. **Use existing utilities** - don't reinvent Excel/JSON handling
5. **Maintain backward compatibility** - default parameters should keep existing behavior
6. **Handle large files carefully** - use generators/iterators instead of loading everything into memory
7. **API rate limits**: Include proper retry logic with exponential backoff
8. **Path handling**: Use `pathlib.Path` for cross-platform compatibility
9. **Logging**: Use appropriate log levels and include context
10. **Type safety**: Validate inputs before use, handle type errors gracefully

### Quick Reference

| Task | Command |
|------|----------|
| Format code | `ruff format src/` |
| Check code | `ruff check src/` |
| Run tests | `uv run pytest tests/` |
| Install deps | `uv pip install -e .` |
| Lint specific | `ruff check src/prompt_defense/run_workflow.py` |
| Format specific | `ruff format src/prompt_defense/run_workflow.py` |

### Cursor/Copilot Integration

This project uses pre-commit hooks with ruff. Ensure your IDE (Cursor, Copilot) uses ruff for consistent formatting.
