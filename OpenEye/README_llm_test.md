# LLM-only Test Harness (No Actions)

This harness lets you test your local model output using a command loop similar to your VR agent input flow, but **without executing any actions**.

## File

- `llm_only_test_harness.py`

## What it does

- Uses an OpenAI-compatible endpoint (`llama-server` style API).
- Sends your text commands (or simulated voice text) to the model.
- Prints only model output.
- Includes a strict simulation system prompt to avoid action execution claims.

## Defaults (already match your setup)

- `OPENAI_BASE_URL=http://100.100.219.101:8001/v1`
- `OPENAI_API_KEY=sk-no-key-required`
- `LLM_MODEL=qwen3`

## Run (PowerShell)

```powershell
python .\OpenEye\llm_only_test_harness.py
```

## Optional overrides (PowerShell)

```powershell
$env:OPENAI_BASE_URL="http://100.100.219.101:8001/v1"
$env:OPENAI_API_KEY="sk-no-key-required"
$env:LLM_MODEL="qwen3"
$env:LLM_TEMPERATURE="0.7"
$env:LLM_TOP_P="0.8"
$env:LLM_MAX_TOKENS="1024"
python .\OpenEye\llm_only_test_harness.py
```

## In-session commands

- `quit` / `exit` → stop harness
- `reset` → clear conversation context
- Empty input line → prompts a "voice simulation" text input
