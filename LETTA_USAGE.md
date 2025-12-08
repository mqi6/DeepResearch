# Letta Integration - Quick Reference

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LETTA_API_KEY` | *Required* | Your Letta Cloud API key |
| `LETTA_BASE_URL` | `https://api.letta.com` | Letta API endpoint |
| `LETTA_AGENT_ID` | *(optional)* | Specific agent ID to use |
| `LETTA_EPHEMERAL` | `false` | Create new agent per run |

## Usage Modes

### Persistent Memory (Default)
```bash
# Reuses existing agent across runs
python your_script.py
```

### Ephemeral Memory (Clean Slate)
```bash
# Creates fresh agent for each run
LETTA_EPHEMERAL=true python your_script.py
```

## How It Works

1. **Before Search**: Agent queries Letta memory
   - If **HIT**: Skip search, use cached result
   - If **MISS**: Execute search

2. **After Tool**: Agent saves result to Letta memory
   - All tool outputs (search, visit, etc.) are stored
   - Results truncated to 5000 chars max

3. **Logging**: Check token logs for:
   - `LETTA_RETRIEVE | length=X | content=...`
   - `LETTA_SAVE | length=X`
   - `FINAL_STATS | ... | letta_saves=X | letta_retrievals=X`

## Example Log Pattern

```
ROUND 1 | ...
LETTA_RETRIEVE | length=159 | content=YES: Saved notes...
# Search skipped, used memory

ROUND 2 | ...
LETTA_SAVE | length=438
# Tool executed, result saved

FINAL_STATS | ... | letta_saves=14 | letta_retrievals=9
```
