# Research, Critique, Revise

A FastAPI service where a Supervisor orchestrates two agents through a
Research &rarr; Critique &rarr; Revise loop, then streams the final answer.
Calls [Groq](https://console.groq.com)'s SDK directly -- no LangChain, no
LangGraph.

## How it works

- **Researcher** drafts an answer, calling a web-search tool when the
  question needs current or specific facts.
- **Critic** grades the draft and returns a typed verdict (`approved` /
  `needs_revision`) plus concrete feedback.
- If revision is needed, the Supervisor sends the critique back to the
  Researcher (capped at two passes). Once approved, the **Supervisor**
  synthesises the final answer itself and streams it to the client.

## Project layout

```
app/
  main.py       FastAPI app, POST /chat/stream
  service.py    Supervisor: orchestrates Researcher and Critic, calls Groq directly
  prompts.py    Raw prompt strings, one pair per agent
  schemas.py    ChatRequestPayload (API request) and CritiqueVerdict (Critic's output)
  tools.py      The Researcher's web-search tool
```

## Setup

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY -- free tier at console.groq.com
```

## Running it

```bash
uvicorn app.main:app --reload --port 8000
```

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What caused the 2008 financial crisis?"}'
```

The response streams as plain text while the Supervisor generates it.

### In VS Code

Open the folder, then Run and Debug (`Cmd+Shift+D`) -> **"Run API
(uvicorn)"** -> press play. The interpreter is pre-set to `.venv` in
`.vscode/settings.json`.

## Design notes

- No LangChain or LangGraph: every LLM call goes through Groq's own SDK
  directly, giving full access to its documented parameters rather than a
  wrapper's subset.
- One provider, no fallback switch -- `GROQ_MODEL` in `.env` picks the model.
- Prompts are plain strings in `prompts.py`; the code that fills them in and
  sends them to the model lives in `service.py`, next to the LLM calls.
