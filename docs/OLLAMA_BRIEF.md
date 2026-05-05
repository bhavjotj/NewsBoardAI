# NewsBoardAI Ollama Briefs

NewsBoardAI can use a local Ollama model to rewrite the dashboard brief and possible impact in more natural language.

The Ollama brief is grounded only in:

- source titles
- source snippets
- detected mode
- sentiment
- event tags
- confidence
- the template brief and possible impact

Ollama does not choose sentiment, event tags, confidence, sources, detected mode, or sources. The existing hybrid analyzer still owns the analysis. Ollama only rewrites presentation text.

## Local Setup

Install Ollama from:

```text
https://ollama.com
```

Pull a small local model:

```bash
ollama pull llama3.2
```

Start Ollama:

```bash
ollama serve
```

NewsBoardAI calls Ollama locally at:

```text
http://127.0.0.1:11434
```

## Request Flags

Dashboard requests support:

```json
{
  "use_llm_brief": true,
  "ollama_model": "llama3.2"
}
```

`llama3.2` is the default. `phi3` is another lightweight local option.

## Fallback Behavior

The backend always creates the normal template brief first.

Ollama is asked for strict JSON with a short `brief` and one-sentence `possible_impact`.

If Ollama is not installed, not running, takes longer than 10 seconds, returns invalid JSON, returns empty text, or returns text that is too long, NewsBoardAI keeps the template text and marks the response as `ollama_fallback`.

If only one generated field is valid, NewsBoardAI uses that field and falls back only for the invalid field. Those fields are marked as `ollama_partial`.

The app works without Ollama. Ollama is local and free to use, but it uses your machine's CPU/GPU resources.
