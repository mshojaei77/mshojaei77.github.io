# Chapter 03 CLI Chatbot

Minimal single-file CLI chatbot for Chapter 3 of *LLM Engineering in Action*.

The default configuration targets OpenRouter through the OpenAI Python SDK's OpenAI-compatible interface.

## Features

- OpenAI chat-completions style message array
- OpenRouter as the default OpenAI-compatible provider
- Single-file Python implementation
- Streaming responses with `stream=True`
- Token usage tracking from the final stream metadata chunk
- Basic per-session cost estimation
- CLI commands for clearing history and showing stats
- `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
- `OPENAI_MODEL=openai/gpt-5-nano`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Then set `OPENAI_API_KEY` in `.env`.

## OpenAI-Compatible Providers

This project uses the official `openai` Python SDK, but the actual provider can be any service that supports the OpenAI chat completions request format. OpenRouter is one such provider, so the same code works by changing the base URL and model name rather than rewriting the app.

In this sample, the defaults are:

```text
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-5-nano
```

That means:

- The SDK still sends `client.chat.completions.create(...)`
- Requests go to OpenRouter instead of OpenAI's own endpoint
- The model name uses OpenRouter's routing format

## Run

```bash
python chatbot.py
```

## Commands

- `/help` shows commands
- `/clear` resets chat history
- `/stats` prints token and cost totals
- `/exit` quits the app

## Project Structure

```text
chapter-03-chatbot/
├── chatbot.py
├── requirements.txt
├── .env.example
└── .gitignore
```
