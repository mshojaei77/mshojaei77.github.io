**Chapter 3: Building a Streaming LLM Chatbot**

A lot of beginners assume the model somehow "remembers" the conversation on its own. In the message-array pattern used in this chapter, that is not how it works. Every request is effectively stateless. If your app does not resend the earlier conversation, the model has no idea what was said a few seconds ago. On top of that, if you wait for a long answer to finish before showing anything, the app feels frozen.

A chatbot is not a magical conversational brain. It is an application loop that keeps packaging conversation history and sending it to a model provider. If you treat the model as a black box, costs climb fast, requests start failing, and small network problems turn into bad UX.

By the end of this chapter, you will have a clean, production-minded Python prototype of a ChatGPT-style CLI chatbot. You will add short-term conversation memory, keep API keys out of your source code, stream responses as they are generated, trim context growth, and track basic token usage with optional cost estimates.

**Code Repository**  
The complete runnable project for this chapter, implemented as a single-file CLI chatbot (`chatbot.py`), is available in the book repository:  
[github.com/mshojaei77/llm-engineering-in-action/chapter-03-chatbot](https://github.com/mshojaei77/llm-engineering-in-action/tree/main/chapter-03-chatbot)

Code evolves faster than prose. This chapter focuses on the underlying mechanics; the repository contains the runnable version. In the pages below, we keep only the snippets that matter so you can see how each part works.

## Chat API Structure

Before writing any code, we need to clear up a common source of confusion: modern LLM platforms often offer more than one way to generate text. In OpenAI-style ecosystems, two important interfaces are the **Chat Completions API** and the newer **Responses API**.

The **Responses API** is the recommended OpenAI interface for new greenfield applications. It is designed for text generation, multimodal input, structured outputs, built-in tools, function calling, streaming, and more agent-like workflows through one API.

So why does this chapter still teach the **message-array pattern** from Chat Completions?

Because this chapter is not about agents, built-in tools, file search, or multimodal orchestration. It is about learning the core loop behind a basic chatbot:

```text
messages -> model call -> streamed answer -> save assistant reply -> repeat
```

The Chat Completions style makes that loop easy to see. It uses a plain list of messages with roles like `system`, `user`, and `assistant`. Many providers and routing layers also expose OpenAI-compatible chat-completion-style APIs, so this pattern is still useful when you want portability across vendors.

An **OpenAI-compatible provider** is a service that accepts the same general request shape as the OpenAI SDK or REST API, even if another company is actually serving the model. In practice, that often means your code stays mostly the same and only your configuration changes:

```text
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-5-nano
```

With a direct REST call, the request usually looks like this:

```text
POST https://openrouter.ai/api/v1/chat/completions
Authorization: Bearer $OPENAI_API_KEY
Content-Type: application/json

{
  "model": "openai/gpt-5-nano",
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user", "content": "Explain KV cache in one paragraph."}
  ],
  "temperature": 0.4,
  "max_tokens": 500,
  "stream": true
}
```

In Python, prefer using the official OpenAI SDK with a `base_url` override instead of hand-rolling HTTP requests. The SDK gives you typed objects, cleaner streaming iteration, built-in request handling, and fewer mistakes around request shape.

```python
from openai import OpenAI

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)
```

Some OpenAI-compatible providers support or request extra attribution headers. For example, OpenRouter commonly uses headers such as `HTTP-Referer` and `X-Title` to identify the app. Treat those as provider-specific configuration, not part of the universal chat-completions contract.

For this chapter, the important difference is **state management**. With Chat Completions, you manage the conversation state yourself: your Python app stores the message history and resends the relevant part of that history on every turn. That manual state management is useful because it forces you to see what a chatbot actually is: an application that manages context, controls token growth, streams output, and decides what the model is allowed to see.

> **Practical rule:**
> Use the message-array pattern to learn the fundamentals. Move to a Responses-style API when you need richer built-in tools, provider-managed conversation state, multimodal input, or more complex agent workflows.

## Roles and Message History

A useful mental model is to imagine handing an actor the full script of a scene, asking for the next line, writing that line at the end of the script, and then reading the whole updated script back to them for the next turn.

In practice, your Python app keeps a list of dictionaries called the **message array**. Each dictionary has a `role` and some `content`.

Typical roles are:

*   `system`: The application-level instruction, such as `"You are a helpful assistant."`
*   `user`: The human message.
*   `assistant`: The model's reply.

Some APIs also support roles such as `developer` or tool-related message types. In this chapter, we use `system`, `user`, and `assistant` because they remain broadly compatible across OpenAI-style providers.

Every time the user types something, you append it to the list. Then you send the relevant part of the list to the API. The API returns an assistant message. You append that reply, then repeat the loop.

<img width="300" height="600" alt="image" src="https://github.com/user-attachments/assets/2e0917d3-c8f6-4b93-8b70-1ca6bcf7a4e5" />

The key point is that the model does not persist memory between requests unless your application explicitly sends earlier messages back.

A minimal in-memory history looks like this:

```python
messages = [
    {"role": "system", "content": "You are a concise assistant."}
]

messages.append({"role": "user", "content": "What is a context window?"})

# Send messages to the API...

messages.append({
    "role": "assistant",
    "content": "A context window is the maximum amount of text..."
})
```

Keep two kinds of context separate:

*   **Persistent instructions:** The system prompt and application policy that should stay at the top of the conversation.
*   **Dynamic history:** The user and assistant turns that grow during the session.

This separation matters when you start trimming history. You usually want to preserve the system instruction while removing, summarizing, or compressing older user and assistant turns.

System instructions are influential, but they are not magic. Later user messages, retrieved context, and tool outputs can still create conflicts. Production systems should use validation, guardrails, and tool permissions instead of relying only on a system prompt.

## Environment and API Keys

Before running the chatbot loop, configure environment variables and provider credentials so the application can run safely across machines and environments.

Never hardcode your API key in a Python file, notebook, or frontend application. Secrets should live outside the codebase and be loaded at runtime. In this chapter, we use `python-dotenv` to read values from a local `.env` file, and that file must be listed in `.gitignore` so it never gets committed by mistake.

A professional project should also include a `.env.example` file. It shows which environment variables are required without exposing real credentials.

```text
# .env.example
OPENAI_API_KEY=your_openrouter_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-5-nano
APP_URL=http://localhost:3000
APP_NAME=Chapter 3 CLI Chatbot
```

Then load and validate the configuration at startup:

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
APP_URL = os.getenv("APP_URL")
APP_NAME = os.getenv("APP_NAME")

missing = [
    name
    for name, value in {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_BASE_URL": OPENAI_BASE_URL,
        "OPENAI_MODEL": OPENAI_MODEL,
    }.items()
    if not value
]

if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
```

For provider-specific headers, keep the code explicit:

```python
client_kwargs = {
    "api_key": OPENAI_API_KEY,
    "base_url": OPENAI_BASE_URL,
}

if APP_URL and APP_NAME:
    client_kwargs["default_headers"] = {
        "HTTP-Referer": APP_URL,
        "X-Title": APP_NAME,
    }

client = OpenAI(**client_kwargs)
```

For a local tutorial, `.env` is fine. In production, use a secret manager supplied by your deployment platform, such as AWS Secrets Manager, Google Secret Manager, Azure Key Vault, Vercel environment variables, GitHub Actions secrets, or your company's internal secret store.

Also set usage limits in the provider dashboard when available. A leaked key or infinite retry loop can become expensive quickly.

## Streaming Response Handling

Latency matters. A long answer can take several seconds to generate. If the interface stays blank until the full response is ready, users will assume the app has stalled. That is why we use **streaming** with `stream=True`. It lets the API send text chunks immediately, which dramatically improves perceived time to first token.

Streaming does not deliver one perfect word at a time. It sends partial chunks, often called **deltas**. In a CLI application, we print each chunk as it arrives while also collecting the pieces so we can reconstruct the full assistant reply afterward.

Under the hood, many streaming APIs use Server-Sent Events (SSE). If you use raw HTTP, you may see lines of event data and a final done marker. If you use the SDK, most of that parsing is handled for you.

One subtle point matters here: streaming APIs do not always return token usage unless you ask for it, and provider support varies. Passing `stream_options={"include_usage": True}` can instruct compatible providers to send a final metadata chunk with usage information.

```python
stream = client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=request_messages,
    temperature=0.4,
    top_p=0.9,
    max_tokens=1000,
    stream=True,
    stream_options={"include_usage": True},
)
```

Then process the chunks incrementally:

```python
reply_parts = []
prompt_tokens = 0
completion_tokens = 0

print("Assistant: ", end="", flush=True)

for chunk in stream:
    if getattr(chunk, "usage", None):
        prompt_tokens = chunk.usage.prompt_tokens
        completion_tokens = chunk.usage.completion_tokens
        continue

    if not chunk.choices:
        continue

    delta = chunk.choices[0].delta
    text = getattr(delta, "content", None)

    if text:
        reply_parts.append(text)
        print(text, end="", flush=True)

print()
assistant_reply = "".join(reply_parts)
```

Because we collect the streamed pieces into `reply_parts`, we still end up with one complete assistant string that can be stored in the message history.

Streaming has a few production details to remember:

*   Some chunks contain metadata, role changes, usage, or finish signals instead of visible text.
*   A stream can fail halfway through, leaving a partial answer.
*   In a web UI, users expect a stop button. That means your frontend needs an abort mechanism and your backend needs to handle cancellation.
*   Streaming makes moderation and validation more complicated because the user may see partial output before the final answer is complete.
*   Tool calling and structured outputs can stream differently from plain text, so test the exact provider and model you plan to use.

## Command-Line Chatbot Design

We will build this chatbot in plain **Python** and run it in the terminal. That keeps the architecture simple and makes the chatbot loop impossible to miss before we bring in a web framework.

There are six mechanics you need to understand.

**1. Secure secret loading**  
Load configuration from environment variables at startup rather than scattering credentials through the code.

**2. Conversation state**  
In a CLI chatbot, there is no browser session and no framework-managed state object. We keep the conversation in a Python list and reuse it inside a `while True` loop. As long as the process stays alive, the message history stays in memory.

```python
messages = [
    {"role": "system", "content": "You are a concise assistant."}
]

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})
```

**3. Control commands**  
A CLI chatbot should support a few commands that never go to the model.

```python
if user_input == "/exit":
    break
if user_input == "/clear":
    messages = messages[:1]
    print("Conversation cleared.")
    continue
if user_input == "/stats":
    print(f"Messages in memory: {len(messages)}")
    continue
```

**4. The stateless API call**  
Before each request, build the message list you want the provider to see. Set an output token limit to prevent runaway generations, and enable streaming.

```python
request_messages = trim_messages(messages, max_messages=20)
```

**5. Streaming the reply**  
Print text chunks as they arrive so the interface feels alive rather than blocked.

**6. Appending the final reply back to history**  
The model will not remember what it just said unless you save that answer yourself.

```python
messages.append({"role": "assistant", "content": assistant_reply})
```

The CLI may look simple, but it already includes the core architecture of many production chat systems: state assembly, request dispatch, streaming output, state update, context control, and failure handling.

## Context Growth Management

As users keep chatting, the `messages` list grows forever unless you control it. Before each request, trim the history to avoid blowing past the context window or paying for unnecessary tokens.

The simplest approach is a sliding window over recent turns:

```python
def trim_messages(messages, max_messages=20):
    system = messages[:1]
    history = messages[1:]
    recent = history[-max_messages:]
    return system + recent
```

This is crude, but it is enough for a first chatbot. It preserves the system prompt and keeps only the most recent dynamic history.

The weakness is obvious: naive trimming can delete important facts from earlier in the conversation. A slightly better pattern is to maintain a summary of older turns and include that summary before the recent messages.

```python
def build_request_messages(messages, summary=None, max_messages=20):
    system = messages[:1]
    history = messages[1:]
    recent = history[-max_messages:]

    if summary:
        summary_message = {
            "role": "system",
            "content": f"Conversation summary so far: {summary}",
        }
        return system + [summary_message] + recent

    return system + recent
```

Some providers prefer a single system message. If you see role-handling differences, merge the summary into the main system prompt or store it as a dedicated assistant summary message. Provider compatibility should always be tested with your exact model and endpoint.

Later chapters will cover more advanced context strategies such as token-aware trimming, summarization, retrieval, vector memory, prompt caching, and long-term memory layers.

There is also a direct cost reason to manage history. Providers bill by token count. If your history contains ten messages, then by the tenth turn you have paid to process the first message ten separate times. The longer the conversation, the more expensive each new turn becomes.

With usage data captured from the stream, you can estimate cost locally:

```python
def estimate_cost(input_tokens, output_tokens, input_price, output_price):
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost
```

For a CLI chatbot, control commands such as `/clear`, `/stats`, and `/exit` are practical tools for managing context and session state.

For production systems, context growth management should be observable. Track prompt tokens, completion tokens, total tokens, total latency, and estimated cost per turn. If a chatbot suddenly becomes slow or expensive, message history growth is one of the first places to inspect.

## Runtime Error Handling

A chatbot that only works on a perfect network is not production-minded. Basic runtime error handling matters even in a small CLI prototype.

Handle these failure modes explicitly:

*   **Missing configuration:** Fail fast if `OPENAI_API_KEY`, `OPENAI_BASE_URL`, or `OPENAI_MODEL` is missing.
*   **Authentication errors:** Show a clear message when the API key is invalid or expired.
*   **Provider-specific header errors:** Some OpenAI-compatible providers require or recommend attribution headers.
*   **Rate limiting (`429`):** If requests come too quickly, the provider may throttle you. In production, this is usually handled with retries, exponential backoff, and concurrency limits.
*   **Context length exceeded:** Trim, summarize, or ask the user to clear the conversation instead of retrying the same oversized request.
*   **Network timeouts and outages:** Wrap the API call in `try...except` and show a short user-facing error instead of dumping a raw traceback.
*   **Interrupted streams:** If the stream ends early, you may have partial output and missing usage metadata. Decide whether to keep the partial answer or discard it.
*   **Unsupported parameters:** OpenAI-compatible does not mean identical. One provider may ignore, reject, or rename parameters such as usage streaming, structured output, or tool calling.
*   **Temporary session memory:** A CLI chatbot keeps memory in RAM only while the Python process is running. If the user exits or the process crashes, the conversation is gone.

A minimal pattern looks like this:

```python
try:
    stream = client.chat.completions.create(...)
except Exception as exc:
    print(f"Request failed: {exc}")
    continue
```

For a sturdier prototype, distinguish transient errors from permanent errors:

```python
import time
from openai import AuthenticationError, BadRequestError, RateLimitError, APIConnectionError

def create_stream_with_retry(client, request, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return client.chat.completions.create(**request)
        except AuthenticationError:
            raise
        except BadRequestError:
            raise
        except RateLimitError:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)
        except APIConnectionError:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** attempt)
```

Do not blindly retry every failure. Retrying an invalid API key or oversized context will not help. Retrying a rate limit or temporary network failure may help. Production systems usually add request timeouts, circuit breakers, fallback models, provider routing, and monitoring.

Also verify behavior when using OpenAI-compatible providers. "Compatible" rarely means identical. Test streaming behavior, supported parameters, usage reporting, rate limits, model names, role handling, and error shapes before you ship.

A simple engineering checklist helps:

*   [ ] Are API keys stored only in a `.env` file or a proper secrets manager?
*   [ ] Is `.env` listed in `.gitignore`?
*   [ ] Is there a committed `.env.example` file?
*   [ ] Are provider-specific headers documented in configuration?
*   [ ] Is the `system` prompt consistently placed at the top of the message list?
*   [ ] Do you have a mechanism such as `/clear` to manage context growth?
*   [ ] Is history trimmed before each request?
*   [ ] Is an output limit such as `max_tokens` set to prevent unbounded generation?
*   [ ] Are rate limits, authentication errors, context-length errors, and stream interruptions handled differently?
*   [ ] Have you verified that your provider handles streaming and usage metadata the way your code expects?
*   [ ] Are token usage, latency, and estimated cost visible during testing?

**Project Walkthrough**

Instead of printing the entire application here, open the **Chapter 3 folder** in the book's GitHub repository.

As you run the app locally, focus on how the snippets from this chapter fit together in one file:

1. How `chatbot.py` loads the API key safely.
2. How `OPENAI_BASE_URL` and `OPENAI_MODEL` redirect the same SDK to an OpenAI-compatible provider.
3. How the `while True` loop keeps the message array alive in memory.
4. How control commands such as `/clear`, `/stats`, and `/exit` change local state without calling the model.
5. How the app trims message history before each request.
6. How the streaming loop separates text chunks from metadata and usage chunks.
7. How the final assistant reply gets appended back into `messages`.
8. How runtime failures are caught and turned into usable CLI messages.

## Capstone Project

Build a practical Telegram AI assistant bot. This is the Part 1 capstone because it combines the fundamentals from Chapters 1, 2, and 3: generation behavior, model selection, API configuration, message history, streaming or partial responses, context trimming, and runtime error handling.

Project brief:

```text
Users message a Telegram bot.
The bot sends their message and recent conversation history to an LLM provider.
The model returns a concise assistant reply.
The bot sends the reply back to Telegram.
The system handles configuration, context growth, rate limits, and basic errors.
```

Requirements:

*   Store `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` in `.env`.
*   Use an OpenAI-compatible chat-completions endpoint through the SDK.
*   Keep separate short message histories per Telegram chat ID.
*   Add Telegram commands:
    *   `/start` to introduce the bot.
    *   `/clear` to reset that chat's message history.
    *   `/stats` to show message count and basic token or request totals.
*   Trim old messages before each API call.
*   Set a maximum output token limit.
*   Use conservative decoding settings for assistant replies, such as low to moderate temperature.
*   Handle missing keys, Telegram API errors, LLM provider errors, rate limits, and context-length failures gracefully.
*   Log model name, chat ID, latency, token usage when available, and error type.
*   Do not store secrets in prompts or logs.

Optional upgrades:

*   Show a "typing..." action while the model is generating.
*   Add a simple per-user rate limit.
*   Add a fallback model if the default provider fails.
*   Persist chat history in SQLite instead of RAM.
*   Add a `/model` command that prints the current configured model.

Right now, the chatbot is still generic. If you ask it to behave like a narrow domain expert, or to produce strict JSON for another system, it will often drift or hallucinate. In the next chapter, we will look at prompt engineering, context engineering, and techniques for getting more structured outputs.
