# Chapter 3: Building a Streaming LLM Chatbot

A lot of beginners assume the model somehow "remembers" the conversation on its own. In the message-array pattern used in this chapter, that is not how it works. Every request is effectively stateless. If your app does not resend the earlier conversation, the model has no idea what was said a few seconds ago. On top of that, if you wait for a long answer to finish before showing anything, the app feels frozen.

A chatbot is not a magical conversational brain. It is an application loop that keeps packaging conversation history and sending it to a model provider. If you treat the model as a black box, costs climb fast, requests start failing, and small network problems turn into bad UX.

By the end of this chapter, you will have a clean, production-minded Python prototype of a ChatGPT-style CLI chatbot. You will add short-term conversation memory, keep API keys out of your source code, stream responses as they are generated, and track basic token usage with optional cost estimates.

**Code Repository**  
The complete runnable project for this chapter, implemented as a single-file CLI chatbot (`chatbot.py`), is available in the book repository:  
[github.com/mshojaei77/llm-engineering-in-action/chapter-03-chatbot](https://github.com/mshojaei77/llm-engineering-in-action/tree/main/chapter-03-chatbot)

Code evolves faster than prose. This chapter focuses on the underlying mechanics; the repository contains the runnable version. In the pages below, we keep only the snippets that matter so you can see how each part works.

## Chat API Structure

Before writing any code, we need to clear up a common source of confusion: modern LLM platforms often offer more than one way to generate text. In OpenAI-style ecosystems, two important interfaces are the **Chat Completions API** and the newer **Responses API**.

The **Responses API** is the newer general-purpose interface for greenfield applications. It is designed to handle text generation, multimodal input, reasoning models, built-in tools, and more agent-like workflows through one API.

So why does this chapter still teach the **message-array pattern** from Chat Completions?

Because this chapter is not about agents, tool use, file search, or multimodal orchestration. It is about learning the core loop behind a basic chatbot:

```text
messages -> model call -> streamed answer -> save assistant reply -> repeat
```

The Chat Completions style makes that loop easy to see. It uses a plain list of messages with roles like `system`, `user`, and `assistant`. Many providers and routing layers also expose OpenAI-compatible chat-completion-style APIs, so this pattern is still useful when you want portability across vendors.

An **OpenAI-compatible provider** is a service that accepts the same request shape as the OpenAI SDK or REST API, even if another company is actually serving the request. In practice, that often means your code stays the same and only your configuration changes:

```text
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-5-nano
```

This chapter uses **OpenRouter** as the concrete example. The Python code still calls `client.chat.completions.create(...)`, but the request is routed to an OpenAI-compatible endpoint instead of OpenAI's default host.

For this chapter, the important difference is **state management**. With Chat Completions, you manage the conversation state yourself: your Python app stores the message history and resends the relevant part of that history on every turn. That manual state management is useful because it forces you to see what a chatbot actually is: an application that manages context, controls token growth, streams output, and decides what the model is allowed to see.

> **Practical rule:**
> Use the message-array pattern to learn the fundamentals. Move to a Responses-style API when you need richer built-in tools, more stateful workflows, or multimodal orchestration.

## Roles and Message History

A useful mental model is to imagine handing an actor the full script of a scene, asking for the next line, writing that line at the end of the script, and then reading the whole updated script back to them for the next turn.

In practice, your Python app keeps a list of dictionaries called the **message array**. Each dictionary has a `role` and some `content`.

Typical roles are:
*   `system`: The application-level instruction, such as `"You are a helpful assistant."`
*   `user`: The human message.
*   `assistant`: The model's reply.

Some newer APIs also use a `developer` role for higher-priority instructions. In this chapter, we use `system` because it remains broadly compatible across OpenAI-style providers.

Every time the user types something, you append it to the list. Then you send the full list to the API. The API returns an `assistant` message. You append that reply, then repeat the loop.

<img width="300" height="600" alt="image" src="https://github.com/user-attachments/assets/2e0917d3-c8f6-4b93-8b70-1ca6bcf7a4e5" />

The key point is that the model does not persist memory between requests unless your application explicitly sends the earlier messages back.

## Environment and API Keys

Before running the chatbot loop, configure environment variables and provider credentials so the application can run safely across machines and environments.

Never hardcode your API key in a Python file, notebook, or frontend application. Secrets should live outside the codebase and be loaded at runtime. In this chapter, we use `python-dotenv` to read values from a local `.env` file, and that file must be listed in `.gitignore` so it never gets committed by mistake.

A professional project should also include a `.env.example` file. It shows which environment variables are required without exposing real credentials.

```text
# .env.example
OPENAI_API_KEY=your_openrouter_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-5-nano
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
```

If a required variable is missing, fail early with a clear error message instead of letting the program crash deep inside the API call.

## Streaming Response Handling

Latency matters. A long answer can take several seconds to generate. If the interface stays blank until the full response is ready, users will assume the app has stalled. That is why we use **streaming** with `stream=True`. It lets the API send text chunks immediately, which dramatically improves time to first token.

Streaming does not deliver one perfect word at a time. It sends partial chunks, often called *deltas*. In a CLI application, we print each chunk as it arrives while also collecting the pieces so we can reconstruct the full assistant reply afterward.

One subtle point matters here: streaming APIs do not always return token usage unless you ask for it. Passing `stream_options={"include_usage": True}` can instruct the provider to send a final metadata chunk with usage information.

```python
stream = client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=messages,
    temperature=0.7,
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
    if chunk.usage:
        prompt_tokens = chunk.usage.prompt_tokens
        completion_tokens = chunk.usage.completion_tokens

    if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
        text = chunk.choices[0].delta.content
        reply_parts.append(text)
        print(text, end="", flush=True)

print()
assistant_reply = "".join(reply_parts)
```

Because we collect the streamed pieces into `reply_parts`, we still end up with one complete assistant string that can be stored in the message history.

## Command-Line Chatbot Design

We will build this chatbot in plain **Python** and run it in the terminal. That keeps the architecture simple and makes the chatbot loop impossible to miss before we bring in a web framework.

There are five mechanics you need to understand.

**1. Secure secret loading**  
Load configuration from environment variables at startup rather than scattering credentials through the code.

**2. Conversation state**  
In a CLI chatbot, there is no browser session and no framework-managed state object. We simply keep the conversation in a Python list and reuse it inside a `while True` loop. As long as the process stays alive, the message history stays in memory.

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

**3. The stateless API call**  
Once the array is ready, send it to the provider. Set an output token limit to prevent runaway generations, and enable streaming.

**4. Streaming the reply**  
Print text chunks as they arrive so the interface feels alive rather than blocked.

**5. Appending the final reply back to history**  
The model will not remember what it just said unless you save that answer yourself.

```python
messages.append({"role": "assistant", "content": assistant_reply})
```

The CLI may look simple, but it already includes the core architecture of many production chat systems: state assembly, request dispatch, streaming output, and state update.

## Context Growth Management

As users keep chatting, the `messages` list grows forever unless you control it. Before each request, trim the history to avoid blowing past the context window or paying for unnecessary tokens.

```python
def trim_messages(messages, max_messages=12):
    system = messages[:1]
    history = messages[1:]
    recent = history[-max_messages:]
    return system + recent
```

This is a crude approach, but it is enough for a first chatbot. Later chapters will cover more advanced context strategies such as summarization, retrieval, and memory layers.

There is also a direct cost reason to manage history. Providers bill by token count. If your history contains ten messages, then by the tenth turn you have paid to process the first message ten separate times. The longer the conversation, the more expensive each new turn becomes.

With usage data captured from the stream, you can estimate cost locally:

```python
def estimate_cost(input_tokens, output_tokens, input_price, output_price):
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost
```

For a CLI chatbot, control commands such as `/clear`, `/stats`, and `/exit` are also practical tools for managing context and session state.

## Runtime Error Handling

A chatbot that only works on a perfect network is not production-minded. Basic runtime error handling matters even in a small CLI prototype.

Handle these failure modes explicitly:
*   **Missing configuration:** Fail fast if `OPENAI_API_KEY`, `OPENAI_BASE_URL`, or `OPENAI_MODEL` is missing.
*   **Network timeouts and outages:** Wrap the API call in `try...except` and show a short user-facing error instead of dumping a raw traceback.
*   **Rate limiting (`429`)**: If requests come too quickly, the provider may throttle you. In production, this is usually handled with retries and exponential backoff.
*   **Interrupted streams:** If the stream ends early, you may have partial output and missing usage metadata. Decide whether to keep the partial answer or discard it.
*   **Temporary session memory:** A CLI chatbot keeps memory in RAM only while the Python process is running. If the user exits or the process crashes, the conversation is gone.

A minimal pattern looks like this:

```python
try:
    stream = client.chat.completions.create(...)
except Exception as exc:
    print(f"Request failed: {exc}")
    continue
```

Also verify behavior when using OpenAI-compatible providers. "Compatible" rarely means identical. Test streaming behavior, supported parameters, usage reporting, and role handling before you ship.

A simple engineering checklist helps:
*   [ ] Are API keys stored only in a `.env` file or a proper secrets manager?
*   [ ] Is `.env` listed in `.gitignore`?
*   [ ] Is there a committed `.env.example` file?
*   [ ] Do you have a mechanism such as `/clear` to manage context growth?
*   [ ] Is the `system` prompt consistently placed at the top of the message list?
*   [ ] Is an output limit such as `max_tokens` set to prevent unbounded generation?
*   [ ] Have you verified that your provider handles streaming and usage metadata the way your code expects?

**Project Walkthrough**

Instead of printing the entire application here, open the **Chapter 3 folder** in the book's GitHub repository.

As you run the app locally, focus on how the snippets from this chapter fit together in one file:
1. How `chatbot.py` loads the API key safely.
2. How `OPENAI_BASE_URL` and `OPENAI_MODEL` redirect the same SDK to an OpenAI-compatible provider.
3. How the `while True` loop keeps the message array alive in memory.
4. How the streaming loop separates text chunks from the final usage metadata.
5. How the final assistant reply gets appended back into `messages`.

## Capstone Project

Build a minimal AI ghostwriting prototype based on this Upwork brief: "The user enters a short text outline in the terminal. The app sends it to an LLM provider API, and the system streams back a clean draft. The user can preview the generated text and continue chatting with the bot to revise it. No advanced editing, no database, and no complex UI are required."

Right now, the chatbot is still generic. If you ask it to behave like a narrow domain expert, or to produce strict JSON for another system, it will often drift or hallucinate. In the next chapter, we will look at prompt engineering, context engineering, and techniques for getting more structured outputs.
