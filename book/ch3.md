**Chapter 3: Building Your First Chatbot**

### **Chapter Introduction: The Engineering Premise**
*   **The misunderstanding:** A lot of beginners assume the model somehow "remembers" the conversation on its own. In the message-array pattern used in this chapter, that is not how it works. Every request is effectively stateless. If your app does not resend the earlier conversation, the model has no idea what was said a few seconds ago. On top of that, if you wait for a long answer to finish before showing anything, the app feels frozen.
*   **What is really happening:** A chatbot is not a magical conversational brain. It is a loop that keeps packaging the conversation history and sending it to a model provider. If you treat the model as a black box, costs climb fast, requests start failing, and network issues turn into broken UX.
*   **What you will build:** By the end of this chapter, you will have a clean, production-minded Python prototype of a ChatGPT-style CLI. You will add short-term conversation memory, keep API keys out of your source code, stream responses as they are generated, and track basic token usage with optional cost estimates.

Before opening the full project, make sure you are comfortable with:
1. a single API call,
2. a message array,
3. a Python `while` loop,
4. streaming chunks,
5. appending the final assistant reply back to history.

**Code Repository**  
The complete, runnable project for this chapter, implemented as a single-file CLI chatbot (`chatbot.py`), is available in the book repository:  
[github.com/mshojaei77/llm-engineering-in-action/chapter-03-chatbot](https://github.com/mshojaei77/llm-engineering-in-action/tree/main/chapter-03-chatbot)
Code evolves faster than prose. This chapter focuses on the underlying mechanics; the repository contains the runnable version. In the pages below, we keep only the snippets that matter so you can see how each part works.

---

### **A Note on APIs: Chat Completions vs. Responses**

Before we write any code, we need to clear up a common source of confusion: modern LLM platforms often offer more than one way to generate text. In OpenAI's ecosystem, two important interfaces are the **Chat Completions API** and the newer **Responses API**.

The **Responses API** is OpenAI's newer recommended interface for greenfield work. It is designed to handle text generation, multimodal input, reasoning models, built-in tools, and more agent-like workflows through one API. OpenAI presents it as the evolution of Chat Completions and recommends it for new projects, while still supporting Chat Completions.

So why does this chapter still teach the **message-array pattern** from Chat Completions?

Because this chapter is not about agents, tool use, web search, file handling, or multimodal orchestration. It is about learning the core loop behind a basic chatbot:

```text
messages -> model call -> streamed answer -> save assistant reply -> repeat
```

The Chat Completions style makes that loop easy to see. It uses a plain list of messages with roles like `system`, `user`, and `assistant`. Many providers and routing layers also expose OpenAI-compatible chat-completion-style APIs, so this pattern is still useful when you want portability across vendors. OpenAI's migration guide makes the distinction clear: Chat Completions works with an array of messages, while Responses uses a more general structure of typed Items that can represent messages, tool calls, function outputs, and other actions.

An **OpenAI-compatible provider** is a service that accepts the same request shape as the OpenAI SDK or REST API, even if another company is actually serving the request. In practice, that often means your code stays the same and only your configuration changes:

```text
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-5-nano
```

This chapter uses **OpenRouter** as the concrete example. The Python code still calls `client.chat.completions.create(...)`, but the request is routed to OpenRouter's OpenAI-compatible endpoint instead of OpenAI's default host.

For this chapter, the important difference is **state management**. With Chat Completions, you manage the conversation state yourself: your Python app stores the message history and resends the relevant part of that history on every turn. The Responses API can chain responses using features like stored state or previous response IDs, which becomes useful in more advanced applications.

For a beginner, manual state is a strength, not a drawback. It forces you to see what a chatbot actually is: an application that manages context, controls token growth, streams output, and decides what the model is allowed to see. Those skills still matter even if you later move to the Responses API, because the engineering trade-offs do not go away.

In this chapter, we use the message-array pattern because it is simple, visible, and widely supported. Later chapters will cover the cases where a Responses-style API makes more sense: structured outputs, tool calling, agents, file search, web search, and richer state management.

> **Practical rule:**
> Use the message-array pattern to learn the fundamentals. Use the Responses API when you need OpenAI-native features such as built-in tools, richer reasoning support, multimodal workflows, or stateful response chains.

---

### **3.1 The Core Concept: The Message Array and the Loop**
*   **A useful mental model:** Imagine handing an actor the full script of a scene, asking for the next line, writing that line at the end of the script, and then reading the whole updated script back to them for the next turn.
*   **How it works in practice:** Your Python app keeps a list of dictionaries called the **message array**. Each dictionary has a `role` and some `content`.
    *   `developer` (or `system`): The app-level instruction, such as "You are a helpful assistant." *Note: In OpenAI's newer reasoning-model APIs, the official instruction role is now `developer`. Older models, and many OpenAI-compatible providers, still expect `system`. We use `system` in this chapter for broad compatibility.*
    *   `user`: The human message.
    *   `assistant`: The model's reply.

    Every time the user types something, you append it to the list. Then you send the full list to the API. The API returns an `assistant` message. You append that reply, then repeat the loop.

<img width="840" height="1680" alt="image" src="https://github.com/user-attachments/assets/2e0917d3-c8f6-4b93-8b70-1ca6bcf7a4e5" />


---

### **3.2 Engineering Realities: Bottlenecks & Trade-offs**
*   **Latency matters:** A 500-word answer can easily take 10 or more seconds to generate. If the interface stays blank until the full response is ready, users will assume the app has stalled. That is why we use **streaming** (`stream=True`). It lets the API send text chunks immediately with Server-Sent Events (SSE), which dramatically improves time-to-first-token.
*   **Context windows are finite:** Because you resend the conversation history every turn, the prompt keeps getting larger. Eventually you will hit the model's context limit. When that happens, the request may fail or the provider may reject it.
*   **Long chats cost more:** Providers bill by token count. If your history contains ten messages, then by the tenth turn you have paid to process the first message ten separate times. The longer the conversation, the more expensive each new turn becomes.

---

### **3.3 Hands-on Implementation: The Core Snippets**

We will build this chatbot in plain **Python** and run it in the terminal. That keeps the architecture simple and makes the chatbot loop impossible to miss before we bring in a web framework.

There are five mechanics you need to understand.

#### 1. Secure Secret Management
Never hardcode your API key in a Python file, notebook, or frontend app. Secrets should live outside the codebase and be loaded at runtime. In this chapter, we use `python-dotenv` to read values from a local `.env` file, and that file must be listed in `.gitignore` so it never gets committed by mistake.

*Tip:* A professional project usually includes a `.env.example` file. It shows which environment variables are required without exposing real credentials. Commit `.env.example`; never commit `.env`. If you are using an OpenAI-compatible provider such as OpenRouter, include `OPENAI_BASE_URL` and the provider-specific model name there as well.

```text
# .env.example
OPENAI_API_KEY=your_openrouter_api_key_here
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-5-nano
```

```python
# loading secrets
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
```

#### 2. Managing Conversation State
In a CLI chatbot, there is no browser session and no framework-managed state object. We simply keep the conversation in a normal Python list and reuse it inside a `while True` loop. As long as the process stays alive, the message history stays in memory.

```python
messages = [
    {"role": "system", "content": "You are a concise, lazy assistant that always answers any query with max 6 words no more."}
]

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})
```

#### 3. The Stateless API Call
Once the array is ready, we send it to the provider. We set an output token limit to prevent runaway generations, and we enable `stream=True`.

*Note: Different APIs name this setting differently. You may see `max_tokens`, `max_completion_tokens`, or `max_output_tokens`, so always confirm in the provider docs.*

One subtle point matters here: **streaming APIs do not usually return token usage unless you ask for it**. Passing `stream_options={"include_usage": True}` tells the API to send a final metadata chunk with usage information.

```python
stream = client.chat.completions.create(
    model="openai/gpt-5-nano",
    messages=messages,
    temperature=0.7,
    max_tokens=1000,
    stream=True,
    stream_options={"include_usage": True},
)
```

#### 4. Streaming Chunk-by-Chunk
Streaming does not deliver one perfect word at a time. It sends *chunks* or *deltas*. In a CLI app, we print each chunk as it arrives while also tracking token usage.

In the SDK, `chunk.choices[0].delta.content` contains the text. When `include_usage` is enabled, the final chunk often has an empty `choices` array but a populated `usage` object. That usage data is best-effort: if the stream is interrupted before the last chunk arrives, the usage block may never show up.

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

Because we collect the streamed chunks into `reply_parts`, we still end up with one complete assistant string that can be stored in `messages`.

#### 5. Appending the Final Reply Back to History
The model will not remember what it just said unless you save that answer yourself. Once streaming finishes, append the final assistant reply to the same message array so the next request includes the latest turn.

```python
messages.append({"role": "assistant", "content": assistant_reply})
```

---

### **3.4 Advanced Patterns & Specialized Use Cases**

*   **Context management (trimming history):** As users keep chatting, the `messages` list grows forever unless you control it. Before each request, trim the history to avoid blowing past the context window or paying for unnecessary tokens.
    ```python
    def trim_messages(messages, max_messages=12):
        system = messages[:1]
        history = messages[1:]
        recent = history[-max_messages:]
        return system + recent
    ```

*   **Basic cost awareness:** Tokens determine the bill. Providers usually charge one rate for input tokens and another for output tokens. With the usage data captured from the stream, you can estimate cost locally:
    ```python
    def estimate_cost(input_tokens, output_tokens, input_price, output_price):
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        return input_cost + output_cost
    ```

*   **Edge cases and error handling:**
    *   *Network timeouts and outages:* If the provider is unavailable, the request will fail unless you handle the exception. Wrap the API call in `try...except`, and do not dump raw stack traces on users.
    *   *Rate limiting (HTTP 429):* If requests come too quickly, the provider will temporarily throttle you. In production, you usually handle this with exponential backoff.
    *   *Session memory is temporary:* A CLI chatbot keeps memory in RAM only while the Python process is running. If the user exits or the process crashes, the conversation is gone. Later chapters will cover persistent memory with a database.
    *   *CLI control commands:* In a terminal app, commands like `/clear`, `/stats`, and `/exit` are the practical equivalent of buttons.

---

### **3.5 The Decision Matrix / Artifact**

*   **Selection table: modern chat APIs**
    *Note: OpenAI recommends the newer Responses API for new projects that need smoother multi-turn interactions. Even so, the Chat Completions message-array pattern remains widely supported and is still the clearest cross-provider starting point.*

    | Approach | When to use it | When to avoid it |
    | :--- | :--- | :--- |
    | **Message Array (Chat Completions)** | Simple chat apps, standard text generation, and projects where cross-provider compatibility matters. | Complex agentic systems with multiple tools and richer state handling. |
    | **Blocking API (No Stream)** | Offline backend jobs, JSON extraction, or evaluation tasks where nobody is waiting on a UI. | User-facing chat interfaces. |
    | **Streaming API** | Interactive experiences such as CLI chatbots, desktop apps, and web chat UIs where responsiveness matters. | Background work where live updates provide no value. |

*   **Notes on OpenAI-compatible providers:** Services like **OpenRouter** let you keep the OpenAI SDK and the same `chat.completions.create(...)` call while routing traffic to another endpoint. In many cases, that is a configuration change rather than an architectural rewrite: set `OPENAI_BASE_URL=https://openrouter.ai/api/v1` and choose a provider-style model ID like `openai/gpt-5-nano`. That said, "compatible" rarely means "identical." Always test streaming behavior, usage reporting, supported parameters, and role handling before you ship.

*   **Engineering checklist:**
    *   [ ] Are API keys stored only in a `.env` file or a proper secrets manager?
    *   [ ] Is `.env` listed in `.gitignore`?
    *   [ ] Is there a committed `.env.example` file?
    *   [ ] Do you have a mechanism such as `/clear` to manage context growth?
    *   [ ] Is the `system` prompt consistently placed at the top of the message list?
    *   [ ] Is an output limit such as `max_tokens` set to prevent unbounded generation?
    *   [ ] Have you verified that your OpenAI-compatible provider handles streaming and usage metadata the way your code expects?

---

### **3.6 Final Project Files**

Instead of printing the entire application here, open the **Chapter 3 folder** in the book's GitHub repository.

As you run the app locally, focus on how the snippets from this chapter fit together in one file:
1. How `chatbot.py` loads the API key safely.
2. How `OPENAI_BASE_URL` and `OPENAI_MODEL` redirect the same SDK to an OpenAI-compatible provider.
3. How the `while True` loop keeps the message array alive in memory.
4. How the streaming loop separates text chunks from the final usage metadata.
5. How the final assistant reply gets appended back into `messages`.

---

### **Chapter Summary & Transition**
*   **Recap:** You now understand the basic architecture of a chatbot. It is an autoregressive loop in which the developer is responsible for storing and resending short-term memory. You reduced perceived latency with streaming, captured input and output token usage for basic cost awareness, and handled credentials in a safer way by loading them from the environment.
*   **What comes next:** Right now, the chatbot is still generic. If you ask it to behave like a narrow domain expert, or to produce strict JSON for another system, it will often drift or hallucinate. In **Chapter 4**, we will look at prompt engineering, context engineering, and techniques for getting more structured outputs.

---

### **Assignment / Capstone Project**
*   **Part 1 Capstone Project: "AI Outline-to-Draft Generator"**
    *   *Freelance Brief (Upwork):* "Build a minimal AI ghostwriting prototype. The user enters a short text outline in the terminal. The app sends it to an LLM provider API, and the system streams back a clean draft. The user can preview the generated text and continue chatting with the bot to revise it. No advanced editing, no database, and no complex UI are required."

---

### **Sources & References**
*   **Primary Sources:**
    * [OpenAI API Reference: Migrate to the Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses)
    * [OpenAI API Reference: Streaming API responses and usage stats](https://developers.openai.com/api/docs/guides/streaming-responses)
*   **Engineering Sources:**
    * [OpenRouter Documentation: OpenAI SDK Compatibility](https://openrouter.ai/docs#quick-start)
    * [OpenAI Help Center: API Key Safety Best Practices](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety)
