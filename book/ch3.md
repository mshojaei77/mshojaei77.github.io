**Chapter 3: Building Your First Chatbot**

### **Chapter Introduction: The Engineering Premise**
*   **The Problem:** Many beginners assume that when they send a prompt to an LLM, the model automatically "remembers" the conversation. The reality is that in the message-array pattern used in this chapter, each request should be treated as stateless. Unless your app explicitly sends prior conversation state, the model will have zero memory of what was said seconds prior. Furthermore, waiting for a long response to generate creates a freezing, unresponsive user experience (UX).
*   **The Reality:** A chatbot is not a magic brain; it is essentially a `while` loop that continuously packages a growing transcript of text and sends it to the API or model provider. Treating the model like a conversational black box leads to skyrocketing token costs, failed API requests, and unhandled network errors.
*   **The Goal:** By the end of this chapter, you will understand how to build a clean, production-minded prototype of a ChatGPT-like interface in pure Python. You will implement short-term conversation memory, secure your API keys, stream text back to the screen chunk-by-chunk to reduce perceived latency, and implement basic token tracking with optional cost estimation.

Before opening the full project, make sure you understand:
1. a single API call,
2. a message array,
3. a Python `while` loop,
4. streaming chunks,
5. appending the final assistant reply back to history.

> **Code Repository**  
> The complete, runnable project for this chapter, implemented as a single-file CLI chatbot (`chatbot.py`), is available in the book repository:  
> `https://github.com/mshojaei77/llm-engineering-in-action/tree/main/chapter-03-chatbot`  
>
> Full code changes faster than concepts. This book teaches the core engineering mechanics; the repository holds the runnable implementation. In the chapter below, we show only the important snippets so you can understand each moving part.

---

### **A Note on APIs: Chat Completions vs. Responses**

Before we write code, we need to clear up one source of confusion: modern LLM providers often expose more than one API for generating text. In OpenAI's ecosystem, two important interfaces are the **Chat Completions API** and the newer **Responses API**.

The **Responses API** is OpenAI's newer recommended API for new projects. It is designed as a unified interface for text generation, multimodal input, reasoning models, built-in tools, and more agent-like workflows. OpenAI describes it as an evolution of Chat Completions and recommends it for new projects, while still continuing to support Chat Completions.

So why does this chapter still teach the **message-array pattern** used by Chat Completions?

Because our goal in this chapter is not to build an agent, use web search, call tools, process files, or manage complex multimodal workflows. Our goal is to learn the universal mechanics of a basic chatbot:

```text
messages -> model call -> streamed answer -> save assistant reply -> repeat
```

The Chat Completions style makes that loop very easy to see. It uses a simple list of messages with roles such as `system`, `user`, and `assistant`. Many providers and routing gateways also support OpenAI-compatible chat-completion-style APIs, so this pattern remains useful when you are learning provider portability. OpenAI's migration guide notes that Chat Completions uses an array of messages, while Responses uses a more general system of typed Items that can represent messages, function calls, tool outputs, and other actions.

An **OpenAI-compatible provider** is a service that accepts the same request shape as the OpenAI SDK or OpenAI REST API, even if the request is actually served by another company or routing layer. In practice, that means your code can often stay the same while you change only configuration such as:

```text
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-5-nano
```

This chapter uses **OpenRouter** as the concrete example. The Python code still calls `client.chat.completions.create(...)`, but the traffic is sent to OpenRouter's OpenAI-compatible endpoint instead of OpenAI's default API host.

The key difference for this chapter is **state management**. With Chat Completions, conversation state must be managed manually: your Python app stores the message history and sends the relevant history again on every request. The Responses API can instead chain responses using features such as stored state or previous response IDs, which is powerful for more advanced applications.

For beginners, manual state is a feature, not a weakness. It forces you to understand what a chatbot really is: an application that manages conversation history, controls token growth, streams model output, and decides what context the model sees. Those skills remain valuable even if you later migrate to the Responses API, because the underlying engineering questions do not disappear.

In this chapter, we will use the message-array pattern because it is simple, visible, and widely compatible. Later chapters will cover the more advanced reasons you might choose Responses-style APIs: structured outputs, tool calling, agents, file search, web search, and richer state management.

> **Practical rule:**
> Use the message-array pattern to learn the fundamentals. Use the Responses API when you need modern OpenAI-native features such as built-in tools, richer reasoning support, multimodal workflows, or stateful response chains.

---

### **3.1 The Core Concept: The Message Array and the Loop**
*   **Mental Model:** Think of a chatbot like reading an entire theatrical script to an actor, asking them to improvise the very next line, writing that line down at the bottom of the script, and then reading the entire script to them again for the next turn.
*   **Simple Theory:** To achieve this, your Python app maintains a list of dictionaries called the **Message Array**. Each dictionary contains a `role` and `content`.
    *   `developer` (or `system`): The app-level instruction (e.g., "You are a helpful assistant"). *Note: In OpenAI's newer reasoning-model APIs, the official role for instructions is now called `developer`. Older models and many OpenAI-compatible providers usually expect `system`. We will use `system` in this chapter for broad compatibility.*
    *   `user`: The human's input.
    *   `assistant`: The model's response.

    When a user types a prompt, you append it to the list. You send the entire list to the API. The API returns an `assistant` message. You append that response to the list, and repeat.
    
*   **Suggested Diagram:**
    ```text
    +--------------+
    | CLI terminal |
    +------+-------+
           | user prompt
           v
    +------------------+
    | messages list    |
    | system/user/AI   |
    +------+-----------+
           | API request
           v
    +------------------+
    | LLM Provider API |
    +------+-----------+
           | streamed text
           v
    +--------------------------+
    | streamed terminal output |
    +--------------------------+
    ```

---

### **3.2 Engineering Realities: Bottlenecks & Trade-offs**
*   **Latency Impact:** Generating a 500-word response can take 10+ seconds. If your app waits for the entire generation to finish before updating the UI, the user will think the app crashed. We solve this bottleneck by enabling **streaming** (`stream=True`). This forces the API to return chunks of text using Server-Sent Events (SSE) the moment they are generated, drastically improving Time-to-First-Token (TTFT).
*   **Context-Window and Token-Budget Constraints:** Because you are resending the entire conversation history with every new message, you will eventually hit the model's context-window limit. If you exceed this limit, the API may return an error or fail the request.
*   **Cost Drivers:** API providers charge you for every token processed. If you have 10 messages in your history, you are paying to process the 1st message for the 10th time. Long conversations inherently cost more per turn than short conversations.

---

### **3.3 Hands-on Implementation: The Core Snippets**

To build our chatbot, we use plain **Python** in the terminal. That keeps the architecture minimal and forces us to understand the actual chatbot loop before adding any web framework.

Here are the five core engineering mechanics you must understand to make it work.

#### 1. Secure Secret Management
Never hardcode your API key directly in Python files, notebooks, or frontend code. Secrets should live outside the codebase and be loaded at runtime. In this chapter, we use the `python-dotenv` library to read credentials from a local `.env` file, and that file must be listed in `.gitignore` so it never gets committed by accident.

*Tip:* A professional repository usually includes a `.env.example` file. It documents which environment variables are required without exposing any real credentials. Commit `.env.example`; never commit `.env`. If you are using an OpenAI-compatible provider such as OpenRouter, document the `OPENAI_BASE_URL` and provider-specific model name there as well.

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
In a CLI chatbot, there is no browser session and no framework-managed state object. Instead, we keep the conversation in a normal Python list and reuse that list inside a `while True` loop. As long as the process keeps running, the conversation history remains in memory.

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
Once we have the array, we pass it to the provider. We set an output token limit to prevent unbounded, expensive generations, and we enable `stream=True`.

*Note: Different APIs may call this setting `max_tokens`, `max_completion_tokens`, or `max_output_tokens`; always check the provider docs.*

Crucially, **streaming APIs do not return token usage by default**. You must explicitly pass `stream_options={"include_usage": True}` to force the API to send a final metadata chunk containing your token costs.

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
Streaming does not return text one perfect word at a time; it returns *chunks* or *deltas*. In a CLI app, we print each chunk directly to the terminal while simultaneously tracking input and output tokens.

The SDK returns chunks where `chunk.choices[0].delta.content` contains the text. When `include_usage` is `True`, the very last chunk will usually have an empty `choices` array but a populated `usage` object. Usage data from streaming is best-effort; if the stream is interrupted before the final chunk arrives, token usage may be missing.

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

Because we accumulate the streamed chunks into `reply_parts`, we still end the turn with one final assistant string that can be appended back into `messages`.

#### 5. Appending the Final Reply Back to History
The model does not remember what it said unless you save that answer yourself. After streaming is complete, append the final assistant reply to the same message array so the next API call includes the latest turn.

```python
messages.append({"role": "assistant", "content": assistant_reply})
```

---

### **3.4 Advanced Patterns & Specialized Use Cases**

*   **Context Management (Trimming History):** As users chat, the `messages` list grows indefinitely. You must manually trim the list before sending it to the API to prevent context errors and cost spikes.
    ```python
    def trim_messages(messages, max_messages=12):
        system = messages[:1]
        history = messages[1:]
        recent = history[-max_messages:]
        return system + recent
    ```

*   **Cost Awareness Tracking:** API tokens dictate your bill. Providers charge different rates for input (prompt) tokens and output (completion) tokens. You can estimate session costs locally using the data collected during the stream:
    ```python
    def estimate_cost(input_tokens, output_tokens, input_price, output_price):
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        return input_cost + output_cost
    ```

*   **Edge Cases & Error Handling:**
    *   *Network Timeouts & Outages:* If the provider goes down, the API call will fail unless you handle the exception. Wrap the API call in a `try...except` block. Do not expose raw stack traces to the user.
    *   *Rate Limit Error (HTTP 429):* If users spam the chat, the provider will temporarily block you. In production, you will need exponential backoff.
    *   *Session Memory is Not Permanent:* A CLI chatbot only keeps memory in RAM while the Python process is alive. If the user exits the program or the process crashes, the conversation disappears. We will tackle permanent database memory in later chapters.
    *   *CLI Control Commands:* In a terminal app, commands such as `/clear`, `/stats`, and `/exit` are a practical replacement for buttons and UI widgets.

---

### **3.5 The Decision Matrix / Artifact**

*   **Selection Table: Modern Chat APIs**
    *Note: OpenAI recommends the newer "Responses API" for new projects requiring seamless multi-turn interactions. However, the "Chat Completions API" using the message array above remains widely supported and is still the most familiar cross-provider pattern.*

    | Approach | When to use it | When to avoid it |
    | :--- | :--- | :--- |
    | **Message Array (Chat Completions)** | Best for simple apps, standard text generation, and easier swapping through compatible APIs or gateways. | Complex agentic workflows with multiple stateful tools. |
    | **Blocking API (No Stream)** | Best for offline backend tasks, JSON extraction, and evaluating data silently. | User-facing chat interfaces. |
    | **Streaming API** | Best for interactive tools such as CLI chatbots, desktop apps, and web chat interfaces where fast feedback matters. | Background tasks where UI updates are irrelevant. |

*   **OpenAI-Compatible Provider Notes:** Services like **OpenRouter** let you keep the OpenAI SDK and the same `chat.completions.create(...)` call while routing traffic through a different endpoint. The usual migration is configuration-level, not architecture-level: set `OPENAI_BASE_URL=https://openrouter.ai/api/v1` and choose a provider-style model ID such as `openai/gpt-5-nano`. That said, compatibility is rarely perfect. Always test streaming behavior, usage reporting, supported parameters, and role handling before deploying.

*   **Engineering Checklist:**
    *   [ ] Are API keys stored strictly in a `.env` file or secrets manager?
    *   [ ] Is `.env` included in `.gitignore`?
    *   [ ] Is there a `.env.example` file committed to the repository?
    *   [ ] Is there a mechanism like a `/clear` command to manage the context window?
    *   [ ] Is the `system` prompt positioned consistently at the top of the message list?
    *   [ ] Is an output token limit such as `max_tokens` set to prevent unbounded generation?
    *   [ ] Have you verified that your OpenAI-compatible provider handles streaming and usage metadata the way your code expects?

---

### **3.6 Final Project Files**

Instead of printing the full application here, open the **Chapter 3 folder** in the book's GitHub repository.

Your goal is to run the app locally and study how the snippets we just covered fit together in one file. Pay special attention to these five mechanics:
1. How `chatbot.py` safely loads the API key.
2. How `OPENAI_BASE_URL` and `OPENAI_MODEL` redirect the same SDK to an OpenAI-compatible provider.
3. How the `while True` loop keeps the message array alive in memory.
4. How the streaming loop isolates text chunks from the final usage metadata.
5. How the final assistant reply is appended back into `messages`.

---

### **Chapter Summary & Transition**
*   **Recap:** You have successfully learned the architecture of a chatbot. You learned that chatbots are autoregressive loops that rely on developers to explicitly manage and pass short-term memory. You mitigated latency by streaming server-sent events, captured separated input and output token usage for basic cost awareness, and secured your infrastructure by pulling credentials from the environment.
*   **What's Next:** Right now, your chatbot is generic. If you ask it to act like a highly specific domain expert, or ask it to format an output as strict JSON for a database, it will likely fail or hallucinate. In **Chapter 4**, we will explore the engineering science behind prompt engineering, context engineering, and forcing the LLM to return structured outputs.

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
