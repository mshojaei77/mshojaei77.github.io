from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-5-nano")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1000"))
MAX_HISTORY_MESSAGES = int(os.getenv("OPENAI_MAX_HISTORY_MESSAGES", "12"))
INPUT_PRICE_PER_MILLION = float(os.getenv("INPUT_PRICE_PER_MILLION", "0.15"))
OUTPUT_PRICE_PER_MILLION = float(os.getenv("OUTPUT_PRICE_PER_MILLION", "0.60"))
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "OPENAI_SYSTEM_PROMPT",
    "You are a concise, lazy assistant that always answers any query with max 6 words no more.",
)

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY. Check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)


def trim_messages(messages: list[dict[str, str]], max_messages: int = 12) -> list[dict[str, str]]:
    if not messages:
        return []

    system = messages[:1]
    history = messages[1:]
    return system + history[-max_messages:]


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_price: float,
    output_price: float,
) -> float:
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    return input_cost + output_cost


def stream_reply(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[str, int, int]:
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )

    reply_parts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    print("\nAssistant: ", end="", flush=True)
    for chunk in stream:
        if getattr(chunk, "usage", None):
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

        if chunk.choices and chunk.choices[0].delta.content is not None:
            text = chunk.choices[0].delta.content
            reply_parts.append(text)
            print(text, end="", flush=True)

    print()
    return "".join(reply_parts), prompt_tokens, completion_tokens


def print_help() -> None:
    print("\nCommands:")
    print("  /help   Show available commands")
    print("  /clear  Clear conversation history")
    print("  /stats  Show token and cost totals")
    print("  /exit   Quit the chatbot")


def print_stats(input_tokens: int, output_tokens: int) -> None:
    total_cost = estimate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_price=INPUT_PRICE_PER_MILLION,
        output_price=OUTPUT_PRICE_PER_MILLION,
    )
    print("\nSession stats:")
    print(f"  Input tokens:  {input_tokens:,}")
    print(f"  Output tokens: {output_tokens:,}")
    print(f"  Estimated cost: ${total_cost:.6f}")


def reset_chat(system_prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt}]


def main() -> None:
    messages = reset_chat(DEFAULT_SYSTEM_PROMPT)
    input_tokens = 0
    output_tokens = 0

    print("Chapter 03 CLI Chatbot")
    print(f"Base URL: {OPENAI_BASE_URL}")
    print(f"Model: {MODEL}")
    print("Type /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("Exiting.")
            break
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/clear":
            messages = reset_chat(DEFAULT_SYSTEM_PROMPT)
            print("Conversation cleared.")
            continue
        if user_input == "/stats":
            print_stats(input_tokens, output_tokens)
            continue

        messages.append({"role": "user", "content": user_input})
        request_messages = trim_messages(messages, max_messages=MAX_HISTORY_MESSAGES)

        try:
            reply, prompt_used, completion_used = stream_reply(
                messages=request_messages,
                model=MODEL,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        except RateLimitError:
            print("\nRate limit reached. Please wait a moment and try again.")
            messages.pop()
            continue
        except (APITimeoutError, APIConnectionError):
            print("\nConnection failed while talking to the model provider. Try again.")
            messages.pop()
            continue
        except APIError as exc:
            print(f"\nProvider error: {exc}")
            messages.pop()
            continue
        except Exception:
            print("\nUnexpected error while generating the response.")
            messages.pop()
            continue

        input_tokens += prompt_used
        output_tokens += completion_used
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
