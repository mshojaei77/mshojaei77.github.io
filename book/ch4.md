# Chapter 4: Prompting and Structured Outputs

Prompt engineering used to sound like a bag of tricks: add a dramatic role, ask the model to "think step by step," threaten it with penalties, or wrap the request in a clever phrase.

That is not production prompt engineering.

In production, prompts are **runtime contracts** between your application and a probabilistic model. They define the task, the authority hierarchy, the context the model may use, the constraints it must follow, and the exact data structure your software expects in return.

This chapter moves from ad-hoc prompting to **prompt architecture**:

*   how to separate instructions from untrusted user data
*   how to order context to optimize for model attention and cache latency
*   how to use JSON Schema and Constrained Decoding to guarantee output shapes
*   how to use Pydantic to turn model text into validated Python objects
*   how to retry, version, evaluate, and debug prompts like real software artifacts

The goal is not to make the model sound clever. The goal is to make it a reliable component inside an automated pipeline.

---

## Production Prompt Anatomy

A production prompt is a structured document, not a wall of text. When a prompt fails in a production pipeline, it usually fails due to ambiguity or instruction dilution.

The simplest useful framework for defining this structure is **KERNEL**:

| Letter | Meaning | Engineering Question |
| :----- | :------ | :------------------- |
| **K** | Keep it simple | Can the model understand the task without redundant rules? |
| **E** | Easy to verify | Can your code check whether the response is acceptable? |
| **R** | Reproducible | Will the same prompt behave consistently across test cases? |
| **N** | Narrow scope | Is the task small enough for one model call? |
| **E** | Explicit constraints | Are boundaries, refusals, and "do not" rules clear? |
| **L** | Logical structure | Are role, task, context, and output format cleanly separated? |

A weak prototype prompt looks like this:

```text
Analyze this email and tell me if it is important.

{{email}}
```

A production version uses clear structural boundaries—often XML tags—so the model knows exactly where your instructions end and the data begins:

```text
You are a support triage assistant.

Task:
Classify the customer email by urgency and required department.

Priority order:
1. Use only evidence from the email.
2. If evidence is missing, mark the field as unknown.
3. Keep explanations short.

<email>
{{email}}
</email>

Output Requirements:
Return only valid JSON matching the provided schema.
```

The exact delimiter style is less important than the discipline. XML-style tags (`<email>...</email>`) are standard because they appear frequently in model training data and make sections visually obvious.

However, delimiters are **not a strict security boundary**. A malicious document can still contain instructions. Delimiters help the model's attention mechanism distinguish instructions from data, but your application must still validate outputs and restrict what the model can do.

---

## System Prompt Design

In chat-style APIs, every major provider allows you to send an array of messages with distinct roles (usually `system`, `user`, and `assistant`). The `system` message operates with the highest authority.

Use the system prompt for stable application rules:

*   the model's exact role
*   the task boundaries and safety constraints
*   what to do when evidence is missing
*   output format requirements

Do not use the system prompt for:

*   raw user input
*   retrieved documents (RAG)
*   secrets, API keys, or hidden business data
*   highly dynamic content

The most common architectural mistake is mixing trusted instructions and untrusted data in the same high-authority message. If a user's input or a retrieved PDF can change the text of your system prompt, you have handed untrusted data the strongest steering wheel in your application, opening the door to **prompt injection**.

---

## Instruction Hierarchy and "Lost in the Middle"

LLMs process tokens sequentially and use attention mechanisms to weigh inputs. As context windows have expanded to hundreds of thousands of tokens, a phenomenon known as the **"Lost in the Middle" effect** has emerged.

A highly cited 2023 paper (*Lost in the Middle: How Language Models Use Long Contexts*, Liu et al.) demonstrated that language models are highly accurate at retrieving information at the very beginning and the very end of their input context. However, their performance degrades significantly when relevant information is buried in the middle of a long prompt.

**Engineering Consequence:**
You must design your prompt hierarchy deliberately.

1.  **Top:** Place your most critical system constraints, roles, and formatting rules at the absolute top of the prompt.
2.  **Middle:** Place bulky, noisy data—like retrieved RAG documents, chat history, or long tool outputs—in the middle.
3.  **Bottom:** Place the immediate user query at the bottom. If an output rule is completely non-negotiable (e.g., "Respond in valid JSON only"), repeat it at the very bottom of the prompt, just before the model generates its response.

---

## Few-Shot Examples and Prompt Caching

Models learn better by imitation than by instruction. Providing two or three examples of "input → output" pairs reduces variance more than any paragraph of text description.

```text
<example>
Input: "The dashboard has been down since 9 AM."
Output: {"topic": "technical", "urgency": "high", "missing_fields": ["user_id"]}
</example>
```

Historically, engineers hesitated to use large sets of examples because of token costs and latency. That changed with the introduction of **Prompt Caching** (supported natively by Anthropic, OpenAI, and open-source engines like vLLM).

Prompt caching allows the inference engine to cache the prefix of your prompt. If a new API request shares the exact same starting tokens as a recent request, the engine skips processing those tokens, resulting in up to **90% cost savings** and **85% lower Time-to-First-Token (TTFT) latency**.

**Pattern: Static-to-Dynamic Ordering**
To utilize caching, your prompts must be ordered from static to dynamic.
*   **Cacheable Prefix:** System instructions, large schema definitions, and few-shot examples.
*   **Dynamic Suffix:** The current user query or retrieved context.

If you inject a dynamic variable (like a unique request ID or the current timestamp) at the *top* of your system prompt, you will break the exact-prefix match and miss the cache on every single request.

---

## Constrained Decoding and JSON Schemas

If another program consumes the model's response, parsing free-text with regular expressions is brittle. A single unexpected space, a Markdown code block, or a conversational prefix ("Here is your JSON:") will crash your application.

There are three levels of structured output reliability:

| Level | Feature | Reliability |
| :---- | :------ | :---------- |
| **1. Prompt-only** | You ask the model to return JSON in text. | Brittle. Often includes conversational filler. |
| **2. JSON Mode** | Provider guarantees the output parses as JSON. | Good, but does not guarantee the keys or types match your schema. |
| **3. Constrained Decoding** | Provider mathematically forces the output to match your schema. | Production standard. |

**Constrained Decoding** (released natively in OpenAI's API in August 2024 as "Structured Outputs", and supported in open-source via vLLM's `xgrammar` or `outlines` backends) uses a Finite State Machine (FSM) at the inference engine level.

Before the model generates the next token, the FSM checks the JSON schema. If your schema requires an integer, the engine masks the probability (logits) of all alphabetical tokens to exactly zero. The model is mathematically forced to output a valid structure.

### The Reasoning-First Pattern
LLMs do not have an internal scratchpad; their only way to "think" is to output tokens. If your JSON schema asks for a `status` before a `reasoning` field, the model has to guess the status first, and will then use the reasoning field to rationalize its guess—even if the guess was wrong.

**Always place reasoning fields at the top of your schema.**

```json
{
  "properties": {
    "step_by_step_reasoning": { "type": "string" },
    "final_status": { "enum": ["approved", "rejected"] }
  }
}
```
Because JSON is generated top-to-bottom, this forces the model to perform Chain-of-Thought analysis *before* committing to a label.

---

## Pydantic Validation Contracts

In Python, Pydantic is the industry standard for defining output schemas. Rather than writing raw JSON schemas, you define a strongly-typed Python class.

Using OpenAI's modern `.parse()` SDK feature, you can pass the Pydantic model directly to the API. The SDK handles generating the schema, applying constrained decoding, and parsing the result back into a Python object.

```python
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()

class ContactExtraction(BaseModel):
    # Reasoning comes first!
    priority_reason: str = Field(description="Evidence-based reason for priority.")
    priority_score: int = Field(description="Urgency score from 1 to 10.")
    department: str = Field(enum=["billing", "technical", "sales", "other"])
    needs_human: bool

# The .parse() method enforces the Pydantic schema
completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": "Extract data from the customer email."},
        {"role": "user", "content": "I cannot log in to the dashboard."}
    ],
    response_format=ContactExtraction,
)

# Returns a validated Pydantic object, not a raw string
contact = completion.choices[0].message.parsed
print(contact.department) # Output: technical
```

For open-source models or providers without native `.parse()`, libraries like `Instructor` or `Marvin` provide identical wrappers.

---

## Retry and Fallback Logic

Constrained decoding guarantees *syntactic* correctness (the output will parse, and the types will match). It does not guarantee *semantic* correctness. The model might still give a "priority 10" to a minor typo, violating your business logic.

When semantic validation fails, you need a resilient retry loop.

```python
from pydantic import ValidationError

def extract_contact_with_retry(email_text: str, max_attempts: int = 2):
    repair_notes = []

    for attempt in range(max_attempts):
        # 1. Call model with current instructions + repair notes
        response = call_model(email_text, repair_notes)
        
        try:
            # 2. Run custom business validation
            contact = ContactExtraction.model_validate_json(response)
            if contact.priority_score > 5 and "urgent" not in contact.priority_reason:
                raise ValueError("High priority requires explicit urgent reasoning.")
            return contact
            
        except (ValidationError, ValueError) as exc:
            # 3. Feed the exact error back to the model
            repair_notes.append(f"Validation failed: {exc}. Please correct the output.")

    raise RuntimeError("Extraction failed validation after retries.")
```

**Retry Rules:**
*   LLMs are excellent at self-correcting when given the exact validation error string.
*   Cap retries to 2 or 3. Infinite loops cause outages and massive bills.
*   If a cheap model fails after two retries, escalate the fallback request to a heavier model.

---

## Failure Modes: Engineering Pitfalls

| Failure Mode | Cause | Fix |
| :----------- | :---- | :-- |
| **Instruction Dilution** | Critical rules buried in the middle of long RAG context. | Repeat non-negotiable rules at the very end of the prompt. |
| **Cache Miss Spikes** | Dynamic data (timestamps, UUIDs) placed at the top of the prompt. | Order prompts static-to-dynamic. Move variables to the bottom. |
| **Reasoning Lag** | Model picks a label, then hallucinates logic to match it. | Move the `reasoning` field to the absolute top of the JSON schema. |
| **Prompt Injection** | Untrusted data treated as instructions. | Use XML delimiters, separate System/User roles, validate outputs. |
| **Token Truncation** | The model hits `max_tokens` before closing the JSON object. | Set `max_tokens` significantly higher than your expected schema size. |

---

## Hands-On Exercise: Build a Validated Extractor

**Goal:** Build a script that extracts structured data from an unstructured email, utilizing prompt caching ordering, constrained decoding, and a semantic retry loop.

### Requirements

1.  **Define a Pydantic Model** called `TicketExtraction` with fields:
    *   `rationale`: string (Must be placed first)
    *   `category`: Literal["bug", "feature_request", "billing"]
    *   `confidence_score`: float (0.0 to 1.0)
2.  **Order for Caching:** Write a `System` message that defines the rules and includes two few-shot examples. Ensure no dynamic variables are in this message.
3.  **Constrained Decoding:** Use OpenAI's `client.beta.chat.completions.parse()` (or an open-source equivalent via `instructor`) to force the model to adhere to the schema.
4.  **Semantic Validation:** Write a Python check that raises an error if `category` is "bug" but the `confidence_score` is less than 0.5.
5.  **Retry Loop:** Catch the error and append a user message saying: *"Validation Error: If category is bug, confidence must be >= 0.5. Adjust category or confidence."* Send it back to the model for correction.

**Test Input:**
Pass in an ambiguous email: *"The screen flickers when I click save, but maybe that's just a new animation design?"* Observe how the model's rationale step guides its initial classification, and how the retry loop enforces your business logic.