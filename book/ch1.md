# Chapter 1: LLM Text Generation

For the first year of the generative AI boom, most developers treated large language models (LLMs) like a magical REST API: you send a string of text, wait a few seconds, and an intelligent response comes back.

If you are building a weekend side project, that mental model is good enough. But if you are building production systems, treating an LLM as a black box is a recipe for expensive mistakes. If you do not understand how an LLM generates text, you will struggle with cost control, latency tuning, GPU memory limits, and debugging hallucinations or malformed outputs.

This chapter demystifies the "magic." We are skipping the calculus and academic history. Instead, we are looking at the lifecycle of a prompt from text, to numbers, through the neural network, and back to text, and what that means for system design.

## The Generation Loop

Move away from the idea that an LLM "thinks" about a problem and then writes an essay. At its core, an LLM is a statistical engine designed to do one thing well: **predict the next token based on the tokens that came before it**.

This process is called **autoregressive generation**. If you want a practical mental model, think of it as the `while` loop of generative AI.

Here is the basic loop:
1. The model takes your prompt, for example, `"The sky is"`.
2. It computes a probability distribution over possible next tokens.
3. It selects one token, such as `" blue"`.
4. It appends that token to the sequence.
5. It feeds the updated sequence back through the model.
6. It repeats until it hits a stop condition, a token limit, or an explicit termination token.

That loop sounds simple, but it explains most production behavior. Generation is sequential, so every new token adds latency. Long outputs cost more because you pay for every generated token. Bad prompt structure compounds because each new token becomes part of the context for the next step.

This same loop powers multiple model behaviors you will encounter in practice:
*   **Base models** continue text in the most statistically likely way.
*   **Instruct models** are fine-tuned to behave like assistants and follow requests.
*   **Reasoning models** often spend more tokens on intermediate reasoning before returning an answer.

The loop is the same in all three cases. What changes is the distribution the model has learned to produce.

## Tokens and Subwords

Neural networks do not process raw English. Before text reaches the model, it must be converted into discrete units called **tokens**.

A token is not always a word. Modern LLMs usually rely on **subword tokenization**, which breaks text into pieces that are efficient for the model to process. Common words may map to a single token, while rare words, code identifiers, or unusual names may be split into multiple tokens.

Why not tokenize by whole words? Because the vocabulary would explode in size. New words, typos, domain jargon, and multilingual text would make the lookup table too large and too brittle.

Why not tokenize by single characters? Because sequences would become too long, which would make computation slower and weaken semantic meaning.

The compromise used in many modern systems is subword tokenization, often implemented with schemes such as **Byte-Pair Encoding (BPE)** or related tokenization algorithms. A rough rule of thumb in English is that **1 token is often about 0.75 words**, though the exact ratio varies widely by language and formatting.

Tokenization has direct engineering consequences:
*   Billing is based on tokens, not words.
*   Long code snippets, JSON payloads, and multilingual inputs can consume tokens much faster than plain English prose.
*   Context limits are enforced in tokens, so prompt length must be measured at the tokenizer level.

Tokenizers also insert **special tokens** that are invisible to users but important to the model. These can mark boundaries between system prompts, user messages, assistant messages, or end-of-sequence signals.

Tokenization also explains some famous model weaknesses. If the model sees text as token IDs rather than as literal letters, it may struggle with tasks like counting characters inside a word or reasoning about exact string shapes.

## Decoder-Only Transformers

If autoregressive generation explains *how* text is emitted, the model architecture explains *why* generation has the performance profile it does.

The dominant architecture behind modern text-generation systems is the **Transformer**, introduced in the 2017 paper *Attention Is All You Need*. Transformers replaced older recurrent architectures by processing sequences in parallel during training, which made scaling practical.

There are three broad Transformer families:

**Encoder-only models** read the full input bidirectionally. They are excellent for understanding and representation tasks such as embeddings, classification, and retrieval.

**Encoder-decoder models** read the input with an encoder and generate output with a decoder. They are useful for sequence-to-sequence tasks such as translation, transcription, and summarization.

**Decoder-only models** read from left to right and are optimized for next-token prediction. This is the architecture most people mean when they talk about modern LLMs used for chat and generation.

A decoder-only prompt lifecycle looks like this:
1. Raw text is tokenized into integer IDs.
2. Each token ID is converted into an embedding vector.
3. Positional information is injected so the model can distinguish order.
4. Transformer blocks update each token representation using attention.
5. The final layer outputs scores for the next possible token.
6. A decoding strategy selects the next token.
7. The loop starts again.

<img width="818" height="218" alt="image" src="https://github.com/user-attachments/assets/9fec574e-cd9b-4a14-b4b2-2cb3ec444bd3" />

For a software engineer, architecture is not trivia. It determines which tasks a model is suited for, how it uses memory, and where latency and cost come from.

## Context Windows and Attention

If token embeddings represent *what* tokens are, **attention** represents how they relate to one another in context.

Consider the word `bank` in these two sentences:
*   `I sat by the river bank.`
*   `I deposited money in the bank.`

The token is the same, but the meaning changes based on surrounding words. Through **self-attention**, the model updates the representation of each token by looking at other relevant tokens in the sequence.

Modern LLMs use **multi-head attention**, which means the model learns several different patterns of attention at once. One head may track syntax, another long-range references, and another factual associations.

Because text generation is autoregressive, decoder-only models apply **causal masking**. A token can only attend to earlier tokens, not future ones. That is what prevents the model from "cheating" during next-token prediction.

This leads to one of the most important concepts in production AI: the **context window**. The context window is the maximum number of tokens the model can consider at one time. If your system prompt, retrieved documents, chat history, and user message exceed that limit, something must be truncated, summarized, or dropped.

Attention also explains why long prompts are expensive. In standard attention, each token interacts with many other tokens in the sequence, so compute and memory costs rise sharply as context grows. In practice, long contexts increase latency, GPU memory pressure, and serving cost.

During generation, inference engines avoid recomputing the entire history from scratch by using a **key-value cache (KV cache)**. The model stores intermediate attention states for previous tokens in GPU memory and reuses them for each next-token step.

That cache is one of the main reasons long contexts are expensive in production. The longer the prompt and the more concurrent users you serve, the larger the KV cache becomes. When an inference server hits an out-of-memory error, the KV cache is often a major factor.

## Logits and Probabilities

After the prompt has passed through the Transformer network, the model produces **logits** for the next token. A logit is a raw numerical score for each possible token in the vocabulary.

Those scores are not probabilities yet. To convert them into probabilities, we apply **softmax**, which turns the logits into a distribution whose values add up to 1.

For example, after the prompt `"The sky is"`, the model might internally assign probabilities like this:
*   `" blue"`: 0.80
*   `" gray"`: 0.12
*   `" falling"`: 0.03
*   other tokens: the remaining probability mass

The model then has to choose what to do with that distribution. If it always picks the highest-probability token, the output becomes more deterministic. If it samples from the distribution, the output becomes more varied.

This is the bridge between model internals and product behavior. Hallucinations, repetition, creativity, and formatting drift all emerge from how the model's probability distribution is shaped and how the decoding step samples from it.

For engineers, the practical takeaway is simple: the model never "knows" the answer in a human sense. It continually assigns probabilities to token continuations and emits one token at a time.

## Decoding Parameters

Once you have logits and probabilities, you need a policy for selecting the next token. That policy is the **decoding strategy**, and API parameters let you control it.

The most important decoding controls are:

**Temperature and `top_p`**  
`temperature` adjusts how sharp or flat the probability distribution feels during sampling. Lower values make outputs more deterministic. Higher values increase variation. `top_p`, also called nucleus sampling, limits sampling to the smallest set of tokens whose cumulative probability reaches the threshold `p`.

**Frequency and presence penalties**  
`frequency_penalty` reduces the chance of repeating tokens that have already appeared often in the generated text. `presence_penalty` reduces the chance of revisiting concepts that have appeared at all, even if they were not repeated heavily.

**Maximum token limits and stop sequences**  
`max_tokens` or `max_completion_tokens` acts as a hard ceiling on output length. This is a cost-control and latency-control tool. `stop` lets you define strings or markers where generation should halt.

**Seed and logit bias**  
`seed` can help make testing more reproducible in systems that support deterministic sampling. `logit_bias` lets you nudge the probability of specific tokens up or down, which can be useful for constrained behaviors, though it is usually a sharp tool and easy to misuse.

**Structured output and streaming**  
`response_format` or equivalent structured-output controls help force the model into valid JSON or other schemas. `stream` returns partial tokens as they are generated, reducing perceived latency for end users.

Good defaults depend on the task. Extraction, classification, and tool invocation usually need lower randomness. Brainstorming, creative writing, and open-ended ideation can tolerate more sampling diversity.

## Generation Failure Modes

Text generation fails in predictable ways, and production systems need to anticipate them.

**Hallucination**  
The model produces plausible-sounding but unsupported content. This often happens when the prompt is ambiguous, the necessary facts are missing from context, or the model is pushed beyond what it actually knows.

**Repetition and looping**  
Poor decoding settings or weak prompt structure can cause the model to repeat phrases, restate itself, or get stuck in loops.

**Truncation**  
The response stops early because it hit a token limit, a stop sequence, or a context constraint.

**Format drift**  
The model was asked for JSON, SQL, markdown, or another strict format but gradually deviates from the contract. This is common when prompts are underspecified or the output is long.

**Latency blowups**  
Long prompts, reasoning-heavy outputs, and large completion limits can make a system feel broken even when it is technically functioning correctly.

**Memory failures**  
Long context windows and high concurrency increase KV-cache pressure and can trigger GPU out-of-memory errors on self-hosted systems.

The engineering lesson is that generation quality is not just a model issue. It is a systems issue. Prompt design, retrieval quality, context management, decoding parameters, output validation, and serving infrastructure all shape the final result.
