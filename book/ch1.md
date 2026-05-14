**Chapter 1: LLM Text Generation**

For the first year of the generative AI boom, most developers treated large language models (LLMs) like a magical REST API: you send a string of text, wait a few seconds, and an intelligent response comes back.

If you are building a weekend side project, that mental model is good enough. If you are building production systems, treating an LLM as a black box is a recipe for expensive mistakes. If you do not understand how an LLM generates text, you will struggle with cost control, latency tuning, GPU memory limits, output validation, and debugging hallucinations or malformed responses.

This chapter demystifies the "magic." We are skipping the calculus and academic history. Instead, we are following the lifecycle of a prompt from text, to tokens, through the neural network, into probabilities, and back to generated text. The goal is not to become a model researcher. The goal is to understand enough mechanics to build better systems.

## The Generation Loop

Move away from the idea that an LLM "thinks" about a problem and then writes an essay. At its core, an LLM is a statistical engine trained to do one thing extremely well: **predict the next token based on the tokens that came before it**.

This process is called **autoregressive generation**. If you want a practical mental model, think of it as the `while` loop of generative AI.

Here is the basic loop:

1. The model receives your prompt, for example, `"The sky is"`.
2. The text is converted into tokens.
3. The model runs a forward pass and produces raw scores, called logits, for the next token.
4. The logits are converted into a probability distribution.
5. A decoding strategy selects one token, such as `" blue"`.
6. That token is appended to the sequence.
7. The updated sequence becomes the input for the next step.
8. The loop repeats until the model emits an end token, hits a stop sequence, reaches a token limit, or is interrupted by the application.

That loop sounds simple, but it explains most production behavior. Generation is sequential, so every new output token adds latency. Long outputs cost more because you pay for generated tokens. Bad early tokens can push later tokens in the wrong direction because every generated token becomes part of the next step's context.

This last point is called **path dependency**. Once the model emits an incorrect assumption, malformed JSON brace, wrong tool argument, or unsupported claim, the next token is generated on top of that mistake. This is why production systems use structured outputs, validation, retries, retrieval, guardrails, and sometimes human review. They are not decorative features. They counteract the compounding nature of autoregressive generation.

There are two important inference phases:

**Prefill**  
The model processes the full prompt once. This includes the system message, conversation history, retrieved documents, tool results, and the latest user message. Long prompts make prefill slower and increase memory usage.

**Decode**  
After prefill, the model generates one new token at a time. Decode is sequential. This is why a model can ingest a long prompt relatively quickly but still take noticeable time to write a long answer.

Inference engines make the decode phase practical with a **key-value cache (KV cache)**. During prefill, the model stores attention information for previous tokens. During decode, it reuses that cached state instead of recomputing the full history from scratch for every new token. Without a KV cache, long generations would scale much worse.

The same loop powers multiple model behaviors you will encounter in practice:

*   **Base models** continue text in the most statistically likely way.
*   **Instruct models** are fine-tuned to behave like assistants and follow requests.
*   **Reasoning models** often spend more tokens on intermediate reasoning before returning an answer.
*   **Tool-calling models** generate structured arguments that your application uses to call code, APIs, or databases.

The loop is the same in all cases. What changes is the distribution the model has learned to produce and the constraints your application puts around the output.

## Tokens and Subwords

Neural networks do not process raw English. Before text reaches the model, it must be converted into discrete units called **tokens**.

A token is not always a word. Modern LLMs usually rely on **subword tokenization**, which breaks text into pieces that are efficient for the model to process. Common words may map to a single token, while rare words, code identifiers, unusual names, and non-English text may be split into multiple tokens.

Why not tokenize by whole words? Because the vocabulary would explode in size. New words, typos, domain jargon, product names, code symbols, and multilingual text would make the lookup table too large and too brittle.

Why not tokenize by single characters? Because sequences would become too long, which would make computation slower and weaken semantic meaning.

The compromise used in many modern systems is subword tokenization. Common approaches include:

*   **Byte-Pair Encoding (BPE):** Starts from small units and repeatedly merges frequent pairs. Many GPT-style tokenizers use BPE or byte-level BPE variants.
*   **WordPiece:** Similar in spirit to BPE and historically common in BERT-style models.
*   **SentencePiece and Unigram tokenization:** Often used for multilingual models because they can work directly from raw text without relying on whitespace in the same way English-centric tokenizers do.

Subword tokenization allows the model to handle words it did not memorize as complete vocabulary entries. For example, a word like `unhappiness` might be represented as pieces such as `un` and `happiness`, depending on the tokenizer. A code identifier like `getCustomerInvoiceById` may split into many more pieces.

A rough rule of thumb in English is that **1 token is often about 0.75 words**, but this varies widely by language, punctuation, whitespace, code, tables, JSON, and formatting.

Tokenization has direct engineering consequences:

*   Billing is based on tokens, not words.
*   Long code snippets, JSON payloads, tables, and multilingual inputs can consume tokens much faster than plain English prose.
*   Context limits are enforced in tokens, so prompt length must be measured with the model's tokenizer.
*   A model can behave differently when the same visible string is tokenized differently by another model family.
*   Domain terms, product names, IDs, and mixed-language text may fragment into more tokens, increasing cost and latency.

Tokenizers also insert **special tokens** that are invisible to users but important to the model. These can mark boundaries between system prompts, user messages, assistant messages, tool calls, tool results, or end-of-sequence signals.

Tokenization also explains some famous model weaknesses. If the model sees text as token IDs rather than literal letters, it may struggle with tasks like counting characters inside a word, reversing exact strings, or reasoning about punctuation-heavy formats. Many "prompt problems" are actually token boundary problems.

For production work, the practical habit is simple: inspect token counts early. Do not estimate context size by words, pages, or characters when money, latency, or context limits matter.

## Decoder-Only Transformers

If autoregressive generation explains *how* text is emitted, the model architecture explains *why* generation has the performance profile it does.

The dominant architecture behind modern text-generation systems is the **Transformer**, introduced in the 2017 paper *Attention Is All You Need*. Transformers replaced older recurrent architectures by processing sequences in parallel during training, which made scaling practical.

There are three broad Transformer families:

**Encoder-only models** read the full input bidirectionally. They are excellent for representation tasks such as embeddings, classification, reranking, and retrieval.

**Encoder-decoder models** read the input with an encoder and generate output with a decoder. They are useful for sequence-to-sequence tasks such as translation, transcription, and summarization.

**Decoder-only models** read from left to right and are optimized for next-token prediction. This is the architecture most people mean when they talk about modern LLMs used for chat, coding, tool calling, and open-ended generation.

Decoder-only models use **causal masking**. During training and generation, each token can attend only to earlier tokens, not future tokens. This preserves the next-token prediction task and prevents the model from using information it would not have at inference time.

A decoder-only prompt lifecycle looks like this:

1. Raw text is tokenized into integer IDs.
2. Each token ID is converted into an embedding vector.
3. Positional information is added so the model can distinguish order.
4. Transformer blocks update each token representation using masked self-attention and feed-forward layers.
5. The final layer outputs logits for the next possible token.
6. A decoding strategy selects the next token.
7. The loop starts again.

<img width="818" height="218" alt="image" src="https://github.com/user-attachments/assets/9fec574e-cd9b-4a14-b4b2-2cb3ec444bd3" />

Modern production models are not identical to the original Transformer. Many use implementation improvements such as rotary positional embeddings, grouped-query attention, RMSNorm, SwiGLU feed-forward layers, mixture-of-experts routing, or specialized attention kernels. You do not need to memorize every variant in Chapter 1. What matters is the engineering implication: the architecture is designed to scale next-token prediction.

This design has two major advantages:

*   Training can process many token positions in parallel using known next tokens from the training data.
*   Inference uses a simple loop that can generate arbitrary-length text, code, JSON, tool arguments, or reasoning traces.

It also has tradeoffs:

*   Generation remains sequential at decode time.
*   Errors can compound because generated tokens become future context.
*   Long context creates memory and attention costs.
*   The model is not inherently grounded in truth, tools, or current data unless the application provides that grounding.

For a software engineer, architecture is not trivia. It determines which tasks a model is suited for, how it uses memory, and where latency and cost come from.

## Context Windows and Attention

If token embeddings represent *what* tokens are, **attention** represents how they relate to one another in context.

Consider the word `bank` in these two sentences:

*   `I sat by the river bank.`
*   `I deposited money in the bank.`

The token may look similar, but the meaning changes based on surrounding words. Through **self-attention**, the model updates the representation of each token by looking at other relevant tokens in the sequence.

Modern LLMs use **multi-head attention**, which means the model learns several different patterns of attention at once. One head may track syntax, another long-range references, another code structure, and another factual association. In practice, attention is not a neat explanation of reasoning, but it is the core mechanism that lets tokens interact.

Because text generation is autoregressive, decoder-only models apply causal masking. A token can only attend to earlier tokens, not future ones. That is what prevents the model from "cheating" during next-token prediction.

This leads to one of the most important concepts in production AI: the **context window**. The context window is the maximum number of tokens the model can consider at one time. If your system prompt, retrieved documents, chat history, tool results, and user message exceed that limit, something must be truncated, summarized, retrieved selectively, compressed, or dropped.

Long-context models are useful, but the advertised context window is not the same thing as effective context. A model may technically accept 128k, 1M, or more tokens, yet still miss important facts buried in the middle. This is often called the **lost-in-the-middle** problem. More context can also dilute attention, increase cost, and make debugging harder.

Attention also explains why long prompts are expensive. In standard attention, each token can interact with many other tokens in the sequence, so compute and memory costs rise sharply as context grows. Naive self-attention has quadratic scaling with sequence length. Production inference engines use optimized kernels, batching, paging, and KV caches to make this manageable, but the constraint does not disappear.

During generation, inference engines avoid recomputing the entire history from scratch by using a **KV cache**. The model stores intermediate key and value states for previous tokens in GPU memory and reuses them for each next-token step.

That cache is one of the main reasons long contexts are expensive in production. The longer the prompt, the longer the output, and the more concurrent users you serve, the larger the KV cache becomes. When a self-hosted inference server hits an out-of-memory error, the KV cache is often a major factor.

Context management is therefore an architecture problem, not just a prompt-writing problem. Common strategies include:

*   **RAG:** Retrieve only the most relevant external knowledge instead of stuffing every document into the prompt.
*   **Summarization:** Compress old conversation history or long documents into shorter state.
*   **Sliding windows:** Keep recent interaction history while dropping or summarizing older turns.
*   **Memory diffs:** Store only what changed or what matters instead of replaying every event.
*   **Structured context:** Separate instructions, facts, examples, tool results, and user input so the model can use them predictably.

The production skill is not asking, "How large is the model's maximum context window?" It is asking, "What context does this task actually need, and how do I keep it small, current, and verifiable?"

## Logits and Probabilities

After the prompt has passed through the Transformer network, the model produces **logits** for the next token. A logit is a raw numerical score for each possible token in the vocabulary.

Those scores are not probabilities yet. To convert them into probabilities, the system applies **softmax**, which turns the logits into a distribution whose values add up to 1.

For example, after the prompt `"The sky is"`, the model might internally assign probabilities like this:

*   `" blue"`: 0.80
*   `" gray"`: 0.12
*   `" falling"`: 0.03
*   other tokens: the remaining probability mass

The model then has to choose what to do with that distribution. If it always picks the highest-probability token, the output becomes more deterministic. If it samples from the distribution, the output becomes more varied.

Temperature changes this step by scaling logits before softmax. Lower temperature sharpens the distribution, making high-probability tokens dominate. Higher temperature flattens the distribution, giving lower-probability tokens more chance to appear.

This is where determinism ends. The model does not select a complete answer in one step. It repeatedly produces a probability distribution, samples or chooses one token, appends it, and continues. Hallucinations, repetition, creativity, and formatting drift all emerge from how the model's probability distribution is shaped and how the decoding step samples from it.

Engineers sometimes manipulate logits directly. For example, an API may support `logit_bias` to make certain tokens more or less likely, or a constrained decoder may restrict the next token to those allowed by a JSON schema or grammar. These techniques are powerful, but they should be used with care because token-level manipulation can have surprising side effects.

For engineers, the practical takeaway is simple: the model never "knows" the answer in a human sense. It continually assigns probabilities to token continuations and emits one token at a time.

## Decoding Parameters

Once you have logits and probabilities, you need a policy for selecting the next token. That policy is the **decoding strategy**, and API parameters let you control it.

The most important decoding controls are:

**Greedy decoding**  
Greedy decoding picks the highest-probability token at each step. It is deterministic, but it can be brittle, repetitive, or too narrow for open-ended tasks. Some APIs approximate greedy behavior with `temperature: 0`.

**Temperature**  
`temperature` adjusts how sharp or flat the probability distribution feels during sampling. Lower values make outputs more deterministic. Higher values increase variation. Extraction, classification, factual question answering, code generation, and tool invocation usually need low randomness. Brainstorming and creative writing can tolerate more.

**Top-p and top-k**  
`top_p`, also called nucleus sampling, limits sampling to the smallest set of tokens whose cumulative probability reaches the threshold `p`. This adapts to the shape of the distribution. `top_k` limits sampling to the `k` most likely tokens, which is simpler but less adaptive.

**Min-p and typical sampling**  
Some local inference stacks expose newer sampling options such as `min_p` or typical sampling. These can help filter very low-probability tokens while preserving diversity, but they are model- and task-dependent. Treat them as tuning tools, not universal fixes.

**Frequency and presence penalties**  
`frequency_penalty` reduces the chance of repeating tokens that have already appeared often in the generated text. `presence_penalty` reduces the chance of revisiting concepts that have appeared at all, even if they were not repeated heavily.

**Maximum token limits and stop sequences**  
`max_tokens` or `max_completion_tokens` acts as a hard ceiling on output length. This is a cost-control and latency-control tool. `stop` lets you define strings or markers where generation should halt.

**Seed and logit bias**  
`seed` can help make testing more reproducible in systems that support deterministic sampling. `logit_bias` lets you nudge the probability of specific tokens up or down, which can be useful for constrained behaviors, though it is a sharp tool and easy to misuse.

**Structured output and grammar constraints**  
`response_format`, JSON schema modes, grammar-constrained decoding, or similar controls help force the model into valid JSON or another strict structure. This is usually more reliable than hoping a prompt like "return valid JSON" will work forever.

**Streaming**  
`stream` returns partial tokens as they are generated, reducing perceived latency for end users. Streaming does not make the model generate faster, but it makes the application feel more responsive and lets clients show progress.

Good defaults depend on the task:

*   **Extraction, classification, routing, and tool calls:** Use low randomness, strict schemas, short outputs, and validation.
*   **Factual answers over retrieved documents:** Use low to moderate randomness, citations, missing-evidence behavior, and retrieval checks.
*   **Coding:** Use low randomness, tests, linters, and output validation.
*   **Creative writing and ideation:** Use higher randomness, larger output budgets, and looser structure.

Avoid "temperature hacking" as a debugging strategy. If the model is using the wrong facts, changing temperature will not fix retrieval. If the output violates a schema, use structured output and validation. If the system is too slow, reduce context or output length before blaming sampling.

## Generation Failure Modes

Text generation fails in predictable ways, and production systems need to anticipate them.

**Hallucination**  
The model produces plausible-sounding but unsupported content. This often happens when the prompt is ambiguous, the necessary facts are missing from context, or the model is pushed beyond what it actually knows.

**Inconsistency**  
The same prompt, or a slightly rephrased version of it, produces meaningfully different answers. This is especially dangerous in customer support, compliance, finance, healthcare, and workflow automation.

**Repetition and looping**  
Poor decoding settings, weak prompt structure, or model distribution collapse can cause the model to repeat phrases, restate itself, or get stuck in structural loops.

**Compounding errors**  
An early wrong token, wrong assumption, wrong tool call, or malformed structure becomes part of the context for the next token. In agents, this can turn one weak step into a chain of bad decisions.

**Truncation**  
The response stops early because it hit a token limit, a stop sequence, a context constraint, or a timeout. This can silently break JSON, code blocks, SQL, or multi-step instructions.

**Format drift**  
The model was asked for JSON, SQL, markdown, XML, YAML, or another strict format but gradually deviates from the contract. This is common when prompts are underspecified or the output is long.

**Lost context**  
Important information is technically present in the prompt but not used well by the model, especially in long contexts or when relevant facts are buried between unrelated chunks.

**Latency blowups**  
Long prompts, reasoning-heavy outputs, tool loops, retries, and large completion limits can make a system feel broken even when it is technically functioning correctly.

**Memory failures**  
Long context windows, long outputs, large batches, and high concurrency increase KV-cache pressure and can trigger GPU out-of-memory errors on self-hosted systems.

**Safety and security failures**  
Prompt injection, unsafe tool calls, data leakage, and policy-violating output are generation failures at the system level, even if the text looks fluent.

The engineering lesson is that generation quality is not just a model issue. It is a systems issue. Prompt design, retrieval quality, context management, decoding parameters, output validation, tool permissions, observability, and serving infrastructure all shape the final result.

Common mitigations include:

*   Ground factual answers with retrieval and citations.
*   Evaluate retrieval separately from generation.
*   Use schemas, parsers, and validators for structured outputs.
*   Add retries with targeted repair prompts for recoverable failures.
*   Keep deterministic work in code.
*   Limit tool permissions and require human review for high-impact actions.
*   Track prompts, retrieved chunks, outputs, token counts, latency, and errors.
*   Run regression tests before changing prompts, models, retrieval settings, or decoding parameters.

Mastering text generation mechanics is what separates prompt users from LLM engineers. The loop is simple, but every production concern in this book flows from it: tokens drive cost, attention drives context limits, logits drive sampling behavior, decode drives latency, and autoregressive errors drive the need for evaluation and guardrails.

## Chat Playgrounds

A **chat playground** is a web interface for testing LLMs before you write code. It usually lets you choose a model, enter messages, adjust generation settings, attach files or images if the model supports them, and inspect the output. Examples include [Google AI Studio](https://aistudio.google.com/), [OpenRouter Chat](https://openrouter.ai/chat), OpenAI Playground, Anthropic Console, Groq, Together.ai, and Hugging Face chat demos.

Chat playgrounds exist because they shorten the feedback loop. Instead of writing a full app just to compare two prompts or two temperatures, you can run the experiment in a browser, observe the behavior, and then copy the useful settings into code.

Use a playground to test:

*   how the same prompt behaves across models
*   how `temperature`, `top_p`, and output limits change the response
*   whether a model follows system instructions
*   whether a model can handle images, files, or long context
*   whether the provider exposes useful code export or API examples

Google AI Studio is useful when you want to experiment with Gemini models and multimodal inputs such as text, images, video, or code. OpenRouter Chat is useful when you want to compare many model families from one interface. Provider-specific playgrounds, such as OpenAI Playground or Anthropic Console, are useful when you plan to build directly on that provider's API.

A simple playground workflow:

1. Pick one model.
2. Paste a short prompt.
3. Run it once with low randomness.
4. Run it again with higher randomness.
5. Change only one setting at a time.
6. Record the model, settings, prompt, and output.
7. Move the best configuration into code.

Engineering consequence: playgrounds are excellent for exploration, but they are not production tests. The UI may add hidden defaults, model availability changes, and free tiers can have different limits from paid API usage. Always reproduce important playground findings through the API before shipping.

Common mistakes:

*   Comparing models with different prompts or settings.
*   Copying playground output into a product without API validation.
*   Ignoring pricing, rate limits, and data-retention settings.
*   Pasting secrets, customer data, or private documents into a playground.

## Hands-On Exercise

Use a chat playground to observe text generation behavior before writing code. You can use [Google AI Studio](https://aistudio.google.com/), [OpenRouter Chat](https://openrouter.ai/chat), OpenAI Playground, Anthropic Console, or another playground you have access to.

Requirements:

1. Choose one model in the playground.
2. Run this prompt:

```text
Explain why long prompts can make LLM applications slower and more expensive.
```

3. Run the same prompt three times with different decoding settings:
   * low randomness: `temperature=0` or as low as your provider supports
   * moderate randomness: `temperature=0.7`
   * high randomness: `temperature=1.2`
4. Keep the output limit small, such as 150 to 250 tokens.
5. Record the model name, temperature, output length, and any visible latency or token usage.
6. If the playground supports model comparison, run the same prompt against two different models with the same settings.
7. Write five short observations:
   * Which setting was most consistent?
   * Which setting was most verbose or surprising?
   * Which output was most useful for a technical user?
   * Did a higher temperature improve or weaken the answer?
   * Did two models behave differently under the same prompt?

Expected lesson: playgrounds are fast laboratories for understanding generation. Decoding parameters are product controls; they change cost, latency, repeatability, and failure risk even when the prompt stays the same.
