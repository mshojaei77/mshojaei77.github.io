# Chapter 2: Production Model Selection

LLM engineering is a logistics problem as much as it is a modeling problem. A few years ago, model selection was relatively simple: pick the strongest proprietary API you could afford and build around it. In production, that mindset creates unnecessary dependency and migration pain.

Models change quickly. Capabilities improve, context limits expand, modalities appear, pricing shifts, and providers deprecate old versions. A model that looks ideal this quarter may become too expensive, too slow, or unavailable later.

That means model selection is not a one-time decision. It is an ongoing evaluation process driven by two moving forces:

1. **Capabilities change:** Quality, context handling, tool use, reasoning behavior, and multimodal support keep improving.
2. **Economics change:** Input pricing, output pricing, caching policies, rate limits, and discount programs shift over time.

**The engineering takeaway:** model selection belongs inside the build-measure-learn loop. You choose a starting model, measure it against real tasks, and stay ready to replace it.

## Model Selection Criteria

A common beginner mistake is assuming an AI application is powered by one giant model that does everything. Production systems rarely work that way. Strong systems often combine several smaller components, each chosen for a specific job.

This reflects a simple architectural rule: **use the right model for the right task**.

| Role | Purpose | Typical Model Category | Example Requirement |
| :--- | :------ | :--------------------- | :------------------ |
| **Generator** | Produce the final user-facing answer. | Chat or instruct model | High-quality natural language output |
| **Embedder** | Convert text into vectors for retrieval. | Embedding model | Strong semantic search |
| **Reranker** | Reorder retrieved chunks by relevance. | Reranking model | Better retrieval precision |
| **Classifier / Router** | Route, moderate, or label requests. | Small fast classifier | Low latency and low cost |
| **Reasoning Model** | Handle multi-step judgment or complex analysis. | Deliberative reasoning model | Higher quality on difficult tasks |
| **Vision Model** | Read images, charts, or scanned documents. | Multimodal model | OCR and visual understanding |
| **Audio Model** | Convert speech to text or text to speech. | Speech model | Real-time or batch voice support |
| **Fallback Model** | Keep the system available during failures. | Alternate provider or cheaper tier | Operational resilience |

By splitting responsibilities across specialized models, you usually improve latency, cost, and reliability at the same time.

When evaluating a candidate model, ask these questions first:
*   Does it fit the task?
*   Is its output reliable enough for the product contract?
*   Is it fast enough for the user experience?
*   Is it affordable at expected scale?
*   Can it be replaced later without major rewrites?

## Closed and Open Models

Before choosing a specific model, decide what kind of access strategy you want. In practice, most teams choose among three approaches:

1. **Buy through a proprietary API:** You call a managed provider and let them handle model serving, scaling, safety layers, and billing.
2. **Rent open-weight models through hosted inference:** A third party hosts open-weight models behind an API, giving you more model variety without running GPUs yourself.
3. **Host models yourself:** You run the model on your own hardware or cloud GPUs using inference engines such as `vLLM`, `SGLang`, or `llama.cpp`.

Each strategy has clear tradeoffs:

| Strategy | Best when... | Avoid when... |
| :------- | :----------- | :------------ |
| **Buy** | You need fast delivery, strong support, and minimal infrastructure work. | You need full control over data flow or model weights. |
| **Rent** | You want access to open-weight models without managing serving infrastructure. | You require strict on-premise deployment or deep serving customization. |
| **Host** | You need privacy, offline use, custom serving, or deep infrastructure control. | You lack MLOps expertise, GPU budget, or operational tolerance. |

The important point is not which option is universally best. The important point is that access strategy affects architecture, compliance, and operations long before you compare raw model quality.

## API, Hosted, and Local Access

Once you know your access strategy, evaluate the operational features that come with it.

With managed APIs, you are not just buying a model. You are buying a platform. That platform can meaningfully change your backend design.

Important platform capabilities include:

*   **Structured outputs:** Can the provider enforce JSON schemas or typed responses, or do you need brittle string parsing?
*   **Prompt caching:** Can repeated prompt prefixes be cached to reduce latency and input cost?
*   **Batch processing:** Is there an asynchronous batch path for large offline jobs?
*   **Data controls:** Can you get zero-retention or enterprise data-handling guarantees?
*   **Regional availability:** Can you run in the geography required by your compliance constraints?
*   **Observability:** Do you get usage logs, traces, or request metadata for debugging and governance?

Hosted inference for open-weight models offers a middle ground. It avoids GPU management while still letting you experiment with a wider catalog of models.

Local hosting gives you the most control, but it also moves the burden onto your team. You now own model downloads, quantization choices, memory planning, autoscaling, failover, observability, and security.

From an engineering perspective, the real question is not "API or local?" The real question is: **which access mode gives us the control we need at an operational cost we can actually sustain?**

## Capability and Task Fit

Not all models are optimized for the same speed-quality tradeoff. One of the most important distinctions in production is between **fast models** and **reasoning models**.

Fast models are optimized for throughput, responsiveness, and low cost. They are usually the right default for repetitive tasks, high-volume workflows, and user-facing systems where latency matters.

Reasoning models spend more computation on difficult problems. They are useful when the task requires multi-step planning, ambiguity resolution, careful tradeoff analysis, or deeper tool-using behavior.

That does not mean reasoning models are always better. They are often slower and more expensive, and many routine tasks do not benefit from extra deliberation.

A practical task-fit table looks like this:

| Real-World Use Case | Fast Model? | Reasoning Model? |
| :------------------ | :---------- | :--------------- |
| Customer-support chatbot over FAQs or docs | Yes | Sometimes |
| Internal knowledge assistant over company documents | Yes | Sometimes |
| Invoice, form, or contract extraction pipeline | Yes | Sometimes |
| Email drafting or CRM reply assistant | Yes | Sometimes |
| Lead qualification and routing | Yes | No |
| Voice intake or appointment-booking agent | Yes | Sometimes |
| Proposal generation for complex RFPs | Sometimes | Yes + human review |
| Legal or compliance review assistant | Sometimes | Yes + human review |
| Risk analysis or policy interpretation workflow | Sometimes | Yes |
| Coding, debugging, or test-generation assistant | Sometimes | Yes |
| High-volume content transformation workflow | Yes | No |

The economic rule is straightforward: **use the cheapest model that still meets the task's quality bar**.

If a simple extraction job succeeds with a smaller fast model, using a frontier reasoning model is wasteful. If a compliance workflow needs careful multi-step judgment, using the cheapest fast model may create downstream failure costs that dwarf the token savings.

## Benchmarks and Product Tests

Public benchmarks are useful, but they are not deployment decisions. They are screening tools.

When you evaluate open-weight models, the first place to look is often the model card. A model card helps you perform both legal and technical due diligence.

A quick model-card checklist:
1. **License:** Can you use it commercially?
2. **Intended use:** Was it built for chat, code, embeddings, vision, or something else?
3. **Training notes:** Are there known data restrictions, domain gaps, or safety concerns?
4. **Maintenance status:** Is the model actively maintained?
5. **Usage constraints:** Is access gated or restricted?

File format also matters because model distribution is part of your supply chain:
*   **Safetensors** is preferred for safer tensor loading.
*   **GGUF** is widely used for efficient local inference, especially with `llama.cpp`.

It is also important to distinguish three terms that people often blur together:
*   **Open source:** the full training and release story is meaningfully open.
*   **Open weight:** the weights are available, but the training pipeline is not fully open.
*   **Open-washing:** the marketing says "open," but the license or restrictions make that claim misleading.

Benchmarks themselves should be treated carefully:
*   **Static benchmarks** are useful, but they can be contaminated if models saw similar tasks during training.
*   **Live benchmarks** reduce contamination by using fresher or delayed-release tasks.
*   **Human preference arenas** tell you what humans prefer in pairwise comparisons, but preference is not the same as factual correctness.
*   **LLM-as-judge workflows** scale well, but judges can be biased and should be calibrated with human review.

Leaderboards are most useful as shortlists. They help you narrow the field, not choose the winner blindly.

The decisive step is building a **golden dataset** for your own workload. Public benchmarks do not know your business rules, internal jargon, edge cases, or formatting contracts. Your private evaluation set should.

A good starter golden dataset often includes:
*   Real user tasks from logs or beta traffic.
*   Human-approved target outputs.
*   Explicit grading rules.
*   Adversarial cases such as prompt injection attempts.
*   Regression cases based on past failures.

Even 50 to 100 carefully chosen examples can catch major regressions when swapping models.

A practical private-evaluation workflow is:
1. Sample representative tasks.
2. Write or approve target answers.
3. Run the candidate model.
4. Score outputs with deterministic checks and targeted review.
5. Compare candidates using task success, not hype.

## Cost, Latency, and Reliability

LLM billing is granular. Providers may price input tokens, output tokens, cached tokens, batch jobs, and premium reasoning differently. That means model economics cannot be judged from a single headline number.

A useful mental model is:

```text
monthly_cost = requests * (
    input_cost
    + output_cost
    + reasoning_cost
    - caching_savings
    - batch_savings
    + retry_cost
)
```

The real metric is usually not cost per request. It is **cost per successful task**.

A cheap model that frequently fails, times out, hallucinates, or produces invalid structured output is not actually cheap if your system has to retry or escalate constantly.

Latency must also be measured as part of product quality. Users do not care that a model is slightly smarter if the system feels unresponsive. Likewise, a powerful model is operationally weak if it cannot meet your peak request rate or token throughput.

Reliability questions should be asked early:
*   What are the request-per-minute and token-per-minute limits?
*   How does the provider behave during spikes?
*   Can you add fallbacks across vendors?
*   What happens if one region or one model tier is unavailable?
*   Does your architecture degrade gracefully?

Privacy and compliance belong in this section too, because they directly affect provider choice:
*   Does the provider retain prompts or responses?
*   Can your data be used for training?
*   Can you choose a processing region?
*   Can provider staff access your prompts during support workflows?

In enterprise systems, cost, latency, reliability, and privacy are not separate concerns. They are one decision surface.

## Model Decision Records

Model choice should not live as tribal knowledge inside Slack messages. Treat it as an engineering decision that deserves a written record.

A simple model decision record can include this matrix:

| Criterion | Engineering Question |
| :-------- | :------------------- |
| **Task Fit** | What exact job must the model do? |
| **Quality** | What error rate is acceptable? |
| **Latency & Cost** | Is it fast enough, and what is the cost per successful task? |
| **Context** | Do we truly need long context, or should we build retrieval? |
| **Output Reliability** | Can it reliably produce the required format? |
| **Privacy / License** | Can the data leave our infrastructure, and is the model legally usable? |
| **Migration** | How easily can we swap providers or versions later? |

This record forces the team to document assumptions instead of hand-waving them.

You should also plan for migration from day one. Providers deprecate models. Behavior drifts. Safety tuning changes. Prices move. Rate limits tighten. A system that cannot swap models cleanly is fragile.

Practical migration defenses include:
*   Versioned model configuration.
*   A routing layer between your code and providers.
*   Golden-dataset regression tests.
*   Fallback models for outages or throttling.
*   Output validation at the application boundary.

The goal is not to predict the perfect model forever. The goal is to make model choice reversible.
