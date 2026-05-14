**Chapter 2: Production Model Selection**

LLM engineering is a logistics problem as much as it is a modeling problem. A few years ago, model selection was relatively simple: pick the strongest proprietary API you could afford and build around it.

That is no longer a mature production strategy.

Models change quickly. Capabilities improve, context limits expand, modalities appear, pricing shifts, rate limits move, and providers deprecate old versions. A model that looks ideal this quarter may become too expensive, too slow, too restricted, or unavailable later.

Model selection is therefore not a one-time decision. It is an ongoing evaluation process driven by two moving forces:

1. **Capabilities change:** Quality, context handling, tool use, reasoning behavior, coding skill, multilingual support, and multimodal support keep improving.
2. **Economics change:** Input pricing, output pricing, caching policies, batch discounts, GPU costs, rate limits, and enterprise agreements shift over time.

The engineering takeaway is simple: model selection belongs inside the build-measure-learn loop. You choose a starting model, measure it against real tasks, and keep the architecture ready for replacement.

There is no universal best model. There is only the best model for a task, under constraints, at a point in time.

## Model Selection Criteria

A common beginner mistake is assuming an AI application is powered by one giant model that does everything. Production systems rarely work that way. Strong systems often combine several specialized components, each chosen for a specific job.

This reflects a simple architectural rule: **use the right model for the right task**.

| Role | Purpose | Typical Model Category | Example Requirement |
| :--- | :------ | :--------------------- | :------------------ |
| **Generator** | Produce the final user-facing answer. | Chat or instruct model | High-quality natural language output |
| **Embedder** | Convert text into vectors for retrieval. | Embedding model | Strong semantic search |
| **Reranker** | Reorder retrieved chunks by relevance. | Reranking model | Better retrieval precision |
| **Classifier / Router** | Route, moderate, or label requests. | Small fast classifier | Low latency and low cost |
| **Reasoning Model** | Handle multi-step judgment or complex analysis. | Deliberative reasoning model | Higher quality on difficult tasks |
| **Code Model** | Generate, edit, or explain code. | Code-specialized model | Strong repository and tool-use behavior |
| **Vision Model** | Read images, charts, screenshots, or scanned documents. | Multimodal model | OCR and visual understanding |
| **Audio Model** | Convert speech to text or text to speech. | Speech model | Real-time or batch voice support |
| **Fallback Model** | Keep the system available during failures. | Alternate provider or cheaper tier | Operational resilience |

By splitting responsibilities across specialized models, you usually improve latency, cost, and reliability at the same time.

Model selection should start with the use case, not the leaderboard. A practical first-pass checklist is:

*   **Task fit:** What exact job must the model do: answer, extract, classify, route, summarize, code, reason, translate, speak, see, or call tools?
*   **Quality bar:** What error rate is acceptable, and which errors are unacceptable?
*   **Latency target:** What time to first token and total response time will users tolerate?
*   **Cost target:** What is the acceptable cost per successful task, not just cost per request?
*   **Context needs:** Does the task need long context, retrieval, memory, or only a short prompt?
*   **Output contract:** Does the model need to produce natural language, JSON, SQL, tool arguments, citations, or a strict schema?
*   **Privacy and compliance:** Can data leave your infrastructure? Is regional processing required? Are prompts retained?
*   **Licensing and control:** Can you legally use the model commercially, fine-tune it, or redistribute derivatives?
*   **Operational burden:** Can your team operate the access mode you choose?
*   **Migration path:** Can you swap models later without rewriting the product?

This is where **small language models (SLMs)** matter. A frontier LLM may be best for broad reasoning, difficult synthesis, and ambiguous workflows. An SLM may be better for classification, extraction, routing, moderation, rewrite tasks, structured transformation, private deployment, or low-latency high-volume work.

The question is not "Which model is smartest?" The question is "Which model meets the requirement with the least cost, latency, risk, and operational complexity?"

## Closed and Open Models

Before choosing a specific model, decide what kind of access and control strategy you want. The usual debate is framed as closed models versus open models, but in production you should be more precise.

**Closed proprietary models** are served by a provider and usually expose only an API. You do not get the weights or training pipeline. You get capability, convenience, support, and managed infrastructure.

**Open-weight models** provide downloadable weights, but not necessarily full training data, training code, or unrestricted licensing. Many "open" LLMs are actually open-weight models.

**Open-source models** are more transparent across weights, code, data process, and licensing, though true openness varies widely.

Each category has advantages:

| Model Type | Strengths | Tradeoffs |
| :--------- | :-------- | :-------- |
| **Closed proprietary** | Strong frontier capability, fast integration, managed scaling, polished APIs, enterprise support. | Vendor lock-in, less control, data-handling concerns, pricing changes, limited customization. |
| **Open-weight** | More control, self-hosting options, fine-tuning, privacy, lower marginal cost at scale. | You own more infrastructure and evaluation work; licenses vary; raw quality may lag frontier models for some tasks. |
| **Open-source** | Best transparency and long-term control when genuinely open. | Fewer models meet this bar; still requires serving, security, and maintenance expertise. |

There is no permanent winner. Closed models often win for speed-to-value, broad capability, and low operational burden. Open-weight models often win for customization, data sovereignty, predictable control, cost at high volume, offline use, or strategic independence.

The production pattern increasingly looks hybrid:

*   Use a strong closed model for ambiguous reasoning, difficult writing, or early product discovery.
*   Use open-weight or smaller models for high-volume extraction, routing, classification, and private workloads.
*   Route requests dynamically based on difficulty, data sensitivity, latency, and cost.
*   Keep fallbacks across providers or model families for outages and regressions.

Scaffolding can narrow capability gaps. A well-designed RAG pipeline, tool workflow, schema validator, or agent state machine can make a smaller or open-weight model perform better than a stronger model used carelessly. Model quality matters, but architecture matters too.

## API, Hosted, and Local Access

Once you know your control strategy, evaluate the access mode. In practice, most teams choose among three approaches:

1. **Managed API:** You call a proprietary or managed model API and let the provider handle serving, scaling, safety layers, uptime, and billing.
2. **Hosted inference:** A third party serves open-weight models behind an API, giving you model variety without running GPUs yourself.
3. **Self-hosted or local inference:** You run the model on your own laptop, server, on-prem cluster, or cloud GPUs using inference engines such as `vLLM`, `SGLang`, `llama.cpp`, or local tools such as Ollama.

Each strategy has clear tradeoffs:

| Strategy | Best When | Avoid When |
| :------- | :-------- | :--------- |
| **Managed API** | You need fast delivery, elastic scaling, strong models, and minimal infrastructure work. | You need full control over data flow, weights, serving behavior, or long-term cost structure. |
| **Hosted inference** | You want access to open-weight models without managing GPU serving. | You require strict on-prem deployment, deep serving customization, or guaranteed provider independence. |
| **Self-hosted / local** | You need privacy, offline use, customization, predictable high-volume economics, or deep infrastructure control. | You lack GPU expertise, MLOps capacity, observability, security support, or operational tolerance. |

With managed APIs, you are not just buying a model. You are buying a platform. That platform can meaningfully change your backend design.

Important platform capabilities include:

*   **Structured outputs:** Can the provider enforce JSON schemas or typed responses, or do you need brittle string parsing?
*   **Tool calling:** Does the API support tool schemas, parallel tool calls, or strict argument validation?
*   **Prompt caching:** Can repeated prompt prefixes be cached to reduce latency and input cost?
*   **Batch processing:** Is there an asynchronous batch path for large offline jobs?
*   **Data controls:** Can you get zero-retention or enterprise data-handling guarantees?
*   **Regional availability:** Can you run in the geography required by your compliance constraints?
*   **Observability:** Do you get usage logs, traces, request IDs, or request metadata for debugging and governance?
*   **Rate limits and quotas:** Can the provider support your peak traffic and burst patterns?

Hosted inference for open-weight models offers a middle ground. It avoids GPU management while still letting you experiment with a wider catalog of models. It is useful for exploration, A/B testing, and workloads that need open-weight flexibility without self-hosting overhead.

Self-hosting gives you the most control, but it also moves the burden onto your team. You now own model downloads, quantization choices, GPU memory planning, autoscaling, batching, failover, security patching, observability, inference upgrades, and capacity planning.

Volume matters. Low or spiky traffic often favors managed APIs because you pay for usage and avoid idle infrastructure. High, steady traffic can favor self-hosting if your team can keep GPU utilization high and operate the system safely.

Sensitivity matters too. If prompts include regulated data, confidential source code, internal strategy, medical records, financial records, or customer PII, provider terms and deployment location may decide the access mode before model quality enters the conversation.

From an engineering perspective, the real question is not "API or local?" The real question is: **which access mode gives us the control we need at an operational cost we can actually sustain?**

## Capability and Task Fit

Not all models are optimized for the same speed-quality tradeoff. One of the most important distinctions in production is between **fast models**, **small specialized models**, and **reasoning models**.

Fast models are optimized for throughput, responsiveness, and low cost. They are usually the right default for repetitive tasks, high-volume workflows, and user-facing systems where latency matters.

Small specialized models can be excellent when the task is narrow and well-defined. A fine-tuned or carefully prompted SLM may outperform a larger general model on a specific extraction, classification, routing, or formatting task because the requirements are stable and the output space is constrained.

Reasoning models spend more computation on difficult problems. They are useful when the task requires multi-step planning, ambiguity resolution, careful tradeoff analysis, mathematical reasoning, code repair, or deeper tool-using behavior.

That does not mean reasoning models are always better. They are often slower and more expensive, and many routine tasks do not benefit from extra deliberation.

A practical task-fit table looks like this:

| Real-World Use Case | Fast / Small Model? | Reasoning Model? | Notes |
| :------------------ | :------------------ | :--------------- | :---- |
| Customer-support chatbot over FAQs or docs | Yes | Sometimes | Retrieval quality often matters more than raw model size. |
| Internal knowledge assistant over company documents | Yes | Sometimes | Long context and citations may matter. |
| Invoice, form, or contract extraction pipeline | Yes | Sometimes | Schema reliability and validation are critical. |
| Email drafting or CRM reply assistant | Yes | Sometimes | Tone and personalization may drive model choice. |
| Lead qualification and routing | Yes | No | Usually a classifier or small model task. |
| Voice intake or appointment-booking agent | Yes | Sometimes | Latency dominates the user experience. |
| Proposal generation for complex RFPs | Sometimes | Yes + human review | Requires synthesis and domain judgment. |
| Legal or compliance review assistant | Sometimes | Yes + human review | Risk and auditability dominate. |
| Risk analysis or policy interpretation workflow | Sometimes | Yes | Must measure consistency and escalation quality. |
| Coding, debugging, or test-generation assistant | Sometimes | Yes | Repo context and tool behavior matter. |
| High-volume content transformation workflow | Yes | No | Small model, batch mode, or caching may be best. |

The economic rule is straightforward: **use the cheapest model that still meets the task's quality bar**.

If a simple extraction job succeeds with a smaller fast model, using a frontier reasoning model is wasteful. If a compliance workflow needs careful multi-step judgment, using the cheapest fast model may create downstream failure costs that dwarf the token savings.

For complex products, model routing is often better than a single-model bet. A router can send simple requests to a cheap model, sensitive requests to a local model, code requests to a code model, and complex reasoning requests to a stronger model. The router itself can be deterministic rules, a classifier, or a small model.

Break workflows into steps before selecting models. "Build an AI assistant" is too vague. A real assistant may contain separate tasks: intent detection, retrieval, reranking, answer generation, citation checking, tool selection, tool argument generation, safety classification, summarization, and escalation. Each step may deserve a different model.

## Benchmarks and Product Tests

Public benchmarks are useful, but they are not deployment decisions. They are screening tools.

Leaderboards and arenas can help you shortlist candidates. Examples include human-preference arenas, open-model leaderboards, coding benchmarks, reasoning benchmarks, latency and price comparison sites, and provider-published evals. They are useful signals, but they do not know your product.

Benchmarks should be treated carefully:

*   **Static benchmarks** are useful, but they can be contaminated if models saw similar tasks during training.
*   **Live benchmarks** reduce contamination by using fresher or delayed-release tasks.
*   **Human preference arenas** capture what people like in side-by-side comparisons, but preference is not the same as factual correctness or workflow reliability.
*   **Automated leaderboards** depend heavily on harness configuration, prompt format, scoring method, and benchmark choice.
*   **LLM-as-judge workflows** scale well, but judges can be biased and should be calibrated with human review.
*   **Provider claims** should be verified on your own prompts, data, latency target, and output contracts.

When you evaluate open-weight models, the first place to look is often the model card. A model card helps you perform both legal and technical due diligence.

A quick model-card checklist:

1. **License:** Can you use it commercially? Can you fine-tune it? Are there field-of-use restrictions?
2. **Intended use:** Was it built for chat, code, embeddings, vision, tool calling, or something else?
3. **Training notes:** Are there known data restrictions, domain gaps, safety concerns, or evaluation caveats?
4. **Maintenance status:** Is the model actively maintained?
5. **Usage constraints:** Is access gated, restricted, or subject to an acceptable-use policy?
6. **Serving requirements:** What precision, context length, VRAM, and inference engine are expected?

File format also matters because model distribution is part of your supply chain:

*   **Safetensors** is preferred for safer tensor loading.
*   **GGUF** is widely used for efficient local inference, especially with `llama.cpp`.

It is also important to distinguish three terms that people often blur together:

*   **Open source:** The full training and release story is meaningfully open.
*   **Open weight:** The weights are available, but the training pipeline is not fully open.
*   **Open-washing:** The marketing says "open," but the license or restrictions make that claim misleading.

Leaderboards are most useful as shortlists. They help you narrow the field, not choose the winner blindly.

The decisive step is building a **golden dataset** for your own workload. Public benchmarks do not know your business rules, internal jargon, user behavior, edge cases, data quality, latency needs, safety policy, or formatting contracts. Your private evaluation set should.

A good starter golden dataset often includes:

*   Real user tasks from logs, beta traffic, support tickets, or subject-matter experts.
*   Human-approved target outputs or grading rubrics.
*   Deterministic checks for schemas, citations, tool arguments, and policy constraints.
*   Adversarial cases such as prompt injection attempts and malformed inputs.
*   Edge cases based on domain terminology, rare documents, and ambiguous requests.
*   Regression cases based on past failures.

Even 50 to 100 carefully chosen examples can catch major regressions when swapping models.

A practical private-evaluation workflow is:

1. Sample representative tasks.
2. Define success criteria and unacceptable failures.
3. Write or approve target answers, grading rubrics, or deterministic checks.
4. Run candidate models with the same prompts, retrieval settings, and decoding parameters.
5. Score outputs by task success, factuality, format validity, latency, cost, and escalation behavior.
6. Review failures manually and update the golden dataset with important misses.
7. Repeat before changing models, prompts, retrieval, or decoding settings.

Early exploration can include quick "vibe checks" across several models, especially when product requirements are still forming. But vibe checks are not launch criteria. Launch criteria require repeatable tests on your data.

## LLM Leaderboards

An **LLM leaderboard** is a website that compares models using benchmark results and operational measurements. A good leaderboard helps you shortlist models before you spend time integrating them.

[Artificial Analysis](https://artificialanalysis.ai/) is a production-oriented example. It compares models across practical dimensions such as quality, output speed, latency, pricing, context window, and provider availability. Other useful references include human-preference arenas such as LMSYS Chatbot Arena, open-model leaderboards such as Hugging Face's Open LLM Leaderboard, and task-specific benchmarks such as coding or agentic leaderboards.

Leaderboards exist because model marketing is noisy. A model can be strong on reasoning but slow, cheap but weak, fast but expensive through one provider, or impressive in chat but poor at your structured extraction task.

When reading a leaderboard, look for:

*   **Quality or intelligence score:** a broad signal of model capability.
*   **Output speed:** how many tokens per second the model produces.
*   **Time to first token:** how quickly the first streamed token appears.
*   **Price:** input and output cost, usually per million tokens.
*   **Context window:** how many tokens the model can consider at once.
*   **Provider:** who serves the model and under what access mode.
*   **Specialized scores:** coding, agentic, multimodal, voice, or other domain-specific views.

A practical leaderboard workflow:

1. Start with your task: chat, coding, extraction, RAG answer generation, summarization, or tool calling.
2. Use a production-oriented leaderboard such as Artificial Analysis to shortlist models with acceptable quality, speed, context, and price.
3. Cross-check a human-preference arena if the task is conversational or writing-heavy.
4. Cross-check an open-model leaderboard if you are considering open-weight models.
5. Remove models that fail your privacy, licensing, provider, or operational constraints.
6. Test the remaining candidates on your own prompts and golden dataset.

Engineering consequence: leaderboards are directional signals, not deployment decisions. They help you avoid bad starting points, but they do not know your users, documents, schemas, latency target, or budget.

Common mistakes:

*   Choosing the top-ranked model without checking cost or latency.
*   Trusting one leaderboard as the whole truth.
*   Comparing benchmark scores from different dates or harnesses as if they were identical.
*   Ignoring provider-specific performance for the same open model.
*   Skipping your own product tests after leaderboard shortlisting.

## Cost, Latency, and Reliability

LLM billing is granular. Providers may price input tokens, output tokens, cached tokens, reasoning tokens, image inputs, audio, batch jobs, and premium model tiers differently. Self-hosted systems replace token billing with GPU rental, hardware purchase, utilization, engineering time, monitoring, and operational risk.

A useful mental model is:

```text
monthly_cost = requests * (
    input_cost
    + output_cost
    + reasoning_cost
    + tool_loop_cost
    - caching_savings
    - batch_savings
    + retry_cost
    + human_escalation_cost
)
```

The real metric is usually not cost per request. It is **cost per successful task**.

A cheap model that frequently fails, times out, hallucinates, or produces invalid structured output is not actually cheap if your system has to retry, escalate, or lose user trust. A more expensive model can be cheaper overall if it reduces retries and manual review. The reverse is also true: a frontier model can be wasteful when a small model or deterministic rule would do the job.

Latency must also be measured as part of product quality. Users do not care that a model is slightly smarter if the system feels unresponsive. Measure:

*   **Time to first token (TTFT):** How long before the user sees the first streamed output?
*   **Tokens per second:** How quickly does the model produce the rest of the answer?
*   **Total task latency:** How long until the user receives a usable result, including retrieval, tools, validation, and retries?
*   **Tail latency:** How bad are the slowest 5% or 1% of requests?
*   **Throughput:** Can the system handle peak request and token volume?

Cost and latency are intertwined with model size, context length, output length, quantization, batching, caching, provider rate limits, network distance, and tool-call design.

Common optimization levers include:

*   Route simple tasks to smaller models.
*   Cache repeated prompt prefixes and retrieval results.
*   Reduce unnecessary context.
*   Use batch mode for offline workloads.
*   Stream responses for interactive UX.
*   Use speculative decoding or optimized serving engines when self-hosting.
*   Validate outputs early so failures do not continue through expensive workflows.

Reliability questions should be asked early:

*   What are the request-per-minute and token-per-minute limits?
*   How does the provider behave during spikes?
*   Can you add fallbacks across vendors or access modes?
*   What happens if one region or one model tier is unavailable?
*   Does your architecture degrade gracefully?
*   How often does the model produce invalid structured output?
*   How consistent is the model under rephrased but equivalent prompts?
*   How will you detect quality drift after a provider updates a model?

Privacy and compliance belong in this section too, because they directly affect provider choice:

*   Does the provider retain prompts or responses?
*   Can your data be used for training?
*   Can you choose a processing region?
*   Can provider staff access your prompts during support workflows?
*   Do licenses permit your commercial use, fine-tuning, or deployment model?
*   Are audit logs and data-deletion guarantees available?

In enterprise systems, cost, latency, reliability, privacy, and licensing are not separate concerns. They are one decision surface.

## Model Decision Records

Model choice should not live as tribal knowledge inside Slack messages. Treat it as an engineering decision that deserves a written record.

A **model decision record (MDR)** is the model-selection equivalent of an architecture decision record. It captures what you chose, why you chose it, what you tested, what risks remain, and when the decision should be revisited.

A simple model decision record can include this matrix:

| Criterion | Engineering Question |
| :-------- | :------------------- |
| **Use Case** | What exact workflow, user, and business outcome does this model support? |
| **Task Fit** | What exact job must the model do? |
| **Candidates** | Which models and access modes were compared? |
| **Quality** | What error rate is acceptable, and how was quality measured? |
| **Latency & Cost** | Is it fast enough, and what is the cost per successful task? |
| **Context** | Do we truly need long context, or should we build retrieval? |
| **Output Reliability** | Can it reliably produce the required format? |
| **Privacy / License** | Can the data leave our infrastructure, and is the model legally usable? |
| **Operational Burden** | Who owns monitoring, rate limits, fallbacks, and incident response? |
| **Migration** | How easily can we swap providers, models, or versions later? |
| **Review Trigger** | What event forces reevaluation: price change, model deprecation, quality drift, volume growth, or new compliance requirement? |

This record forces the team to document assumptions instead of hand-waving them.

A minimal MDR can look like this:

```text
Decision:
  Use Model X through Provider Y for customer-support answer generation.

Context:
  The system answers questions over indexed help-center documents.
  The target latency is under 3 seconds to first useful output.
  Customer PII may appear in prompts, so enterprise data controls are required.

Candidates:
  Model A through managed API.
  Model B through hosted inference.
  Model C self-hosted with vLLM.

Evaluation:
  120 golden examples from real support tickets.
  Metrics: answer correctness, citation correctness, JSON validity, TTFT,
  total latency, cost per successful answer, and escalation rate.

Decision Rationale:
  Model A had the best citation correctness and lowest retry rate.
  Model C was cheaper at projected volume but required GPU operations
  the team cannot support yet.

Risks:
  Vendor pricing may change.
  Long-context requests are expensive.
  A fallback model is required for provider incidents.

Review:
  Reevaluate after 90 days, after a major price change, or when traffic
  exceeds 1 million requests per month.
```

You should also plan for migration from day one. Providers deprecate models. Behavior drifts. Safety tuning changes. Prices move. Rate limits tighten. Open-weight model licenses change. Better small models appear. A system that cannot swap models cleanly is fragile.

Practical migration defenses include:

*   Versioned model configuration.
*   A routing layer between your code and providers.
*   Provider adapters with common request and response contracts.
*   Golden-dataset regression tests.
*   Fallback models for outages or throttling.
*   Output validation at the application boundary.
*   Monitoring for quality, latency, cost, and invalid-output drift.

The goal is not to predict the perfect model forever. The goal is to make model choice measurable, documented, and reversible.

## Hands-On Exercise

Use leaderboard pages to create a simple model decision record for a support assistant. You do not need to run a full benchmark yet. The goal is to practice turning public model data into a shortlist, then documenting why the shortlist is not enough.

Scenario:

```text
You are building a customer-support assistant for a small SaaS product.
It answers questions from help-center text, classifies support messages,
and drafts short replies for human agents to review.
The team has a limited budget and no GPU operations experience.
Some prompts may contain customer email addresses or account IDs.
```

Requirements:

1. Open a production-oriented leaderboard such as [artificialanalysis.ai](https://artificialanalysis.ai/).
2. Choose three candidate models or access modes:
   * one managed API model
   * one cheaper or smaller model
   * one open-weight or hosted open-weight option
3. For each candidate, record visible leaderboard information:
   * quality or intelligence score
   * output speed or latency
   * input and output price
   * context window
   * provider or access mode
4. Cross-check at least one other source, such as a human-preference arena, an open-model leaderboard, or the model's official model card.
5. Build a comparison table with these columns:
   * task fit
   * expected quality
   * latency risk
   * cost risk
   * privacy risk
   * operational burden
   * migration difficulty
6. Pick a default model and one fallback model.
7. Write a short model decision record that explains:
   * why you chose the default
   * what the leaderboard helped you understand
   * what the leaderboard could not prove
   * what you would measure before launch
   * when you would revisit the decision

Expected lesson: leaderboards are useful for shortlisting, especially when they include cost and latency. They do not replace your own evals on real support questions, private documents, and product constraints.
