---
layout: article
lang: en
title: "How to Evaluate Reasoning Effort in Inkling and Other LLMs"
description: "Reasoning effort is a test-time control, not a quality score. Learn how to evaluate accuracy, token use, latency, and cost without confusing vendor benchmarks with your workload."
date: 2026-07-16 12:28:49 +0000
permalink: /writing/inkling-effort-evaluation/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# How to Evaluate Reasoning Effort in Inkling and Other LLMs

Some language models expose a control that trades inference budget for more test-time computation. Inkling describes this as controllable thinking effort. The setting can change how many tokens a model generates and how it performs on a task, but the control is not a universal quality score. You need to measure the whole operating point against the work your system must do.

This matters because a single benchmark result can hide the decision engineers actually face: how much quality is gained by spending more time, tokens, memory, or money on this workload?

## Effort is a curve, not a model ranking

An effort setting is meaningful only with its model version, runtime, prompt, sampling settings, tool configuration, and output budget. A value such as `0.7` from one model does not mean the same compute or reasoning depth as `0.7` from another. Even within one family, a new checkpoint or serving stack can alter the relationship between the setting and the outcome.

Think of an evaluation as a set of operating points. Each point pairs a task result with the resources consumed to obtain it:

```text
operating point = (quality, latency, generated tokens, tool calls, cost)
```

There may be no single best point. A support assistant, a code repair agent, and an offline research job can have different latency limits and different costs for an error. The useful question is which point meets each workload's constraints.

## What the public Inkling results do and do not show

The [original Inkling release](https://thinkingmachines.ai/news/introducing-inkling/) described a mixture-of-experts model with 975 billion total parameters, 41 billion active parameters, and a context window of up to one million tokens. The release also described an effort sweep from 0.2 to 0.99 on selected evaluations. Its plots compare Inkling's sweep with other models at default operating points, and the release notes that some results were externally reported or used an earlier checkpoint. These are vendor-reported results under a particular setup, not an independent comparison under your harness.

The model family has since changed. As of November 27, 2026, the [Inkling 2 Small release](https://thinkingmachines.ai/news/inkling-2-small/) describes a 594B-total, 17B-active MoE with image input and variable effort, and the [Inkling 2 Mini release](https://thinkingmachines.ai/news/inkling-2-mini/) describes a 26.4B-total model with a looped transformer design. The Mini release describes 3.4B active parameters and 5.4B weight activations per token to account for looping; its comparison table uses “activated” for 5.4B. Treat those labels as release-specific definitions rather than assuming they are directly comparable to a conventional active-parameter count. The release also reports an approximately 20 GB weight footprint, which is not a complete estimate of runtime memory.

These model specifications can help you identify what to test. They do not establish performance, latency, memory use, or cost in your environment. Vendor evaluation pages may be updated, so capture the model revision and page date when you use a reported result.

## Build an evaluation that answers an engineering question

### 1. Define the task and a failure

Use representative examples from the actual workflow, or a carefully reviewed evaluation set that reflects it. Include easy cases, difficult cases, ambiguous inputs, and cases where the correct response is to ask for more information or decline to act. Decide how each response will be graded before comparing settings.

For an agentic task, measure the final action as well as the text answer. A model that describes the right patch but fails to apply it, or that calls a tool with incorrect arguments, has not completed the task.

### 2. Sweep the effort control

Choose several values supported by the exact model and runtime. Begin with a small low-to-high sweep, then add values near the useful frontier. Hold other settings fixed: model revision, prompt, tool definitions, sampling parameters, context, and output limits.

Do not assume a monotonic relationship. Higher effort can use more tokens without improving the answer, and repeated runs can vary. Run multiple trials for tasks where output variance matters. If the provider does not expose deterministic seeds, record that limitation instead of treating repetitions as identical.

### 3. Measure the costs that affect your product

For each trial, record:

- Task success and the severity of any error.
- Input and generated tokens, if exposed.
- End-to-end latency and relevant stage timings.
- Tool calls, retries, and failed actions.
- Human correction or review effort.
- The actual serving price and infrastructure cost for that model version.

Use the cost formula that matches the deployment. An API's billed token cost is different from self-hosted GPU cost, and neither alone captures review time or failed actions. Report distributions such as median and tail latency where service-level objectives require them; do not infer p95 from a small sample.

### 4. Compare under the same harness

A model comparison is confounded if one system receives a different prompt, uses a stronger tool loop, gets more time, or has access to a different search index. If you are evaluating a provider's model rather than the whole product, keep the harness fixed. If you are evaluating complete systems, record the differences and frame the result as a system comparison.

For published benchmarks, read the task definitions, grader, model version, sampling procedure, and tool configuration. Check for contamination, duplicated or near-duplicated prompts, and scoring that rewards verbosity. A public benchmark is a useful reference point; your own representative task set is still needed for a deployment decision.

## A compact results table

Use one row for each model and effort setting. This blank template is deliberately not filled with fabricated results:

| Model revision | Effort | Success rate | Median / tail latency | Generated tokens | Tool calls | Human correction | Cost per task |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact revision | low | measure | measure | measure | measure | measure | calculate |
| exact revision | medium | measure | measure | measure | measure | measure | calculate |
| exact revision | high | measure | measure | measure | measure | measure | calculate |

Plot quality against latency or cost rather than sorting by a single score. A candidate that is worse on every measured dimension is dominated. Otherwise, select the operating point that satisfies the workload's quality and latency requirements at an acceptable cost.

## Check model capacity separately from effort

Test-time computation cannot replace every limitation of model scale or architecture. In particular, the Inkling 2 Mini release says looping improved selected long-context and serial-reasoning results, but it did not improve knowledge-heavy evaluations in the same way. Search or domain data may be needed when the limitation is missing knowledge rather than insufficient test-time compute.

Treat context length, model memory, and reasoning effort as separate deployment constraints:

- **Context length** is the maximum input and output context accepted by a particular checkpoint and serving endpoint.
- **Model memory** includes weights and the runtime memory needed for activations, caches, and framework overhead.
- **Effort** changes inference behavior; it may increase generation or computation, depending on the model.

A public model card's context limit does not guarantee that a hosted endpoint exposes the same limit. A quantized weight download that fits on a device does not guarantee that the full runtime workload will fit. Verify these limits with the exact checkpoint and serving path you plan to use.

## A practical decision rule

Use a reasoning-effort control when your evaluation shows a useful quality gain for the additional latency and cost, and when the task benefits from more test-time computation. Keep the control low or use a simpler model when additional effort does not change the outcome. For tasks that need external facts, improve retrieval or tool access before spending more reasoning tokens.

The result should be a measured operating policy, not a claim that one model or effort setting is universally best. Record the model revision, prompt, harness, evaluation set, date, and pricing assumptions so the comparison can be repeated when any of them changes.

For related topics, see [how to evaluate LLM reasoning claims](/fa/writing/ai-reasoning-evaluation/) and [an introduction to open-source AI](/writing/open-source-ai/).

## References

- [Thinking Machines: Inkling release and effort evaluation](https://thinkingmachines.ai/news/introducing-inkling/)
- [Thinking Machines: Inkling 2 Small](https://thinkingmachines.ai/news/inkling-2-small/)
- [Thinking Machines: Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/)
