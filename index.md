---
title: "Home"
nav_order: 0
---

# Mohammad Shojaei
{: .fs-9 }

AI Engineer building practical LLM systems, research agents, RAG pipelines, and fine-tuned models that survive outside notebooks.
{: .fs-6 .fw-300 }

I work on the messy middle between papers and production: retrieval quality, evaluation, tool use, inference cost, dataset design, and the boring reliability details that make AI products usable.
{: .fs-5 .fw-300 }

[Read my work](#writing--notes){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[Resume & Portfolio](about.html){: .btn .fs-5 .mb-4 .mb-md-0 }

[Email](mailto:shojaei.dev@gmail.com) · [LinkedIn](https://www.linkedin.com/in/mshojaei77) · [GitHub](https://github.com/mshojaei77) · [Hugging Face](https://huggingface.co/mshojaei77)

---

## What I build

I design and ship LLM-powered systems where quality, latency, cost, and maintainability all matter at the same time.

| Area | What I focus on |
|---|---|
| **RAG & Search** | Chunking, retrieval, reranking, citation-grounded answers, and evaluation loops. |
| **Agents & Workflows** | Tool use, LangGraph-style state machines, validation gates, human approval points, and recovery from failure. |
| **Fine-tuning & Datasets** | SFT datasets, instruction formatting, small-model adaptation, and domain-specific training pipelines. |
| **LLM Infrastructure** | Inference optimization, observability, cost control, deployment, and production debugging. |

---

## Current direction

Right now I am mostly exploring how to make **research agents** more reliable: not just better prompts, but better state, evidence tracking, validation, citations, and human checkpoints.

The question I keep coming back to:

> How do we turn deep research, RAG, and agent workflows into systems that are trustworthy enough for real users?

That means fewer demos, more measurements, and more attention to the boring failure modes: stale context, weak retrieval, silent tool failures, retry bugs, hallucinated citations, and expensive inference paths.

---

## Writing & notes

I write like a builder: experiments, breakdowns, bugs, implementation notes, and lessons from trying to make LLM systems behave in the real world.

Good starting points:

- **Production LLM systems** — RAG, agents, evals, observability, and deployment tradeoffs.
- **Fine-tuning small models** — dataset design, training recipes, and practical failure cases.
- **Research agents** — planning, evidence collection, source validation, and long-running workflows.
- **AI engineering notes** — the small technical details that usually get skipped in polished tutorials.

---

## Selected work

- [GitHub](https://github.com/mshojaei77) — open-source experiments, agents, RAG systems, and ML engineering projects.
- [Hugging Face](https://huggingface.co/mshojaei77) — datasets, models, and fine-tuning experiments.
- [Resume & Portfolio](about.html) — full background, experience, and technical profile.

---

## How I think about AI engineering

AI products do not fail only because the model is weak. They fail because the system around the model is vague: no evals, no recovery path, no source discipline, no observability, and no honest measurement.

My work is about closing that gap — turning research ideas into systems that can be tested, deployed, improved, and trusted.
