---
layout: article
lang: en
title: "Building Deep Research Agents: Architecture, RL Rewards, and Context Management"
description: "What the QUEST and S1-DeepResearch papers report about rubric-based evaluation, context management, and long-horizon research agents, with limits on the evidence."
date: 2026-06-30 05:44:31 +0000
permalink: /writing/deep-research-agents/
nav_exclude: true
article_kind: essay
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
source_url: "https://medium.com/@mshojaei77/building-deep-research-agents-architecture-rl-rewards-and-context-management-2ef1cf269dc3"
---

# Building Deep Research Agents: Architecture, RL Rewards, and Context Management

*Originally published on [Medium](https://medium.com/@mshojaei77/building-deep-research-agents-architecture-rl-rewards-and-context-management-2ef1cf269dc3) on June 30, 2026.*

How to Build Deep Research Agents: Lessons from QUEST and S1-DeepResearch

A practical look at why retrieval, context condensation, and structured rubrics matter more than raw model scale.

## The short version

Most “deep research” systems in production are not actually doing deep research. They are just running deep search.

There is a massive practical difference between the two. Deep search is extractive; it’s about finding a specific hidden needle in a massive haystack of web pages or PDFs. Deep research is constructive; it’s about synthesizing conflicting evidence, following multi-dimensional constraints, handling messy file inputs, and writing a coherent, structured, and cited deliverable.

A long-running research workflow has several operational risks: raw pages can consume context, repeated search and evaluation calls can raise cost, external services can throttle requests, and an agent can repeat work without a clear stopping condition. These are system-design risks to measure and handle, not inevitable outcomes.

Two recent papers offer concrete designs to study: **QUEST**, by researchers from The Ohio State University and Amazon AGI SF Lab, and **S1-DeepResearch**, whose authors are affiliated with XScience Lab and Wenge AI. They report strong results on selected benchmarks, but those results do not show that a 32B model generally matches proprietary research systems. Performance depends on the task, tools, evaluation setup, and model version.

The sections below distinguish the papers’ reported methods and results from engineering recommendations inferred from them.

![](https://cdn-images-1.medium.com/max/1024/1*DlWxkHH9sQhYBRupz89iYg.png)

## What the papers actually say

The reported numbers below are the papers’ results, not experiments reproduced for this article. Both are preprints. Treat benchmark comparisons as specific to their model versions, tool setups, and evaluation protocols.

Both projects aim to transition LLMs from search-centric QA engines to long-horizon research agents. They do this by focusing heavily on high-quality synthetic trajectory generation and targeted post-training rather than raw parameter scaling.

## QUEST: Rubric trees and structured context management

The QUEST paper introduces open-weight models from 2B to 35B and a training recipe using 8,000 synthesized tasks. The authors report that QUEST-35B matches or exceeds selected proprietary systems on some of their eight benchmarks; results vary by benchmark and evaluation setting. ([QUEST paper](https://arxiv.org/abs/2605.24218))

The QUEST recipe consists of:

1. **Rubric trees:** QUEST maps task requirements into hierarchical criteria. Objective tasks can use executable checks for verifiable leaves; open-ended tasks use rubric-based model judgments. Internal nodes aggregate criteria into a more detailed reward signal than a single pass/fail label. This is the paper’s reported design, not a guarantee that every criterion is objectively verifiable.
2. **Context Condenser:** When the context window fills up, a dedicated model compresses the raw history into a structured JSON state. This state explicitly lists *trusted* facts, *untrusted* claims, and *uncertain* gaps requiring follow-up.
3. **Three-Stage Training:** A pipeline consisting of mid-training (learning context summarization and web page info extraction), supervised fine-tuning (SFT) on trajectory sessions, and GRPO-style reinforcement learning (RL) optimized against the rubric-tree and fact-checking rewards.

## S1-DeepResearch: Graph-grounded task evolution

The S1-DeepResearch paper describes a 32B model initialized from Qwen3-32B, and its dataset card lists 15K training rows. The paper focuses on multi-dimensional research constraints and structured deliverables. The authors report gains over the base model on selected benchmarks, while noting remaining gaps on some difficult evaluations. ([paper](https://arxiv.org/abs/2606.15367); [dataset card](https://huggingface.co/datasets/ScienceOne-AI/S1-DeepResearch-15k))

Their three-stage synthesis framework consists of:

1. **Graph-Grounded Task Formulation:** Initializing seed entity pools from Wikipedia/Wikidata, expanding them into directed acyclic graphs (DAGs) using web search, and injecting complex constraints across nine dimensions (e.g., source boundaries, argumentation guidelines, and formatting limits).
2. **Topology-Aware Filtering:** They estimate task difficulty by analyzing the structural complexity of a task’s expected reasoning graph. They quantify this using metrics like Information Flow Complexity and Feedback Dependency Complexity (which identifies non-linear or cyclic reasoning dependencies) and keep the top 30% most complex tasks within each batch for trajectory generation.
3. **AgentLoop Sandbox Rollout:** They execute tasks inside a sandbox environment where the agent interacts with nine categories of atomic tools (including code execution, academic search, file parsing, and multimodal QA). Crucially, they apply scenario-specific post-processing to refine raw trajectories into polished, high-quality deliverables.

## The useful technical ideas

For an agentic pipeline, three design patterns in these papers are worth evaluating. The recommendations below are engineering interpretations; neither paper tests them as universal production rules.

## 1. Rubric-tree rewards for reinforcement learning

If you try to train an agent with RL on open-ended tasks (like writing a competitive analysis report), you will quickly run into a credit assignment wall. Pointwise scoring (asking an LLM to rate a report 1–10) suffers from severe score inflation, while pairwise comparison is notoriously unstable and expensive.

A rubric tree decomposes a task into explicit criteria. Some criteria can be checked mechanically; others, such as insight or readability, need a judge and can be wrong. For example, if you ask an agent to investigate a product recall:

- The root node aggregates the total score.
- The next layer decomposes the task into *Comprehensiveness*, *Instruction Following*, and *Insight*.
- The leaf nodes check highly specific, binary conditions: “Did the agent cite the primary recall notice URL?”, “Is the death toll reported as exactly 10?”, or “Is there a Markdown comparison table comparing Outbreak A and Outbreak B?”

![](https://cdn-images-1.medium.com/max/1024/1*Nbt1iXzpFLDOrybBWHhYpA.png)

In the QUEST implementation, objective leaves are checked by executable evaluators, while open-ended criteria use a judge model. The paper combines rubric rewards with a citation fact-checking reward for GRPO. This provides finer-grained feedback than one overall score, but judge quality and evaluation design remain sources of error.

## 2. Structured Context State vs. Naive Truncation

Most long-running agents break because their context window gets polluted. If you use a naive “keep-last-N-turns” or “discard-all” strategy, the model will forget verified facts, lose track of pending sub-goals, or endlessly repeat web searches it already completed.

The solution is to decouple the raw history from the agent’s working memory by using a structured JSON state:

```
{
  "search_queries": [
    { "q": "SME forensic accounting guidelines 2024", "intent": "locate warning indicators" }
  ],
  "visited_sources": [
    { "url": "https://example.com/report.pdf", "note": "Confirmed cash-flow metrics" }
  ],
  "information_state": {
    "trusted": [
      { "id": "T1", "claim": "SME cash-flow threshold is $50k", "sources": ["https://example.com/report.pdf"] }
    ],
    "untrusted": [
      { "id": "U1", "claim": "SME cash-flow is unlimited", "reason": "Contradicted by T1" }
    ],
    "uncertain": [
      { "id": "C1", "claim": "Intervention strategies are required", "need": "visit https://example.com/guide.html" }
    ]
  }
}
```

When context window usage crosses a threshold (for example, 16K tokens), a Context Condenser model reads the history and updates this JSON state. The raw history is wiped, and the agent resumes in a fresh context window initialized only with this compact state.

Uncertain entries explicitly tell the agent what to search next, while trusted entries let it write the final report without wasting tokens on redundant tool calls.

## 3. Knowing when to stop searching

The S1-DeepResearch team highlights a fascinating metric: in search-centric datasets, unsuccessful agentic trajectories often contain *more* tool calls than successful ones.

When an agent lacks native synthesis capabilities, it gets stuck in infinite validation loops. It will search, visit a page, find slightly ambiguous text, search again, revisit the same page, and eventually crash or hit the turn limit.

A research workflow needs a clear transition from evidence gathering to drafting. The S1-DeepResearch paper describes trajectories that include research tasks and structured outputs, but it does not compare incremental drafting with one-shot document generation. Treat that workflow choice as a hypothesis to evaluate on your own tasks.
## What I would actually use

If you want to build this without a massive GPU cluster, here is how to implement the most practical parts of these architectures.

## 1. Build a dual-cache system for your tools

If you let your agents hit live search and scraping APIs during development or RL rollouts, your API bills will be astronomical, and your pipeline will run incredibly slowly.

You need a strict two-tier caching layer:

- **Exact-Match Cache:** For any visit or parse\_file tool call, key the cache by the exact URL or file hash. Return the cached markdown or HTML digest instantly.
- **Semantic-Similarity Cache:** For web searches, run an exact-match lookup first. If it misses, encode the query string and run a similarity search (e.g., using a local FAISS index) against previously cached queries. If a query matches with high cosine similarity (e.g., > 0.92), serve the cached search results.

The QUEST paper describes exact-match and semantic caches for search and scholarly queries, plus an exact-URL cache for page visits. It does not report a “40% of tool calls” cache-hit figure, so no numerical savings claim is made here. In another system, cache value would need to be measured on that system’s own query and URL distribution.

## 2. Session-level trajectory formatting

Long trajectories can exceed the context or memory budget of a training setup. The specific limit depends on the model, sequence length, batching, and hardware; the paper’s session-level training design splits trajectories at context-condensation points.

One option is to segment training trajectories into sessions bounded by context-condensation steps:

```
[User Query + Prev State] -> [Agent Steps 1-5] -> [Next State]
```

Training on shorter sessions can reduce per-example sequence length and peak memory, but the paper does not establish that this makes training feasible on consumer GPUs. Measure throughput and quality with the target model, hardware, and sequence lengths.

## 3. Asynchronous evaluation pipelines

In an RL training pipeline, slow report evaluation can leave rollout or training workers waiting if the stages are tightly coupled. The QUEST paper describes asynchronous evaluation with Ray actors to overlap rubric-based scoring and training work. Whether this improves utilization depends on evaluator latency, queue depth, and how the trainer handles delayed rewards; the paper does not establish a universal evaluation-time range.

If synchronous evaluation leaves workers idle, one possible design is to separate rollout, scoring, and parameter updates:

1. Rollout workers generate trajectories and immediately offload them to an asynchronous task queue (like a Ray actor pool).
2. The trainer continues executing parameters updates on a sliding window of already-scored samples.
3. Slower, long-tail evaluations are processed out-of-order and naturally absorbed back into the training queue as they complete.

## The annoying details

These papers are excellent, but if you try to replicate their findings in production, several broken abstractions and edge cases will immediately cause problems.

## 1. Tool-augmented multimodal reasoning is a hack

The S1-DeepResearch paper starts from the text-only Qwen3-32B model and adds image- and video-question-answering tools to its sandbox. This is tool-mediated multimodal handling, not native multimodal perception by the base model. The paper reports multimodal evaluation separately from its textual results.

This is a tool-mediated setup. The agent has to look at a user query, realize there is an image attachment, call an image-QA tool with a textual question, receive a textual description, and then integrate that description into its planning loop.

This multi-step translation layers latency and adds points of failure. If the first tool call generates a generic or slightly misaligned description of the image, the entire reasoning path is permanently derailed. For true multimodal research tasks (like visual geo-localization or parsing complex diagrams in scientific PDFs), you will eventually need a native vision-language model as your agent core rather than relying on textual tool-use wrappers.

## 2. The RL “Alignment Tax” on hard reasoning

In the QUEST ablation, the authors report that adding RL improves open-ended tasks while slightly reducing scores on HLE and GAIA under their evaluation setup.

The authors hypothesize that optimizing for research-report behavior can trade off against some general reasoning behavior. This is a result for one training recipe and those benchmarks, not a general rule that RL weakens reasoning. For a different task, measure the trade-off on that task’s own evaluation set before choosing a post-training objective.

## Limitations

Before you commit to these architectures, keep these limitations in mind:

- **Coding tasks are ignored:** Neither QUEST nor S1-DeepResearch focuses on software engineering. Coding agents require continuous execution-based debugging, repository-scale context parsing, and deep compiler-error feedback loops. The training datasets and evaluation rubrics presented here will not transfer well to building coding assistants.
- **Deep dependence on a strong judge model:** Generating synthetic trajectories and evaluating reports relies heavily on frontier models (like Claude Sonnet or GPT-5) acting as judges. If your evaluator model has formatting biases or fails to detect subtle factual hallucinations, your synthetic dataset will quickly degrade in quality.
- **Messy real-world data:** The graph-grounded evolution pipelines assume that your source material is clean and crawlable. In enterprise settings, you will be dealing with poorly OCR’ed PDFs, broken tables, and missing metadata. None of these benchmarks fully reflect the structural pain of parsing real-world enterprise documents.

## Final takeaway

Taken together, these papers suggest that model size is only one factor in research-agent performance. Their results point to task construction, tool interaction, context management, and evaluation as important parts of the system. They do not show that model scale or context length no longer matter, or that one recipe is best for every research task.

For an engineering team, the transferable lesson is to treat research agents as complete systems: define task requirements, track evidence and uncertainty, evaluate the final deliverable, and measure where the workflow fails before investing in training.

## References

- [Paper: QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks](https://arxiv.org/abs/2605.24218)
- [Paper: S1-DeepResearch: Beyond Search, Toward Real-World Long-Horizon Research Agents](https://arxiv.org/abs/2606.15367)
- [Dataset: S1-DeepResearch-15k](https://huggingface.co/datasets/ScienceOne-AI/S1-DeepResearch-15k)
- [Repository: QUEST Project Page](https://osu-nlp-group.github.io/QUEST)
