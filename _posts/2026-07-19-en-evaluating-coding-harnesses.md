---
layout: article
lang: en
title: "How to Evaluate an AI Coding Harness: Start with the Workflow"
description: "A practical way to compare coding agents by their execution environment, tool loop, permissions, recovery, and results on representative engineering tasks."
date: 2026-07-19 07:52:19 +0000
permalink: /writing/evaluating-coding-harnesses/
nav_exclude: true
article_kind: guide
last_modified_at: 2026-11-27 00:00:00 +0000
author: "Mohammad Shojaei"
show_article_header: false
image: /assets/img/social-card.png
feed:
  excerpt_only: true
---

# How to Evaluate an AI Coding Harness: Start with the Workflow

A coding model is only one part of a coding agent. The harness around it determines what work the agent can see, which tools it can use, how it runs code, and what happens when a step fails. Compare that complete workflow against your engineering work, rather than choosing from a model leaderboard or feature checklist.

## What belongs to the harness?

The word *harness* is used differently across products, but a useful working definition is the software that runs the model-and-tool loop. Depending on the system, that can include:

- The model, system instructions, and context assembly.
- Tool definitions and the code that executes tool calls.
- A shell, editor, browser, repository, or other work environment.
- Permission checks, network controls, and human approvals.
- Session state, retries, compaction, and recovery after interruption.
- The interface that presents progress, changes, and results to a person.

These responsibilities may live in one application or several services. For example, the [OpenAI Agents API architecture](https://developers.openai.com/api/docs/guides/agents-api/architecture) distinguishes the harness that runs the model and tool loop, the environment where commands and files are available, and the application server that connects the session to a product. That separation is a helpful way to inspect other systems too: ask which component owns each capability, and where its trust boundary sits.

An IDE completion tool, a terminal agent, a background repository worker, and a hosted coding agent may use similar models while providing very different working conditions. They should not be treated as interchangeable because they all accept a prompt and return code.

## Choose a representative task before choosing a product

Start with a real task shape from your team. A small bug fix might require an issue description, a repository, a failing test, a change, a local test run, and a reviewable diff. A dependency migration might require many files, external documentation, repeated test runs, and a rollback plan. A background agent may need to wait for CI or a human response and then resume safely.

Write down the expected task path:

1. What repository, issue, design document, and history does the agent receive?
2. Where does it work: the developer's checkout, a container, a remote workspace, or a disposable worktree?
3. Which tools can it call, and which of those can change external state?
4. What must pass before the change is considered complete?
5. Who reviews the diff and approves sensitive actions?
6. What should happen if a command fails, the session stops, or the agent reaches its time limit?

The answers expose requirements that a generic “supports tools” feature list hides. If an agent cannot inspect the project instructions or run the relevant tests, a strong model may still produce a plausible but unusable patch. If it can modify files but cannot show a clear diff, the review boundary becomes harder to maintain.

## Compare the execution model

### Context and repository access

Check exactly what the agent can see: the current file, nearby code, project-wide search, issue history, documentation, and generated files. Ask how ignored files, secrets, and large repositories are handled. If the context is assembled automatically, find out whether you can inspect or correct that context when it is incomplete.

For a repository task, test whether the agent follows local instructions, finds the code path that owns the behavior, and notices relevant tests. The useful signal is not just whether it edits the right file; it is whether it identifies the right boundary and produces a change that fits the repository.

### Tools and feedback loops

List the tools the agent can use and the feedback each one returns. A shell that can run tests is different from a tool that can only read files. A browser tool that can inspect a preview is different from a general HTTP request. A Git integration may create a branch or pull request, while another tool may only prepare a diff.

Observe the loop after the first attempt. Does the agent inspect a failing test and make a targeted correction? Does it rerun the relevant check? Does it distinguish a test failure from a tool or environment failure? Can it stop when evidence shows the task is not safe to continue?

### Workspace isolation and permissions

Find out whether work is isolated per task and how writes, network access, and credentials are controlled. The [OpenAI sandbox security guide](https://developers.openai.com/api/docs/guides/agents-api/environments/security) states a general risk that applies to agent environments: generated code can access the files, credentials, and network that are available to that environment. Treat the environment as part of the security design, not as a convenience setting.

For routine code changes, a disposable branch or worktree can limit accidental damage. For production systems, credentials should stay behind a trusted service that checks authorization independently. Network access should be limited to destinations the task actually needs. Human approval is useful for consequential actions, but it does not replace isolation or least privilege.

### State and recovery

Long tasks need a clear answer to “what happens next?” Check whether the agent can resume after a timeout or interruption without repeating side effects. Look for persisted task state, idempotent operations, event-driven waits, retry limits, and a visible record of tool calls. A summary written to the model's context is not, by itself, durable workflow state.

If the task is short, a simpler synchronous loop can be easier to inspect. If a task waits on CI, a reviewer, or a deployment event, durable resumption may justify the extra infrastructure.

## Build a small, fair evaluation

Use a few task families that resemble your actual workload: a localized bug fix, a cross-file feature, a code review, a test repair, a dependency update, and a task with ambiguous requirements. You do not need dozens of artificial prompts to learn whether a tool fits your work.

For each trial, keep the starting commit, issue description, project instructions, available tools, and success criteria fixed. Give each system the same clean environment and the same time or tool budget. If the interface or environment differs by design, record the difference instead of pretending the comparison is controlled.

Score the result on several dimensions:

| Dimension | What to inspect |
| --- | --- |
| Correctness | Does the change satisfy acceptance criteria and pass the relevant checks? |
| Reviewability | Is the diff focused, explainable, and easy to review? |
| Tool use | Does the agent choose useful commands and respond to their actual output? |
| Reliability | Does it recover from ordinary failures without looping or hiding errors? |
| Safety | Are writes, network calls, and privileged actions bounded and visible? |
| Human effort | How much clarification, correction, and review did the task require? |
| Operational fit | Can the team manage its access, logs, cost, isolation, and lifecycle? |

Record failures as well as successful completions. A task that passes once may still be unreliable; repeat ambiguous or multi-step cases enough to reveal variation. Track wall-clock time and model or tool usage if those matter to your budget, but keep the task outcome and human review effort visible. A shorter run is not better if it leaves unsafe or incorrect work behind.

Do not collapse unlike tasks into one score without explaining the weighting. A team can use a rubric to make trade-offs explicit, but the rubric is a decision aid, not a universal benchmark.

## Match the harness to the work

| Work pattern | Capabilities to prioritize |
| --- | --- |
| Inline edits and quick questions | Low-friction editor context, predictable suggestions, easy rejection |
| Repository-wide changes | Search, multi-file editing, test execution, diff review, isolated workspace |
| Background tasks | Durable state, clear progress, bounded permissions, safe interruption and resume |
| CI or maintenance automation | Reproducibility, idempotency, restricted credentials, machine-readable results |
| Sensitive code or data | Strong isolation, narrow network access, auditable tools, explicit approval boundaries |

These are starting points, not product categories with fixed winners. A team may use an interactive editor for exploration and a separate isolated worker for longer changes. Choose the least complicated arrangement that satisfies the workflow's access and reliability requirements.

## A practical selection checklist

Before adopting a coding harness, answer these questions with a small pilot:

- Can it see the instructions, code, and tests needed for a representative task?
- Can you tell which files and tools it used?
- Does it return a reviewable diff and explain what it did not verify?
- Are commands and external actions limited to the intended permissions?
- Can you reproduce or resume work after interruption?
- Does it fit the team's repository, identity, and review workflow?
- Are latency, usage, and human correction effort acceptable for the tasks that matter?

If those answers are unclear, keep the pilot small. Start with tasks that produce inspectable changes and objective feedback, then widen access only after the boundaries and failure handling are understood. The best coding harness is the one whose workflow you can evaluate, constrain, and review against your own engineering requirements.

For related design questions, see [how to build a reliable agent harness (in Persian)](/fa/writing/agent-harness/), [common production agent failures (in Persian)](/fa/writing/agent-production-failures/), and [the hidden costs of AI agents (in Persian)](/fa/writing/agent-costs/).

## References

- [OpenAI Agents API: Architecture](https://developers.openai.com/api/docs/guides/agents-api/architecture)
- [OpenAI Agents API: Sandbox security](https://developers.openai.com/api/docs/guides/agents-api/environments/security)
