**Curriculum Design**

This document defines the complete curriculum design for the book. It covers the introduction and Chapters 1-20 as one coherent learning path: foundations, first applications, retrieval and grounding, tools and agents, local models and customization, production operations, evaluation, security, multimodal systems, voice, and the final domain-product capstone.

The curriculum is market-aligned and portfolio-driven. Every major chapter should move the reader from concept to runnable artifact, then from runnable artifact to measurable engineering evidence.

The second half of the design also incorporates the two market reports:

- Job-market-2026.md
- llm_engineer_market_analysis_report.md

The reports agree on the core market signal: employers want engineers who can build, evaluate, secure, deploy, and explain production LLM systems. The curriculum therefore moves from "can call a model API" to "can ship a measurable, defensible AI product."

Guiding requirements:

- RAG remains the strongest applied entry point.
- Agentic systems are the fastest-growing specialization.
- Evaluation, observability, and LLMOps are major differentiators.
- SQL, FastAPI, Docker, cloud deployment, and access control must be visible.
- Open-source model skills should include local inference, quantization, PEFT, and serving trade-offs.
- Every major chapter should produce a portfolio artifact with metrics, tests, or architecture notes.
- Named tools should be taught at the depth appropriate to the chapter: one primary hands-on path, plus explicit comparison labs for credible alternatives. Listing a product is not enough; students should know what problem it solves, when to choose it, and its operational trade-offs.
- Foundational Python data work needed after Chapter 5 should remain visible through NumPy, pandas, and scikit-learn exercises rather than being assumed.

## Introduction: Modern LLM Engineering

Market alignment:
- Target roles: Applied AI Engineer, LLM Engineer, GenAI Developer, AI Solutions Engineer.
- Job keywords: LLM application stack, RAG, agents, evaluation, observability, security, production AI.
- Portfolio artifact: personal learning roadmap and project plan for the book.

Students learn what an LLM engineer actually does: build systems around models rather than train foundation models from scratch. The introduction frames LLM engineering as systems engineering for probabilistic software, combining software engineering, data engineering, ML engineering, security, and product judgment.

Core skill: translate vague business needs into AI system requirements, constraints, metrics, and risks.

Scope boundary:
- In scope: role definition, production mindset, application stack, business-to-system translation, reliability, cost, risk, reader prerequisites, and learning outcomes.
- Out of scope: hands-on coding, deep model internals, detailed retrieval, tool calling, fine-tuning, deployment, and governance mechanics.
- Handoff: Chapter 1 begins the technical foundation by explaining how generation works.

Mini-project: write a one-page AI product brief for a support assistant, internal knowledge assistant, or domain copilot. Identify users, data sources, model responsibilities, deterministic code responsibilities, privacy risks, cost risks, and success metrics.

Evaluation signal: the brief must distinguish model behavior, application logic, data pipelines, evaluation, and operations instead of treating the LLM as a single black box.

## Chapter 1: LLM Text Generation

Market alignment:
- Target roles: Applied AI Engineer, LLM Application Engineer, AI Product Engineer.
- Job keywords: tokens, tokenizer, context window, attention, logits, sampling, temperature, streaming, latency, KV cache.
- Portfolio artifact: generation-behavior lab notes comparing decoding settings and model behavior.

Students learn the core mechanics of text generation: prompts become tokens, decoder-only Transformers predict the next token, logits become probabilities, and decoding controls shape the output. The chapter connects model internals directly to production concerns: output latency, token cost, context limits, hallucinations, truncation, format drift, repetition, lost context, and KV-cache memory pressure.

Core skill: reason about LLM behavior using generation mechanics rather than treating model output as magic.

Scope boundary:
- In scope: autoregressive generation, prefill and decode phases, tokenization, subwords, decoder-only Transformers, causal masking, attention, context windows, KV cache, logits, softmax, temperature, top-p/top-k, stop sequences, max tokens, structured-output constraints, playground experimentation, and common generation failures.
- Out of scope: training math, backpropagation, full Transformer implementation, inference-server internals, retrieval, tool calling, and production deployment.
- Handoff: Chapter 2 uses these mechanics to evaluate which model is appropriate for a task.

Mini-project: use a chat playground to compare one prompt across multiple decoding settings and, if possible, multiple models. Record model name, temperature, output length, visible latency, and token usage.

Production upgrade: reproduce the best playground configuration through an API call before trusting the result.

Evaluation signal: write five observations explaining consistency, verbosity, usefulness, temperature impact, and cross-model differences.

## Chapter 2: Production Model Selection

Market alignment:
- Target roles: LLM Engineer, AI Platform Engineer, Solutions Engineer, Applied AI Engineer.
- Job keywords: model selection, closed models, open-weight models, hosted inference, local inference, benchmarks, leaderboards, model routing, cost, latency, reliability.
- Portfolio artifact: model decision record for a realistic support-assistant scenario.

Students learn that model choice is an ongoing engineering decision, not a one-time leaderboard pick. They compare generator, embedder, reranker, classifier, router, reasoning, code, vision, audio, and fallback models. They evaluate closed proprietary APIs, hosted open-weight inference, and self-hosted/local inference under task fit, quality bar, latency, cost, context needs, output contract, privacy, licensing, operational burden, and migration path.

Core skill: choose the cheapest and simplest model strategy that meets the product's quality, latency, privacy, and reliability requirements.

Scope boundary:
- In scope: model-role decomposition, SLMs versus frontier models, open versus closed models, managed APIs, hosted inference, local inference, model cards, leaderboards, public benchmarks, private golden datasets, task-fit matrices, cost per successful task, reliability, privacy, licensing, fallback planning, provider adapters, and model decision records.
- Out of scope: full benchmarking infrastructure, detailed self-hosted serving, fine-tuning, RAG implementation, deployment automation, and governance frameworks.
- Handoff: Chapter 3 turns a selected model and API strategy into a working chatbot application.

Mini-project: create a model decision record for a customer-support assistant. Compare one managed API model, one cheaper/smaller model, and one open-weight or hosted open-weight option.

Production upgrade: define a default model, fallback model, review trigger, migration strategy, and private evaluation plan.

Evaluation signal: the MDR must explain what public leaderboards prove, what they cannot prove, and what must be measured on real product data before launch.

## Chapter 3: Streaming Chatbot Applications

Market alignment:
- Target roles: LLM Application Engineer, Backend AI Engineer, Full-Stack AI Engineer.
- Job keywords: chat API, message history, streaming, OpenAI-compatible providers, environment variables, API keys, context trimming, token usage, error handling.
- Portfolio artifact: Telegram AI assistant bot or CLI chatbot with streaming, state, and operational logs.

Students learn the basic application loop behind chat systems: maintain message history, send the relevant context to a stateless API, stream the response, append the assistant reply, and repeat. The chapter covers roles, system prompts, user messages, assistant messages, `.env` configuration, OpenAI-compatible endpoints, provider headers, streaming deltas, usage metadata, command-line controls, context growth, trimming, optional summaries, token/cost visibility, and runtime failure handling.

Core skill: build a working chat application that manages state, secrets, streaming, context size, and API failures explicitly.

Scope boundary:
- In scope: Chat Completions-style message arrays, provider configuration, `.env` and `.env.example`, API key safety, OpenAI-compatible base URLs, streaming output, usage metadata, CLI control commands, context trimming, latency and token logging, and handling missing config, auth errors, rate limits, context-length errors, network failures, and interrupted streams.
- Out of scope: structured outputs, RAG, tool calling, persistent production memory, web UI architecture, advanced retries, observability dashboards, and deployment.
- Handoff: Chapter 4 makes prompts and model outputs reliable enough for downstream software.

Mini-project: build a Telegram assistant bot that keeps separate short histories per chat ID, trims history, calls an OpenAI-compatible model, handles Telegram and provider errors, and exposes `/start`, `/clear`, and `/stats`.

Production upgrade: add per-user rate limits, optional SQLite history, fallback model configuration, and logs for model, chat ID, latency, token usage, and error type.

Evaluation signal: test missing keys, normal messages, long conversations, `/clear`, rate-limit behavior, provider errors, context-length failures, and secret-safe logs.

## Chapter 4: Prompting and Structured Outputs

Market alignment:
- Target roles: LLM Application Engineer, AI Product Engineer, LLM Evaluation Engineer.
- Job keywords: system prompt, user prompt, instruction hierarchy, prompt injection, structured outputs, JSON Schema, Pydantic, validation, prompt versioning.
- Portfolio artifact: validated support-ticket triage extractor with typed output, repair retry, and structured logs.

Students learn that production prompting is a runtime contract, not clever wording. They separate stable system instructions from dynamic user data, use delimiters for untrusted inputs, apply instruction hierarchy, design prompt security boundaries, use task-specific prompt patterns, assemble context deliberately, control cost and latency, version prompts as Markdown assets, debug failures systematically, and validate machine-consumed outputs with JSON Schema and Pydantic.

Core skill: design prompts and output contracts that downstream software can validate, debug, version, and evaluate.

Scope boundary:
- In scope: system/user prompt separation, KERNEL prompt anatomy, XML-style delimiters, instruction hierarchy, prompt injection basics, classification/extraction/RAG/tool/rewrite prompt patterns, assumption audits, anti-prompts, prompt chaining, few-shot examples, context assembly, prompt caching order, cost and latency effects, prompt versioning, debugging logs, JSON Schema, structured outputs, Pydantic validation, business-rule validation, refusals, and one repair retry.
- Out of scope: full RAG systems, production tool calling, agents, complete security red teaming, full eval platforms, deployment, and observability dashboards.
- Handoff: Chapter 5 uses prompt/context discipline to begin retrieval and semantic search.

Mini-project: build a support-ticket triage extractor that classifies category and priority, summarizes the ticket, flags human escalation, records missing fields, validates with Pydantic, and retries once on validation failure.

Production upgrade: store the system prompt in `prompts/ticket_triage.system.md`, log prompt version, model, raw output, parsed output, validation result, token usage, latency, retry count, and business-rule failures.

Evaluation signal: test valid tickets, ambiguous tickets, prompt-injection text inside customer input, missing fields, urgent tickets that must require human review, malformed model output, and schema violations.

## Chapter 5: Embeddings and Semantic Search

Market alignment:
- Target roles: RAG Engineer, Search Engineer, Applied AI Engineer, AI Integration Engineer.
- Job keywords: embeddings, semantic search, vector database, FAISS, pgvector, Qdrant, chunking, metadata, cosine similarity, retrieval metrics.
- Portfolio artifact: local semantic-search workbench with persistent index and retrieval metrics.

Students learn how to turn text into vectors, split documents into useful chunks, preserve metadata, choose embedding models, generate and cache embeddings, store vectors, search by similarity, and evaluate retrieval separately from generation. The chapter establishes the retrieval foundation for RAG without adding answer generation yet.

Core skill: build and evaluate a local semantic search engine over real documents.

Scope boundary:
- In scope: vectors, embeddings, dense representations, cosine similarity, dot product, normalization, chunking strategies, overlap, metadata, tenant IDs, hosted versus local embedding models, MTEB as a shortlist tool, embedding model/version coupling, batch generation, embedding caches, raw text storage, async ingestion concepts, FAISS, hnswlib, LanceDB, ChromaDB, pgvector, Qdrant, Milvus, Weaviate, Pinecone, Elasticsearch/OpenSearch-style options, ANN concepts, persistent FAISS indexes, search result mapping, Hit Rate@k, Recall@k, MRR, nDCG, and retrieval debugging.
- Out of scope: answer generation, RAG prompting, citations, hybrid retrieval, BM25, reranking, permission-aware production retrieval, tool use, agents, deployment, and LLM-as-judge evaluation.
- Handoff: Chapter 6 uses semantic search results to build a grounded RAG assistant with citations.

Mini-project: build a local semantic document search engine over 5-20 files. Chunk the documents, embed with `sentence-transformers/all-MiniLM-L6-v2`, persist vectors and metadata, implement `search(query, top_k)`, and create at least 10 labeled retrieval queries.

Production upgrade: change one retrieval variable, such as chunk size or embedding model, rebuild the index, rerun the evaluation, and document whether Hit Rate@3 and MRR improved or degraded.

Evaluation signal: report retrieved chunks, source filenames, scores, Hit Rate@3, MRR, and a short failure analysis for missed or weak queries.

## Chapter 6: Retrieval-Augmented Generation

Market alignment:
- Target roles: RAG Engineer, Applied AI Engineer, Full-Stack LLM Engineer.
- Job keywords: RAG, embeddings, vector database, grounding, citations, context window, FastAPI, Streamlit.
- Portfolio artifact: PDF Q&A bot with source citations and an evidence inspector.

Students learn the complete basic RAG loop: reuse the Chapter 5 search index, retrieve relevant chunks, build a context-aware prompt, manage context-window budgeting, generate grounded answers, cite sources, handle "I don't know," and debug missing or irrelevant context. The chapter makes prompt engineering explicit through instruction hierarchy, few-shot examples, delimiters, context compression, refusal rules, and prompt templates. It introduces structured outputs with schema validation and contrasts a direct SDK implementation with LangChain and LlamaIndex. Prerequisites are Chapter 5 semantic search plus Chapter 4 prompting discipline.

Core skill: build a document-grounded assistant that answers from company knowledge instead of guessing.

Scope boundary:
- In scope: single-pass RAG answer generation, prompt engineering, context budgeting, grounded prompting, structured outputs, schema validation, "I don't know" behavior, citation formatting, evidence inspection, LangChain/LlamaIndex orientation, and a small API/UI around the loop.
- Out of scope: hybrid retrieval, reranking, permission-aware retrieval, tool calling, agents, production deployment, observability, and full evaluation/security frameworks.
- Handoff: Chapter 7 improves the retriever; Chapter 8 adds enterprise data and permissions; Chapters 15-17 productionize, monitor, evaluate, and secure the system.

Mini-project: build a PDF Q&A bot for HR, product, or policy documents that answers with file name, page or section, and quoted supporting chunks. Return a validated structured response containing the answer, citations, confidence, and refusal reason.

Production upgrade: expose the RAG loop through a minimal FastAPI endpoint and a small UI so the project feels like a product, not only a notebook.

Evaluation signal: add smoke tests for "answerable," "not answerable," "citation required," and invalid structured-output cases.

## Chapter 7: Hybrid Retrieval, Reranking, and RAG Evaluation

Market alignment:
- Target roles: RAG Engineer, Search Engineer, LLM Evaluation Engineer.
- Job keywords: BM25, hybrid search, metadata filters, reranking, RRF, recall@k, MRR, NDCG, RAGAS, DeepEval.
- Portfolio artifact: retrieval quality report showing measurable improvement over basic vector search.

Students learn why basic vector search fails and how to improve it with custom chunking, BM25, metadata filtering, hybrid dense+sparse retrieval, query rewriting, reciprocal rank fusion, cross-encoder reranking, and retrieval evaluation. Embedding and reranking labs use sentence-transformers. NumPy supports vector inspection, pandas supports evaluation tables, and scikit-learn provides baseline similarity, classification, and metric utilities. Prerequisites are a working RAG system from Chapter 6.

Technology coverage:
- Local and embedded search: FAISS and Chroma.
- Open-source vector databases: Milvus, Qdrant, Weaviate, and pgvector.
- Managed vector search: Pinecone.
- Search-engine and cloud alternatives: Elasticsearch, OpenSearch, and Azure AI Search.
- Selection criteria: filtering, hybrid-search support, scale, latency, consistency, operations, tenancy, and cost. Students implement one local option and one production-oriented option, then document why the others would or would not fit.

Core skill: increase recall and precision for real enterprise documents where exact names, IDs, dates, abbreviations, and domain terms matter.

Scope boundary:
- In scope: custom chunking, sentence-transformers, vector-database selection, hybrid dense+sparse search, query rewriting, metadata filters, reranking, golden query sets, retrieval metrics, and false-positive/false-negative analysis.
- Out of scope: basic answer synthesis, enterprise authorization, SQL-backed grounding, tool execution, agent workflows, production monitoring, and security red teaming.
- Handoff: Chapter 8 applies retrieval to enterprise data and permissions; Chapter 9 turns retrieval into one callable tool inside an LLM application.

Mini-project: upgrade the Chapter 6 bot into a production-style support search system with keyword+vector search, custom chunking, metadata filters, reranking, and retrieval-quality tests. Compare FAISS or Chroma with one of Qdrant, Weaviate, Milvus, Pinecone, pgvector, Elasticsearch, OpenSearch, or Azure AI Search.

Production upgrade: create an evaluation script that runs a fixed query set and prints retrieval metrics before and after reranking.

Evaluation signal: show a measurable lift in recall@k, precision@k, retrieval hit rate, MRR, NDCG, or citation faithfulness. Use RAGAS and DeepEval for an initial automated RAG evaluation, while explaining where their scores need human validation.

## Chapter 8: Enterprise Data Integration and Permission-Aware RAG

Market alignment:
- Target roles: Enterprise RAG Engineer, LLM Solutions Engineer, AI Integration Engineer.
- Job keywords: SQL, Postgres, ETL, metadata, access control, data freshness, row-level security, audit logging, knowledge graph.
- Portfolio artifact: permission-aware internal assistant connected to both documents and structured data.

Students learn how LLM systems connect to enterprise data without leaking information or serving stale context. Topics include scraping, dataset creation, deduplication, preprocessing, custom embeddings, domain datasets, document ingestion pipelines, parser selection, metadata contracts, incremental indexing, deletion and reindexing, SQL-backed grounding, access-control filters, tenant isolation, PII handling, audit logs, and freshness checks.

Data-platform coverage:
- PostgreSQL production practices: schema design, indexing, query tuning, migrations, transactions, connection pooling, row-level security, and pgvector operations.
- Batch and transformation orchestration: Airflow and dbt.
- Distributed and streaming pipelines: Spark and Kafka, including event-driven ingestion.
- Data quality: Great Expectations and Soda.
- Reusable online features and signals: Feast as a feature store, including when a feature store is unnecessary for an LLM application.
- Lakehouse integration: Databricks data lakehouse patterns for governed domain datasets. Mosaic AI is introduced as the Databricks model and GenAI application layer.

Core skill: connect LLMs to private company data in a way that respects permissions, provenance, and update cycles.

Scope boundary:
- In scope: scraping, dataset creation, deduplication, preprocessing, custom embeddings, domain datasets, batch/streaming ingestion, data quality, metadata contracts, incremental indexing, freshness, deletion/reindexing, PostgreSQL production, SQL grounding, access filters, tenant isolation, PII handling, audit logs, and data lakehouse integration.
- Out of scope: write-capable tool execution, multi-step planning, agent memory, model serving infrastructure, dashboards, and full security red teaming.
- Handoff: Chapter 9 wraps these data sources as safe tools; Chapters 15-17 deploy, observe, evaluate, and govern the resulting application.

Mini-project: build an internal policy assistant that combines a vector store for documents with a production-style PostgreSQL database for employees, teams, or product records. Create a small Airflow or dbt pipeline, add Great Expectations or Soda checks, and optionally consume a Kafka event to refresh one data source. The assistant must filter results based on user role and explain which source was used.

Production upgrade: add migrations, indexes justified with query plans, a reindex command, a data freshness timestamp, and an access-control test suite.

Evaluation signal: include tests proving that unauthorized users cannot retrieve restricted chunks or rows.

## Chapter 9: Tool-Connected LLM Apps and MCP-Style Interfaces

Market alignment:
- Target roles: Agentic AI Engineer, Applied AI Engineer, LLM Application Engineer.
- Job keywords: function calling, tool schemas, MCP, A2A, API integration, validation, approval gates, audit trail.
- Portfolio artifact: customer-support copilot that safely calls read and write tools.

Students learn tool/function calling, tool schemas, structured outputs, argument validation, tool-result injection, API safety, safe repeatable tool calls, permissions, human approval for risky actions, and tool-call logging. Students first implement the contract directly, then compare LangChain, LlamaIndex, and PydanticAI abstractions. The Model Context Protocol (MCP) is implemented rather than only described: students build a small server and client, inspect capabilities and schemas, and compare lightweight ReActMCP and EasyMCP patterns where appropriate. Safety guardrails include input validation, harmful-content filtering, output validation, and explicit read/write boundaries.

Core skill: let the model operate software safely: search a database, check an order, create a ticket, draft a message, or call an internal API.

Scope boundary:
- In scope: single-step tool/function calling, structured outputs, tool schemas, argument validation, read/write tool separation, idempotency, approval gates, MCP server/client basics, ReActMCP, EasyMCP, safety guardrails, harmful-content filtering, and tool-call audit logs.
- Out of scope: autonomous planning loops, persistent memory, checkpointed workflows, multi-agent collaboration, deployment infrastructure, and broad eval/security suites.
- Handoff: Chapter 10 adds durable state and multi-step control flow; Chapter 11 coordinates multiple agents; Chapter 17 stress-tests unsafe tool use.

Mini-project: build a customer-support copilot that can search docs, look up order status, draft a refund message, and require approval before any write action. Expose at least two capabilities through MCP and validate all tool inputs and structured outputs with PydanticAI or equivalent Pydantic schemas.

Production upgrade: log every tool call with user, arguments, result summary, latency, and approval status.

Evaluation signal: test malformed arguments, schema violations, harmful inputs, missing permissions, repeated calls, unsafe write attempts, and MCP client/server compatibility.

## Chapter 10: Stateful Agent Workflows

Market alignment:
- Target roles: Agentic AI Engineer, Workflow Automation Engineer, Applied AI Engineer.
- Job keywords: LangGraph, state machine, planning, memory, checkpoint, retry, interruption, human-in-the-loop.
- Portfolio artifact: resumable research assistant workflow with checkpoints and approval steps.

Students learn ReAct reasoning/action loops, state machines, planning loops, conversation state, user preferences, intermediate task data, checkpoints, retries, interruptions, human-in-the-loop review, resumable execution, workflow graphs, and basic agent observability. LangGraph is the primary hands-on framework. PydanticAI and the OpenAI Agents SDK are compared for typed agents, handoffs, sessions, and guardrails, while direct implementations keep framework behavior understandable. Prerequisites are Chapter 9 tool calling plus earlier RAG/search skills.

Core skill: build one agent or one workflow that can work across many steps without becoming an uncontrolled loop.

Scope boundary:
- In scope: one stateful ReAct agent or workflow, LangGraph, PydanticAI, OpenAI Agents SDK, state machines, planning loops, checkpoints, retries, interruptions, resumability, human review, structured agent outputs, guardrails, and basic workflow observability.
- Out of scope: teams of agents, supervisor/router patterns, cross-agent negotiation, production deployment, full observability dashboards, and enterprise governance.
- Handoff: Chapter 11 expands from one workflow to multiple collaborating agents; Chapters 15-17 later deploy, monitor, and evaluate agent systems.

Mini-project: build a LangGraph research assistant workflow that takes a topic, plans subtasks, searches documents or notes, saves findings, asks for approval before finalizing, and produces a structured report. Rebuild one thin path with PydanticAI or the OpenAI Agents SDK and compare complexity and control.

Production upgrade: persist state so an interrupted workflow can resume without losing intermediate work.

Evaluation signal: test retry behavior, checkpoint recovery, human approval, and failure handling.

## Chapter 11: Multi-Agent Systems and Agent Evaluation

Market alignment:
- Target roles: Agentic AI Engineer, Senior LLM Engineer, AI Automation Architect.
- Job keywords: multi-agent, supervisor, router, specialist agent, handoff, shared state, agent evaluation, cost control.
- Portfolio artifact: multi-agent proposal builder with a supervisor and specialist workers.

Students learn when multi-agent systems are useful and when they are overkill: supervisor/router patterns, specialist agents, handoffs, shared state, parallel research, conflict resolution, cost explosion, observability, and evaluation. CrewAI and AutoGen are compared with LangGraph and the OpenAI Agents SDK so students can distinguish role-based teams, conversational agents, and explicit workflow graphs. Prerequisites are Chapter 10 stateful single-agent workflows and Chapter 9 tool use.

Core skill: decompose complex work into coordinated specialists without making the system impossible to debug.

Scope boundary:
- In scope: CrewAI, AutoGen, LangGraph and OpenAI Agents SDK comparisons, multi-agent coordination, supervisor/router patterns, specialist agents, handoffs, shared state, parallel work, conflict handling, agent tracing, cost comparison, and agent-level evaluation.
- Out of scope: introductory tool schemas, single-agent state management, local model operation, production serving, infrastructure dashboards, and security governance.
- Handoff: Chapters 9 and 10 provide the tool and state foundations; Chapters 15-17 make the multi-agent system deployable, observable, and safe.

Mini-project: build a multi-agent proposal builder with CrewAI or AutoGen: one agent gathers requirements, one searches case studies, one drafts the proposal, one checks risks/compliance, and a supervisor merges the final output. Compare it with a LangGraph or OpenAI Agents SDK implementation of the same workflow.

Production upgrade: add trace logs for each agent decision, handoff, and final merge.

Evaluation signal: compare single-agent and multi-agent outputs on quality, cost, latency, and failure rate.

## Chapter 12: Local LLM Inference

Market alignment:
- Target roles: Open-Source LLM Engineer, Privacy-Focused AI Engineer, AI Platform Engineer.
- Job keywords: Llama, Mistral, Qwen, DeepSeek, GGUF, quantization, Ollama, llama.cpp, Hugging Face, local inference.
- Portfolio artifact: local private document assistant using an open-weight model.

Students learn how to run open-source models locally or on private servers with Hugging Face Transformers and Ollama: model weights, tokenizers, licensing, VRAM/RAM limits, quantization, GPTQ, GGUF, llama.cpp-style runtimes, context length, CPU vs GPU trade-offs, basic batching, and privacy trade-offs. Students compare full precision, common low-bit quantization, GPTQ, and GGUF packaging, including quality, compatibility, memory, and speed. Prerequisites are all earlier app patterns, because local inference replaces the model backend without changing the product design.

Core skill: deploy a small local model for private, low-cost, or offline use cases.

Scope boundary:
- In scope: Hugging Face Transformers, Ollama, llama.cpp, model files, tokenizers, licenses, quantization, GPTQ, GGUF, local runtimes, hardware limits, privacy trade-offs, and hosted-vs-local comparisons.
- Out of scope: modifying model weights, supervised fine-tuning, preference tuning, distillation, high-throughput serving, Kubernetes deployment, and production observability.
- Handoff: Chapter 13 modifies model behavior through PEFT; Chapter 15 serves models and apps under production constraints.

Mini-project: build a local private document assistant that loads one model with Hugging Face Transformers and serves another quantized variant through Ollama or llama.cpp, then connects it to the Chapter 6 RAG pipeline.

Production upgrade: compare local and hosted model outputs for quality, latency, privacy, and cost.

Evaluation signal: record p50/p95 latency, memory use, answer quality, and failure cases.

## Chapter 13: LLM Fine-Tuning with LoRA, QLoRA, and PEFT

Market alignment:
- Target roles: LLM Fine-Tuning Specialist, Applied ML Engineer, Open-Source Model Engineer.
- Job keywords: SFT, LoRA, QLoRA, PEFT, Hugging Face, TRL, dataset quality, train/validation split, baseline comparison.
- Portfolio artifact: fine-tuned support-response model with a before/after evaluation.

Students learn instruction tuning with Hugging Face Transformers, Hugging Face Datasets, PEFT, and TRL: dataset formatting, dataset quality, train/validation splits, supervised fine-tuning, LoRA, QLoRA, hyperparameters, evaluation before/after tuning, overfitting, safety risks, and when not to fine-tune. pandas supports dataset inspection and cleaning, NumPy supports numerical analysis, and scikit-learn supports splitting and baseline metrics. MLflow and Weights & Biases (W&B) track experiments, artifacts, datasets, and model versions. Prerequisites are local inference from Chapter 12 plus evaluation habits from earlier RAG chapters.

Core skill: adapt model behavior for a narrow task, style, or format, not inject fresh factual knowledge that belongs in RAG.

Scope boundary:
- In scope: Hugging Face Transformers, Hugging Face Datasets, PEFT, TRL, LoRA, QLoRA, pandas, NumPy, scikit-learn, supervised fine-tuning, dataset formatting, data quality, train/validation splits, experiment tracking with MLflow and W&B, hyperparameter basics, overfitting checks, and before/after evaluation.
- Out of scope: DPO, RLHF/RLAIF, distillation, continued pre-training, foundation-model training, production model serving, and using fine-tuning as a replacement for factual retrieval.
- Handoff: Chapter 14 covers advanced training judgment; Chapter 15 packages and serves adapted models when they are worth deploying.

Mini-project: fine-tune a support-response model on approved company replies, then compare it against a prompt-only baseline and a RAG baseline.

Production upgrade: package the fine-tuned model with a model card, dataset card, reproducible MLflow or W&B run, model version, known limitations, and rollback guidance.

Evaluation signal: compare accuracy, format compliance, refusal behavior, cost, and latency before and after tuning.

## Chapter 14: Preference Tuning, Distillation, and Training Judgment

Market alignment:
- Target roles: LLM Training Specialist, Applied Research Engineer, Model Optimization Engineer.
- Job keywords: DPO, RLHF, RLAIF, synthetic data, distillation, teacher-student, continued pre-training, compute budget.
- Portfolio artifact: distilled classifier or mini assistant with quality, speed, and cost comparison.

Students gain advanced awareness of preference tuning with TRL, DPO, RLHF/RLAIF concepts, synthetic data generation, teacher-student distillation, rationale transfer, data quality, tokenizer choices, continued pre-training, compute budgeting, and why most companies should not pre-train from scratch. Evaluation includes task metrics plus BLEU and perplexity where they are meaningful, with explicit discussion of why neither metric is a universal measure of assistant quality.

Core skill: decide when a model should be adapted, distilled, routed, or left alone.

Scope boundary:
- In scope: TRL preference-tuning workflows, DPO/RLHF/RLAIF awareness, synthetic data, teacher-student distillation, BLEU, perplexity, continued pre-training judgment, tokenizer/data-quality considerations, compute budgeting, and build-vs-buy-vs-fine-tune decisions.
- Out of scope: hands-on foundation-model pre-training at scale, distributed training internals, research-grade alignment algorithms, and production serving mechanics.
- Handoff: Chapter 13 provides basic PEFT experience; Chapter 15 turns selected model decisions into deployable services.

Mini-project: build a distilled classifier or mini assistant from a stronger teacher using synthetic examples, then evaluate quality, speed, and cost.

Production upgrade: write a build-vs-buy-vs-fine-tune decision memo.

Evaluation signal: show whether the smaller model is good enough under real latency and cost constraints.

## Chapter 15: Production LLM Serving and LLMOps

Market alignment:
- Target roles: LLMOps Engineer, AI Platform Engineer, Production LLM Engineer.
- Job keywords: FastAPI, Docker, Kubernetes, CI/CD, cloud, vLLM, TGI, TensorRT-LLM, batching, KV cache, autoscaling.
- Portfolio artifact: productionized RAG/agent API with deployment checklist.

Students learn production serving in two levels. First, they learn application-level serving: API design, API contracts, authentication, queueing, async background jobs, event-driven systems, rate limits, caching, streaming at scale, autoscaling, secrets, deployment, rollback, microservices, and service boundaries. Second, they learn open-model serving concepts: continuous batching, paged attention, KV-cache optimization, model versioning, GPU utilization, vLLM, TGI, and TensorRT-LLM. Students perform model lifecycle analysis from candidate selection and validation through registry, deployment, monitoring, replacement, and retirement.

Infrastructure and cloud coverage:
- Container and cluster foundations: Docker, Kubernetes, Helm, and Terraform.
- CI/CD pipelines: GitHub Actions and Azure DevOps, including tests, image builds, security checks, deployment approvals, and rollback.
- Azure path: Azure OpenAI, Azure AI Search, and Azure AKS.
- AWS path: AWS Bedrock, AWS SageMaker, and AWS EKS.
- GCP path: GCP Vertex AI and Gemini Enterprise.
- Databricks path: Databricks and Mosaic AI for governed data, model endpoints, evaluation, and GenAI applications.
- Deployment strategies: rolling deployment, blue/green deployment, canary deployment/canary release, and shadow deployment. Students implement one cloud path and compare the managed-model, managed-platform, and self-hosted alternatives.

Core skill: turn prototypes into services that survive real users.

Scope boundary:
- In scope: API serving and API contracts, authentication, queueing, async background jobs, event-driven systems, microservices and service boundaries, rate limits, caching, streaming, Docker, Kubernetes, Helm, Terraform, GitHub Actions, Azure DevOps, cloud platforms, model lifecycle analysis, model versioning, deployment handover, canary and shadow deployment, secrets, rollbacks, health checks, and open-model serving with vLLM and TensorRT-LLM.
- Out of scope: detailed observability dashboards, cost optimization programs, red-team testing, governance, dataset-level evaluation design, and model training.
- Handoff: Chapter 16 adds monitoring, reliability, and cost controls; Chapter 17 adds evaluation, security, and governance.

Mini-project: productionize a RAG/agent API with authentication, Docker, environment configuration, logging, retries, caching, a queue-backed async job, and explicit API contracts. Serve an open model with vLLM or TensorRT-LLM where hardware permits.

Production upgrade: provision infrastructure with Terraform and Helm, deploy through GitHub Actions or Azure DevOps to AKS, EKS, or another Kubernetes environment, and document a deployment handover runbook. Demonstrate a canary release or shadow deployment before full rollout.

Evaluation signal: run smoke tests, basic load tests, API contract tests, queue-failure tests, canary/shadow comparisons, rollback tests, and health checks.

## Chapter 16: Observability, Cost, and Reliability Engineering

Market alignment:
- Target roles: LLMOps Engineer, AI Platform Engineer, Senior Applied AI Engineer.
- Job keywords: observability, tracing, prompt versioning, model routing, fallback, p95 latency, token cost, dashboard, incident response.
- Portfolio artifact: LLMOps dashboard showing cost, latency, errors, traces, and model behavior.

Students learn the operational layer that makes LLM systems maintainable after deployment: structured logs, traces, dashboards, prompt versioning, model versioning, latency/quality tracking, token usage, model routing, semantic caching, fallbacks, budget alerts, SLA/SLO thinking, queue depth, retry storms, rate-limit handling, incident review, and production debugging. Cost tracking includes token cost, per-request cost, cache-hit rate, provider/model cost, and budget attribution.

Observability stack:
- OpenTelemetry instrumentation and the OpenTelemetry Collector for vendor-neutral logs, metrics, and traces.
- Prometheus for metrics and alerting, with Grafana dashboards.
- LangSmith and Langfuse for LLM and agent traces, prompt/version analysis, datasets, and feedback.
- Weights & Biases (W&B) and MLflow for experiment-to-production lineage and model lifecycle records.
- GPTCache as a concrete semantic caching implementation, including similarity thresholds, invalidation, privacy, and stale-answer risks.

Core skill: diagnose and improve live LLM systems using evidence instead of guesswork.

Scope boundary:
- In scope: logs, traces, dashboards, OpenTelemetry, OpenTelemetry Collector, Prometheus, Grafana, LangSmith, Langfuse, W&B, MLflow, prompt/model versioning, p50/p95 latency and quality tracking, token and per-request cost, model routing, semantic caching with GPTCache, fallbacks, cache-hit rates, budget alerts, incident review, and reliability debugging.
- Out of scope: first-time deployment setup, model training, retrieval algorithm design, prompt-injection defense, red teaming, and governance sign-off.
- Handoff: Chapter 15 provides the deployed service; Chapter 17 uses observability data inside evaluation, security, and governance workflows.

Mini-project: instrument the Chapter 15 app with OpenTelemetry, export through the OpenTelemetry Collector, and add Prometheus/Grafana plus LangSmith or Langfuse views for p50/p95 latency, quality, token cost, per-request cost, cache-hit rate, error categories, model/provider routing, and failed-request traces.

Production upgrade: add GPTCache, set a monthly cost budget, and implement fallback behavior when the preferred model is unavailable, too slow, or too expensive. Preserve prompt and model versions in every trace.

Evaluation signal: demonstrate a measurable reduction in cost, latency, or failure rate after one optimization.

## Chapter 17: Evaluation, AI Security, and Governance

Market alignment:
- Target roles: LLM Evaluation Engineer, AI Safety Engineer, Enterprise AI Architect, Senior LLM Engineer.
- Job keywords: golden datasets, LLM-as-judge, human review, hallucination testing, prompt injection, data leakage, guardrails, governance, red teaming.
- Portfolio artifact: evaluation and security harness for a production RAG/agent system.

Students learn evaluation pipelines, test datasets, golden answers, regression tests, LLM-as-Judge, human review, retrieval metrics, hallucination checks, citation faithfulness, structured-output checks, and A/B testing for prompts/models. RAG evaluation uses RAGAS, DeepEval, Giskard, and TruLens with MRR, retrieval hit rate, faithfulness, and task-specific measures; model evaluation revisits BLEU and perplexity only where appropriate.

Security and governance coverage includes prompt-injection testing, PII redaction, GDPR-aware AI, audit logs, harmful-content filtering, content moderation, safety guardrails, responsible AI, data leakage, insecure tool use, excessive agency, vector-store risks, red teaming, incident response, and risk ownership. Students compare cloud governance controls across Azure/GCP/AWS, including identity, data residency, retention, encryption, approval evidence, and auditability.

Core skill: measure and defend LLM systems instead of trusting demos.

Scope boundary:
- In scope: evaluation pipelines, RAGAS, DeepEval, Giskard, TruLens, LLM-as-Judge, MRR, retrieval hit rate, BLEU, perplexity, golden datasets, regression tests, human review, hallucination checks, citation faithfulness, structured-output checks, A/B testing for prompts/models, prompt-injection testing, PII redaction, GDPR-aware AI, audit logs, content moderation, harmful-content filtering, guardrails, responsible AI, data leakage, unsafe tools, excessive agency, cloud governance, red teaming, risk registers, and governance review.
- Out of scope: introducing new retrieval, agent, fine-tuning, serving, observability, multimodal, or voice primitives.
- Handoff: Chapters 6-16 supply systems to test; Chapters 18-20 reuse the same evaluation and governance discipline for multimodal, voice, and domain products.

Mini-project: build an eval-and-security harness for the previous RAG/agent app: run RAGAS/DeepEval plus Giskard or TruLens checks, grade retrieval and citation faithfulness, validate structured outputs, compare prompt/model variants with an A/B test, test prompt-injection attempts and harmful content, verify PII redaction and tool permissions, and block unsafe tool calls.

Production upgrade: write a lightweight responsible-AI risk register with model risks, data and GDPR risks, tool risks, user risks, cloud-governance evidence, mitigation owners, and audit-log requirements.

Evaluation signal: report pass/fail rates for quality tests, retrieval tests, safety tests, and regression tests.

## Chapter 18: Vision-Language and Document Intelligence Applications

Market alignment:
- Target roles: Multimodal AI Engineer, Document AI Engineer, Applied AI Engineer.
- Job keywords: vision-language model, OCR, document understanding, table extraction, visual grounding, multimodal RAG.
- Portfolio artifact: document-intelligence assistant for invoices, forms, or reports.

Students learn how to work with image inputs, document screenshots, OCR alternatives, chart/table understanding, visual question answering, image-grounded extraction, multimodal prompting, multimodal RAG, visual evidence checking, safety checks, and when to combine vision models with classic parsers.

Core skill: build apps that reason over both text and images: invoices, forms, diagrams, product photos, screenshots, compliance documents, and visual reports.

Scope boundary:
- In scope: image inputs, screenshots, OCR alternatives, chart/table understanding, visual question answering, image-grounded extraction, multimodal prompting, visual evidence, and document-intelligence evaluation.
- Out of scope: speech/audio interaction, real-time voice UX, full domain product rollout, general agent architecture, and broad production serving mechanics.
- Handoff: Chapter 19 covers audio and voice; Chapter 20 integrates document intelligence into a domain product when useful.

Mini-project: build a document-intelligence assistant that accepts invoice or form images, extracts structured fields, answers questions, asks follow-up questions, and cites visual evidence.

Production upgrade: compare classic OCR+parser, VLM-only, and hybrid approaches on accuracy, cost, and failure modes.

Evaluation signal: measure field extraction accuracy, citation correctness, and hallucinated visual claims.

## Chapter 19: Speech and Voice Agents

Market alignment:
- Target roles: Voice AI Engineer, Agentic AI Engineer, Applied AI Product Engineer.
- Job keywords: speech-to-text, text-to-speech, real-time audio, interruption handling, latency budget, consent, call workflow.
- Portfolio artifact: voice appointment assistant with tool use and transcript review.

Students learn speech-to-text, text-to-speech, real-time audio streaming, voice activity detection, turn-taking, interruption handling, latency budgets, voice UX, tool use during calls, transcript storage, consent, and failure recovery.

Core skill: design voice agents that feel responsive and useful, not just a chatbot with a microphone.

Scope boundary:
- In scope: speech-to-text, text-to-speech, real-time audio streaming, VAD, turn-taking, interruption handling, latency budgets, voice UX, call-time tool use, transcript storage, consent, and fallback behavior.
- Out of scope: visual document extraction, general RAG construction, multi-agent architecture, full enterprise rollout, and broad governance design.
- Handoff: Chapter 18 covers visual input; Chapter 20 combines voice with retrieval, tools, evaluation, and governance inside a complete domain product.

Mini-project: build a voice appointment assistant that listens to a user, checks availability through a tool, confirms details, handles interruptions, and creates a booking summary.

Production upgrade: add transcript storage, consent messaging, and fallback to text when audio quality is poor.

Evaluation signal: measure turn latency, interruption recovery, task completion, and transcript accuracy.

## Chapter 20: Domain LLM Product and Portfolio Capstone

Market alignment:
- Target roles: Applied AI Engineer, LLM Solutions Engineer, Forward-Deployed AI Engineer, LLM Architect.
- Job keywords: domain discovery, workflow mapping, build-vs-buy, governance, rollout, product metrics, portfolio, impact metrics.
- Portfolio artifact: complete domain copilot plus architecture doc, evaluation report, cost report, and public technical walkthrough.

Students learn how to combine everything into company-grade products: domain discovery, workflow mapping, user roles, data access, RAG vs fine-tuning decisions, tool integration, human review, compliance, governance, evaluation, monitoring, rollout, pricing, maintenance, and product metrics. Product implementation includes React, Next.js, accessible chat UI patterns, dashboard building, streaming responses, error states, feedback capture, and secure frontend/backend boundaries. The chapter also teaches market positioning: how to turn the capstone into a portfolio project with measurable impact and how to align practical work with a relevant Azure, GCP, or AWS cloud certification without treating certification as a substitute for shipped systems.

Core skill: product judgment, meaning the ability to choose the right mix of generation, retrieval, tools, agents, local models, fine-tuning, multimodal input, voice, serving, evaluation, and security for one real business workflow.

Scope boundary:
- In scope: end-to-end domain architecture, workflow mapping, role/data modeling, RAG-vs-fine-tuning decisions, tool integration, React, Next.js, chat UI, dashboard building, human review, compliance, rollout planning, product metrics, cloud certification planning, portfolio packaging, and resume-ready impact framing.
- Out of scope: teaching new primitives that belong in earlier chapters, deep treatment of every domain regulation, and exhaustive enterprise platform design beyond a focused pilot.
- Handoff: this is the integration chapter; it pulls from Chapters 6-19 and converts the work into a coherent capstone and career artifact.

Mini-project: build a complete React/Next.js domain copilot for a real department: support, admissions, HR, legal, finance, healthcare, sales, or operations. It must combine private knowledge, structured data, tools, guardrails, evaluation, chat and dashboard interfaces, optional voice/vision, documented risks, and a small pilot deployment.

Production upgrade: create a portfolio package containing a README, architecture diagram, model decision record, eval report, security notes, deployment notes, cost/latency report, and resume bullets using the formula "Built [system] using [stack] to achieve [measurable outcome] under [constraint]."

Evaluation signal: demonstrate one measurable outcome such as improved retrieval recall, reduced latency, lower cost, fewer hallucinations, faster workflow completion, or higher task success rate.
