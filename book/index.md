---
title: "LLM Engineering In Action"
nav_order: 1
has_children: true
permalink: /book/
description: "A practical guide to building and shipping LLM products."
keywords: "LLM, GenAI, RAG, Agents, Fine-Tuning, Inference, Evaluation"
author: "Mohammad Shojaei"
date: 2025-01-01
---

# LLM Engineering in Action

## Table of Contents

## [Introduction: Modern LLM Engineering](intro.md)

- The LLM Engineer Role
- The Production AI Mindset
- The LLM Application Stack
- Business-to-System Translation
- Reliability, Cost, and Risk
- How to Use This Book

# Part 1: Foundations and First Applications

## [Chapter 1: LLM Text Generation](ch1.md)

- Generation Loop
- Tokenization and Subwords
- Decoder-Only Transformers
- Context Windows and Attention
- Logits and Sampling
- Decoding Controls
- Generation Failure Modes
- Chat Playgrounds
- Hands-On Exercise

## [Chapter 2: Production Model Selection](ch2.md)

- Model Selection Criteria
- Closed and Open Models
- API, Hosted, and Local Access
- Capability and Task Fit
- Benchmarks and Product Tests
- LLM Leaderboards
- Cost, Latency, and Reliability
- Model Decision Records
- Hands-On Exercise

## [Chapter 3: Streaming Chatbot Applications](ch3.md)

- Chat API Structure
- Roles and Message History
- Environment and API Keys
- Streaming Response Handling
- Command-Line Chatbot Design
- Context Growth Management
- Runtime Error Handling
- Capstone Project

# Part 2: Context, Retrieval, and Grounding

## [Chapter 4: Prompting and Structured Outputs](ch4.md)

- System and User Prompts
- Production Prompt Anatomy
- System Prompt Design
- Instruction Hierarchy
- Prompt Security Boundaries
- Prompt Patterns
- Prompt Chaining and Review Loops
- Few-Shot Examples
- Context Assembly
- Prompting for Cost and Latency
- Prompt Versioning
- Prompt Debugging
- Prompt Anti-Patterns
- JSON Schema Outputs
- Pydantic Validation Contracts
- Hands-On Exercise

## [Chapter 5: Embeddings and Semantic Search](ch5.md)

This chapter will turn embeddings from an abstract "meaning as numbers" idea into a practical search layer that every LLM Engineer can use for grounding, memory, recommendation, code search, document search, and retrieval-heavy product features. It will map the embedding model landscape across hosted APIs and open/local models, including OpenAI `text-embedding-3-small` and `text-embedding-3-large`, Voyage AI `voyage-3-large`, Cohere Embed v3, Google Generative AI embeddings, `nomic-embed-text` through Ollama, `mxbai-embed-large`, `UAE-Large-V1`, Stella, EmbeddingGemma, Qwen Embedding, `sentence-transformers`, and `all-MiniLM-L6-v2`, then explain under-the-hood topics such as embedding geometry, cosine similarity versus dot product versus Euclidean distance, normalization to unit length, decoder-only embedding tricks like logit averaging, token limits, dimensionality, pooling, embedding drift, and Matryoshka Representation Learning for truncatable vectors such as 768 to 512 to 256 to 128 dimensions. The chapter will make chunking a first-class engineering skill: fixed-size overlap, recursive and structural splitting, heading-aware sentence splitters in LlamaIndex, semantic chunking, agentic chunking, late chunking, character-based versus token-based splitting, lost-in-the-middle risk, fragmented context, and chunk-size tuning. It will compare vector storage and search options such as ChromaDB, Qdrant, Pinecone, Weaviate, Milvus, pgvector, Elasticsearch, OpenSearch, MongoDB Atlas Vector Search, Supabase vecs, Astra DB, Azure AI Search, FAISS, LEANN, and LanceDB, while covering batching, embedding caches, async generation jobs, GPU versus CPU embedding inference, metadata filters, namespaces, HNSW, IVF, PQ, dimension reduction, refresh workflows, and retrieval evaluation with MTEB, Recall@k, nDCG@k, MRR, latency, cost, freshness, and coverage.

## [Chapter 6: Retrieval-Augmented Generation](ch6.md)

This chapter will show how semantic search becomes a complete Retrieval-Augmented Generation system that can answer with evidence, cite sources, respect private data boundaries, mitigate knowledge cutoff problems, and stay useful when the base model does not know the answer. It will connect the LLM Engineer's daily product work to the full RAG path: ingestion pipeline, chunking, embedding, vector store, retrieval, context assembly, prompt construction, LLM generation, source-linked answers, feedback loops, and evaluation. It will compare common production stacks without locking the book to one vendor, including LangChain with ChromaDB and OpenAI or Gemini, LlamaIndex with Qdrant and Mistral-style self-managed deployments, Haystack pipelines with explicit indexing, retrieval, reranking, summarization, and validation stages, Langflow with Astra DB and Ollama, n8n-RAG with `nomic-embed-text`, and FastAPI services around LangChain, ChromaDB, Gemini, Postgres, Redis, object storage, Streamlit, or Gradio. The chapter will cover advanced ingestion with Docling, Marker, Unstructured.io, LlamaParse, Azure Document Intelligence, OCR, PDF parsing, HTML cleaning, metadata extraction, multilingual RAG, finance/logistics/support domain RAG, 100GB-plus source data architectures, incremental vector updates, change-detection pipelines, streaming context assembly, and source provenance. It will contrast naive single-retrieve single-generate systems with advanced RAG patterns such as query transformation, self-querying, HyDE, step-back prompting, multi-hop retrieval, agentic RAG, GraphRAG, LightRAG, corrective RAG, parent-document retrieval, context compression, and fallback behavior, while repeatedly grounding the production reality that teams often blame the model when the real issue is weak chunking, short-query retrieval signal, stale indexes, context overflow, missing RBAC, or hallucination on unretrieved topics. Security and governance topics include OpenFGA-style document and field-level access control, audit trails for document access, pre-filtering before generation rather than post-filtering, data residency, citation requirements, RAGAS, DeepEval, golden datasets, trace inspection, cost budgets, latency budgets, and production rollbacks.

## [Chapter 7: Hybrid Retrieval and Reranking](ch7.md)

This chapter will explain why production search rarely survives on dense vectors alone and how LLM Engineers combine exact lexical matching, semantic similarity, metadata filters, and reranking to retrieve the right evidence before the model writes a single token. It will show how dense embeddings miss product codes, legal citations, names, acronyms, rare phrases, dates, negation, and exact compliance language, while BM25 and full-text search preserve lexical precision; then it will build hybrid architectures where dense retrieval, sparse retrieval, BM25 or TF-IDF, learned sparse methods such as SPLADE, ColBERT-style late interaction, DRAGON-style retrieval, and metadata-aware ranking feed Reciprocal Rank Fusion, score normalization, candidate expansion, Maximal Marginal Relevance, and business rules. Implementation patterns will include pgvector plus PostgreSQL full-text search with configurable weights, Elasticsearch KNN plus BM25 in one retrieval path, OpenSearch hybrid search including image-and-text queries, ChromaDB plus custom BM25, and Qdrant hybrid search APIs. The reranking section will connect cross-encoder rerankers, BGE Reranker, Cohere Rerank API, mixedbread-ai `mxbai-rerank`, Jina Reranker, Qwen rerankers, LLM-as-a-reranker with single-token relevance scoring, RankGPT, and Anthropic-style contextual retrieval where chunk-specific context is prepended before embedding. For LLM Engineers, this chapter is about measurable retrieval quality rather than hoping the vector database found something: parallel lexical and semantic retrieval branches, RRF fusion, reranker latency budgets, GPU versus CPU reranker serving, cached frequent rerankings, sub-second answer requirements, hard negatives, click logs, query logs, A/B tests against CSAT, nDCG@k, Recall@k, MRR, hit rate, offline evaluation, online evaluation, and the practical insight that small retrieval and ranking changes can improve user satisfaction more than swapping the generation model.

# Part 3: Tools, State, and Agents

## [Chapter 8: Tool-Connected LLM Apps](ch8.md)

This chapter will move from chatbots that only talk to applications that can safely do work: call APIs, query databases, create tickets, send messages, search files, invoke internal services, run CLI commands, and return structured results that the model can reason over. It will cover OpenAI function calling with `tool_choice` and parallel tool calls, Anthropic tool use, Gemini function calling, the OpenAI-compatible tool format adopted by many inference servers, and the distinction between fixed tool-calling workflows and autonomous agent-based tool selection. LLM Engineers will design function schemas with JSON Schema, Pydantic model to JSON Schema conversion, type hints, dataclasses, Zod-style validation, and OpenAPI contracts; bind Python functions with decorators such as LangChain `@tool`, compare `bind_tools` against agent loops, validate arguments before execution, dry-run side-effecting actions, and enforce human confirmation before destructive tools. The chapter will cover REST integrations, SQL query tools, human-in-the-loop approval tools, CLI execution tools, SDK clients, background jobs, webhooks, OAuth scopes, service accounts, least privilege, idempotency keys, retries, exponential backoff, circuit breakers, graceful degradation, structured logging, token-cost tracking per tool call, and compact machine-readable tool responses returned to the conversation loop. It will also show FastAPI tool endpoints, SSE progress streams, MCP servers over FastAPI and SSE, LiteLLM as a unified gateway with tool passthrough across providers, Advisor Chain-style pre-processing for relevant data selection, Go service integration with `langchaingo`, YAML-based context engineering, and monitoring patterns that make tool execution auditable instead of mysterious.

## [Chapter 9: Stateful Agent Workflows](ch9.md)

This chapter will show how to turn one-shot tool use into stateful workflows that can plan, pause, resume, recover, ask for review, stream intermediate progress, and keep consistent state across many model calls. It will connect LLM Engineering to workflow engineering through LangGraph-style graph orchestration, cyclic agent architectures, state machines, directed graphs, nodes for LLM calls and tool executions, conditional edges for routing, `StateGraph` typed schemas with TypedDict or Pydantic, reducers for concurrent updates, `Command` for dynamic routing, `Send` for parallel fan-out, subgraphs for composition, checkpoints for durable state, and `interrupt` gates for human approval. The chapter will compare ReAct loops, Plan-Execute, Observe-Reflect-Act, routing, branching, parallelization, map-reduce, session memory, sliding windows, summarization-based memory, persistent SQLite or Postgres histories, vector-based semantic memory, episodic memory, procedural memory, approval nodes, rollback points, deterministic replay, retries, escalation rules, confidence thresholds, uncertainty detection, and fallback to human operators. The production message will be clear: an agent should not loop forever; it should run inside a constrained graph with explicit nodes for classification, retrieval, validation, tool execution, user clarification, human review, and final response. LLM Engineers will learn drift prevention, state validation between steps, stop conditions, infinite-loop controls, idempotent tools, concurrency locks, `astream_events` token and state streaming, FastAPI SSE integration, unit tests for individual nodes, integration tests for full graphs, deterministic replay from checkpoints, adversarial state tests, timeout tests, traces, observability, and workflow evaluation.

## [Chapter 10: Multi-Agent Systems](ch10.md)

This chapter will explain when multiple agents are useful, when they are expensive theater, and how LLM Engineers can design multi-agent systems that have clear ownership, bounded communication, measurable outcomes, and production failure controls. It will cover use cases such as collaborative code generation, researcher-coder-reviewer teams, customer support triage, logistics planning, enterprise architecture copilots, complex reasoning with specialized perspectives, and long-running workflows where one orchestrator delegates subtasks to workers and aggregates results. The framework landscape will be treated pragmatically: LangGraph for maximum control, stateful graphs, and production workflows; CrewAI for role-based teams and static structures; Microsoft AutoGen for event-driven multi-agent coordination; OpenAI Swarm and Agents SDK for lightweight educational and SDK-driven patterns; Agno for multimodal agents with memory; Hugging Face `smolagents` for code-first agents; PydanticAI for type-safe agent construction; Google ADK, AWS Strands Agents, n8n, Semantic Kernel, LlamaIndex agents, MCP tools, task queues, shared workspaces, and structured message protocols. The chapter will compare planner-worker, supervisor-worker, orchestrator, debate, committee, router, swarm, blackboard, centralized versus decentralized coordination, message-passing versus shared-state, handoff protocols, shared conversation history, concurrent state access, conflict resolution, role prompts, per-agent tool permissions, memory isolation, budget enforcement, trace trees, termination conditions, sandboxed code execution, deadlock, loop control, dropped handoffs, duplicated tasks, conflicting instructions, brittle retrieval, and the practical framework insight that customization, ease of use, and ability to ship rarely point to the same choice. Evaluation topics will include task completion rate, coordination overhead, agent-specific metrics, end-to-end eval suites, trace review, and cost-quality tradeoffs.

# Part 4: Local Models and Customization

## [Chapter 11: Local LLM Inference](ch11.md)

This chapter will teach LLM Engineers how to run open-weight models locally or inside private infrastructure for privacy, air-gapped environments, HIPAA or GDPR-sensitive workloads, cost control, low-latency edge deployment, customer demos, offline capability, and fallback routing. It will explain model formats and runtimes: safetensors as a Hugging Face standard, GGUF as the portable llama.cpp ecosystem format, PyTorch checkpoints, tokenizer compatibility, chat templates, Modelfiles, OpenAI-compatible local endpoints, `llama.cpp`, `llama-server`, Ollama, `llama-cpp-python`, LM Studio, Jan, MLX on Apple Silicon, Transformers, ExLlama-style backends, local web UIs, Harbor, OneInfer, and LiteLLM as a proxy across local and cloud models. Quantization will be treated as an engineering decision, not a checkbox: Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16, GPTQ, AWQ, GGUF, EXL2, FP8, quality-speed tradeoffs, perplexity preservation, CUDA optimization, CPU/GPU hybrid inference, and the practical role of consumer GPUs, Apple unified memory, NVIDIA RTX 3090/4090, A100/H100-class servers, Metal, CUDA, ROCm, Vulkan, and CPU-only inference with system RAM. The chapter will cover VRAM limits, context-window scaling, KV cache memory growth, rope frequency scaling, prompt processing versus token generation speed, batch size effects, streaming, multi-threading, prompt caching through llama.cpp KV slots, provider prompt caching through gateways, cache hit rates, cached-token pricing, tokens per second, time to first token, RAM/VRAM measurement, tool-calling compatibility, structured output reliability, local versus hosted routing, and the practical production split between llama.cpp/Ollama-style local scenarios and vLLM-style serving for higher-throughput hosting.

## [Chapter 12: LLM Fine-Tuning](ch12.md)

This chapter will show when fine-tuning is the right tool and when prompting, retrieval, tool design, or product constraints should be fixed first. It will frame fine-tuning as a product decision for domain vocabulary, consistent output formatting, task-specific behavior, smaller-model optimization, predictable outcomes, refusal behavior, and style alignment, while RAG remains the better fit for fresh facts and real-time data access. LLM Engineers will learn dataset design with system/user/assistant triples, ShareGPT, Alpaca, ChatML, Llama 3 chat templates, data quality over quantity, synthetic data generation from larger models, train-validation-test splits, contamination checks, deduplication, refusal and jailbreak datasets, domain-specific corpora, and held-out evaluation. The chapter will cover supervised fine-tuning, full-parameter versus parameter-efficient methods, LoRA rank `r`, alpha scaling, rank 16 as a common default, QLoRA with NF4 quantization, adapter merging and swapping, PEFT, `bitsandbytes`, `accelerate`, learning rates, epochs, loss curves, overfitting detection, reward modeling, DPO, ORPO, KTO, GRPO, constitutional AI-style alignment, and side-by-side comparisons with base models. The dominant practical stack will include Hugging Face Transformers, Datasets, TRL, PEFT, Unsloth for fast local fine-tuning and memory reduction, Axolotl with YAML orchestration and multimodal support, LLaMA-Factory with a broad model interface and web UI, torchtune, LitGPT, MLX on Apple Silicon, Weights and Biases, Trackio-style experiment logs, model cards, and Hub publishing. Acceleration topics will include Liger Kernel, Flash Attention 2, gradient checkpointing, mixed precision bf16/fp16, DeepSpeed ZeRO with QLoRA, and deployment handoff; evaluation topics will include task-specific metrics, held-out loss, insurance or legal domain adaptation patterns, structured extraction reliability, ablations, regression tests, and responsible handling of alignment modification or uncensoring workflows.

## [Chapter 13: Distillation and Pre-Training](ch13.md)

This chapter will give LLM Engineers the mental model for building smaller, cheaper, domain-specialized models through distillation and continued pre-training, even if they are not training frontier models from scratch. It will cover teacher-student pipelines where a larger model such as GPT-4o or Claude-class systems generates demonstrations for smaller models such as Mistral, Qwen, Llama, or other 1B to 8B candidates; task-based distillation for legal summarization, chess Q&A, structured extraction, and code workflows; logit distillation, hidden-state distillation, soft labels, preference distillation, multi-teacher ensembles, self-instruct pipelines, synthetic instruction data, quality filtering, diversity checks, curriculum design, thought-template distillation, self-correction traces, and safe handling of chain-of-thought-derived training signals. The chapter will explain continued pre-training and domain-adaptive pre-training on legal, medical, financial, cybersecurity, or enterprise corpora, including general-to-domain data mixing ratios, vocabulary expansion, tokenizer changes, specialized terminology, catastrophic forgetting mitigation, and the sequence from pre-training to SFT to alignment methods such as RLHF or GRPO. It will also teach the infrastructure vocabulary behind serious training: PyTorch, Transformers, Datasets, nanoGPT-style minimal pre-training, LitGPT, Megatron-style parallelism, DDP gradient synchronization, FSDP sharding of parameters and optimizer states, DeepSpeed ZeRO stages 1 through 3, ZeRO-3 plus QLoRA, mixed precision, FP8 training with TransformerEngine, Flash Attention, Liger Kernel, gradient accumulation, activation checkpointing, sequence packing, checkpoint sharding, data mixture weights, deduplication, eval harnesses, experiment tracking, and model maintenance. For LLM Engineers, this chapter is about deciding when a distilled model can replace a larger hosted model, when continued pre-training beats SFT, and how to version checkpoints, manage adapter weights, schedule continual fine-tuning, test regressions, respect data licensing, avoid benchmark gaming, and plan compute budgets.

# Part 5: Serving, Evaluation, and Reliability

## [Chapter 14: Production LLM Serving](ch14.md)

This chapter will turn model inference into an operating service with throughput, latency, reliability, observability, and cost controls. It will explain serving requirements such as requests per second, time to first token, inter-token latency, high concurrency, cost efficiency, and OpenAI-compatible APIs for drop-in replacement, then show how LLM Engineers serve open and fine-tuned models behind streaming endpoints, internal gateways, and product backends using vLLM, SGLang, Hugging Face Text Generation Inference, TensorRT-LLM, Triton Inference Server, KServe, Kubernetes, Docker, GPU nodes, and managed inference endpoints. The mechanics section will cover KV cache as the major transformer inference memory bottleneck, PagedAttention-style fixed-block allocation, continuous batching and in-flight batching where requests join and leave at each decoding step, static versus dynamic versus continuous batching, prefill versus decode, chunked prefill, prefill-decode disaggregation, prefix caching, speculative decoding, LoRA adapter serving, model warmup, rate limits, request queues, backpressure, health checks, load balancing, canary deployments, fallback models, and model-aware routing. It will compare vLLM for PagedAttention and high-throughput OpenAI-compatible serving, SGLang for RadixAttention-style prefix-aware scheduling and structured generation, TensorRT-LLM for NVIDIA-optimized inference with FP8/INT4/INT8 and Triton integration, plus LiteLLM as a universal gateway for 120-plus providers, failover, load balancing, semantic routing, and cost tracking. Quantization and hardware topics will include FP8 on H100/H200-class deployments, AWQ and GPTQ for serving, GGUF for CPU/GPU hybrid paths, INT4 for fitting large models on constrained hardware, GPU utilization, multi-GPU tensor parallelism, capacity planning, cache hit rates, LMCache-style distributed KV cache offload, consistent hashing for prefix-aware routing, horizontal autoscaling by GPU utilization and queue depth, SSE token streaming, FastAPI `StreamingResponse`, WebSocket bidirectional streaming for voice, Prometheus, Grafana, structured logs, traces, error budgets, SLOs, multi-tenant isolation, security boundaries, cost dashboards, and rollback playbooks.

## [Chapter 15: Evaluation and AI Security](ch15.md)

This chapter will show how LLM Engineers prove that an AI system works, keeps working after changes, and fails safely under attack or messy real-world inputs. It will connect offline and online evaluation to golden test sets, domain-specific benchmarks, adversarial cases, human evaluation, automated evaluation, exact match, F1, BLEU, ROUGE, BERTScore, pass@k, task-specific scorecards, rubric scoring, pairwise comparison, LLM-as-a-judge, G-Eval-style chain-of-thought scoring, and judge-bias problems such as position bias, verbosity bias, and self-enhancement bias. RAG evaluation will cover RAGAS metrics such as Faithfulness, Answer Relevancy, Context Relevancy, Context Recall, and Context Precision; DeepEval plug-and-play metrics for RAG, chatbots, and agents; TruLens feedback functions and the RAG triad; Arize Phoenix with OpenTelemetry-native observability and eval; MLflow Evaluate; Weights and Biases Prompts; promptfoo; OpenAI eval-style harnesses; lm-evaluation-harness; custom metrics; CI/CD integration; and production monitoring decorators such as `@observe`. Observability and reliability topics will include OpenTelemetry traces, spans, Langfuse self-hosted OTLP-native tracing, Grafana dashboards, StepsTrack-style pipeline visualization, token counting, cost attribution by user or feature, p50/p95/p99 latency, cache hit rates, embedding drift, semantic firewalls, production traffic sampling, prompt version tests, retrieval-quality drift, canary releases, shadow deployments, A/B experiment proxies, gradual traffic shifting, automated rollback, and release gates. The security half will cover prompt injection, indirect prompt injection in retrieved documents and tool outputs, instruction hierarchy, input sanitization, second-LLM security audits, LLMSEC-style malicious prompt classifiers, UPSS-style governance, OWASP LLM risks, PII masking with Microsoft Presidio, regex and LLM-based PII detection, data residency, jailbreaks, many-shot jailbreaking, data exfiltration, insecure output handling, tool abuse, excessive agency, vector-store poisoning, insecure plugins, secrets in context, tenant isolation, audit trails, human review, red teaming with HarmBench-style automation, Cybench-style cybersecurity tasks, AIRTBench-style AI red teaming, and incident response.

# Part 6: Multimodal and Domain Systems

## [Chapter 16: Vision-Language Applications](ch16.md)

This chapter will extend LLM Engineering beyond text into systems that read images, screenshots, forms, charts, receipts, PDFs, scanned documents, tables, diagrams, product photos, and multimodal streams. It will map the VLM landscape across GPT-4V and GPT-4o-style multimodal models, Claude vision models, Gemini Pro Vision, Qwen2-VL and Qwen2.5-VL document-optimized models, LLaVA, SmolVLM, Pixtral, Florence-2, PaliGemma, InternVL2, Gemma multimodal variants, GLM-OCR-style systems, any-to-any multimodal models, mixture-of-experts scaling, InfiniteVL-style long multimodal streams, and progressive multimodal training. The chapter will show how image understanding becomes product software: visual question answering, image captioning, zero-shot classification, RefCOCO-style grounding, MathVision-style reasoning, screenshot analysis, chart and diagram understanding, visual instruction tuning, spatial reasoning, interleaved image-plus-text prompts, multimodal embeddings, visual document retrieval, ColPali-style retrieval, and retrieval over document pages. Document automation will cover OCR and layout extraction with dots.ocr, OCRVerse, DocTR, Nougat, Tesseract, PaddleOCR, OpenCV, Docling, Marker, multi-engine OCR with VLM fallback, invoice processing, form field extraction, table extraction, reading-order preservation, bounding boxes, crop-and-tile strategies for high-resolution pages, and JSON extraction with schema validation. The chapter will compare OCR-only, VLM-only, and OCR-VLM hybrid pipelines under document degradation, low resolution, rotation, noise, and missing layout cues, using Levenshtein distance, nDCG@k, Recall@k, EM/F1, visual extraction accuracy, and human review. For LLM Engineers, vision-language applications matter because many enterprise workflows are trapped in PDFs, scans, dashboards, legal and financial tables, retail images, and medical images, so the chapter will emphasize latency, cost, object hallucination, OCR hallucination, structured outputs, uncertainty handling, and domain examples such as medical multimodal LLMs, retail visual search, and legal or financial document retrieval.

## [Chapter 17: Speech and Voice Agents](ch17.md)

This chapter will build the LLM Engineer's map of voice systems: speech input, real-time turn taking, language model reasoning, tool execution, and speech output under strict latency and reliability constraints. It will compare cascading voice architectures, where STT feeds an LLM and TTS reads the response, against end-to-end speech models, voice-plus-text fallback, push-to-talk, always-listening mode, persona-driven voice agents, and multimodal voice interfaces. Speech-to-text topics will include OpenAI Whisper `large-v3`, faster-whisper with CTranslate2 optimization, WhisperX for word-level timestamps and diarization, WhisperKit, whisper-jax, wav2vec2, Deepgram, AssemblyAI, Silero VAD, pyannote-style diarization, voice activity detection, endpointing, punctuation restoration, STT confidence thresholds, accent robustness, ambient-noise handling, streaming transcripts, and GPU acceleration for real-time transcription. Text-to-speech topics will include ElevenLabs, Cartesia, Edge-TTS, Google WaveNet or Cloud TTS, Amazon Polly, Kokoro, Coqui XTTS, AivisSpeech, Orpheus TTS, Chatterbox, OmniVoice, multilingual TTS, voice cloning, low-latency streaming TTS, chunked audio generation, and graceful degradation when TTS fails. Implementation will cover FastAPI WebSocket endpoints, bidirectional audio streams, SSE token streams into TTS, WebRTC, LiveKit full-stack voice agents, Pipecat, Cloudflare Agents SDK WebSockets, audio buffers, sample rates, codecs, jitter, `ffmpeg`, queues, reconnection logic, silence detection, dropped WebSocket recovery, dual-buffer turn detection, barge-in, interruption handling, context preservation, telephony integrations with Twilio, Vonage, and Telnyx, bidirectional RTP, OpenAI Realtime API-style integrations, and production stack examples such as Whisper plus Silero VAD plus Llama or Agno, faster-whisper plus LangChain plus GPT-4o, and Whisper plus Hugging Face LLM plus Edge-TTS. For LLM Engineers, voice agents are demanding because small delays, broken turn boundaries, or unreliable tool calls are immediately visible to users, so the chapter will emphasize latency budgets, observability, fallback to text input, retry behavior, and conversation recovery.

## [Chapter 18: Domain LLM Products](ch18.md)

This chapter will synthesize the book into domain-grade LLM products that solve a real business workflow rather than a generic demo. It will show how LLM Engineers translate healthcare, finance, legal, customer support, education, cybersecurity, analytics, sales, DevOps, and enterprise knowledge problems into scoped product surfaces that combine RAG, fine-tuning, custom tokenizers or domain vocabulary handling, structured extraction, tool use, workflow state, evaluation, security, monitoring, human review, and stakeholder reporting. The chapter will cover enterprise copilots built with patterns such as Microsoft Copilot Studio, Azure AI Foundry, Semantic Kernel, LangGraph, Flyte, AWS Bedrock, Claude-style model backends, DevOps copilots, and production infrastructure thinking; healthcare applications such as HIPAA-compliant deployment, VPC isolation, clinical note summarization, medical coding automation, patient-facing chatbots, private hosting, PII redaction with Presidio, RBAC, safety guardrails, and medical multimodal models; finance applications such as SEC/SOX-aware document analysis, PCI-DSS data handling, KPI-level financial summarization, fraud detection copilots, investment research assistants, and regulatory filing automation; legal applications such as contract analysis, clause extraction, case-law retrieval, e-discovery, legal summarization, LoRA-based document models, specialized embeddings, and reranking for legal nuance; and customer support systems with strict sequencing from indexing to retrieval to reranking to summarization to validation, ticket routing, triage, deflection-rate optimization, CSAT-driven retrieval tuning, and human escalation with full conversation context. Compliance and architecture topics will include pre-filtering access before generation, not relying on the model to remember what it should not disclose, document and field-level RBAC, OpenFGA-style access checks, audit trails, data residency, egress controls, HIPAA, SEC, SOX, GDPR, PCI-DSS, SOC 2, confidence-based escalation, review queues, annotation interfaces, model cascading, token budgets per domain, prompt caching for static compliance instructions, cache hit-rate optimization, cost dashboards, architecture decision records, evaluation reports with domain-specific metrics, public demo packaging, technical walkthroughs, compliance-officer collaboration, and the career-level insight that LLM plus domain experts often beats generic AI talent.

# After the Book: Career and Portfolio

## Portfolio Development

- Project Selection
- Architecture Documents
- Public Demo Applications
- Technical Walkthroughs
- Evaluation Reports
- Trade-Off Explanations

## LLM Engineering Careers

- Applied AI Engineer
- GenAI Developer
- AI Integration Engineer
- LLM Solutions Engineer
- Freelance LLM Engineer
- Enterprise AI Consultant
