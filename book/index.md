---
title: "LLM Engineering In Action"
nav_order: 1
has_children: true
permalink: /book/
description: "A comprehensive guide to Large Language Models for production."
keywords: "LLM, Large Language Models, Transformer Architecture, Fine-Tuning, RAG, LLMOps, Deep Learning"
author: "Mohammad Shojaei"
date: 2025-01-01
---

# LLM Engineering In Action

## [Introduction: The Modern LLM Engineer](intro.md)

*   **What an LLM Engineer Actually Does**: Bridging the gap between raw foundation models and production-ready enterprise features. Merging Data Engineering, DevOps, and LLMs.
*   **The Job Market & Freelancing**: Breaking down what clients on Upwork and recruiters at Accenture are actually paying for (Hint: shipping reliable products, not just Jupyter Notebooks).
*   **Translating Business to AI**: Managing ROI expectations, talking to stakeholders, and writing technical AI specs.

---

## Part 1: The Engine Room (Foundations & First Steps)

*Focus: Grasping the core mechanics and interacting with the AI ecosystem.*

*   [Chapter 1: LLM Foundations](ch1.md)
    *   How autoregressive Transformers actually work.
    *   Understanding Tokenization (BPE, Subwords).
    *   The mechanics of the Context Window and Attention.
*   [Chapter 2: Choosing Models in a Moving Market](ch2.md)
    *   Navigating Hugging Face and finding the right models.
    *   Integrating commercial APIs (OpenAI, Anthropic, Gemini).
    *   Understanding Benchmarks (MMLU, HumanEval) and Model Comparison.
*   [Chapter 3: Building Your First Chatbot](ch3.md)
    *   Setting up the dev environment and managing API keys securely.
    *   Making your first API call with OpenAI SDKs.
    *   Understanding streaming vs. non-streaming responses.
    *   Managing the message array and context window.
*   **💼 Part 1 Capstone Project: "AI Outline-to-Draft Generator"**
    *   *Context*: Build a minimal AI ghostwriting MVP using OpenAI APIs.
    *   *Skills*: Python scripting, API integration, FastAPI, and file I/O.

---

## Part 2: The Context Layer (Data Pipelines & RAG)

*Focus: Moving from raw prompting to grounded generation with structured outputs, embeddings, retrieval, reranking, and citations.*

*   [Chapter 4: Prompting, Context Engineering & Structured Outputs](ch4.md)
    *   Moving beyond basic chatting with system prompts, few-shot prompting, and repeatable prompt patterns.
    *   Forcing LLMs to return strict JSON using Pydantic schemas and validation layers.
    *   Prompt versioning, context engineering, and production-minded output control.
*   [Chapter 5: Embeddings, Vector Databases, and Semantic Search](ch5.md)
    *   Turning text into searchable representations with dense, sparse, and hybrid retrieval signals.
    *   Setting up vector databases such as Pinecone, Weaviate, Qdrant, and FAISS.
    *   Implementing semantic search with metadata filtering and retrieval diagnostics.
    *   Designing ingestion pipelines: cleaning, chunking, embedding, indexing, and refresh workflows.
    *   Choosing between semantic search, keyword search, and hybrid retrieval for real workloads.
*   [Chapter 6: Retrieval-Augmented Generation (RAG)](ch6.md)
    *   Document ingestion, chunking strategies, and retrieval-aware prompt design.
    *   Context injection, grounding, and citation-friendly answer generation.
    *   Building a practical RAG pipeline from scratch.
    *   Handling pipeline failure modes: stale data, broken chunking, bad retrieval, and missing citations.
    *   Measuring retrieval quality separately from generation quality.
*   [Chapter 7: Advanced RAG Architecture](ch7.md)
    *   Hybrid search with BM25, vector retrieval, and Reciprocal Rank Fusion (RRF).
    *   Reranking with cross-encoders such as Cohere and FlashRank.
    *   Citation tracking, freshness controls, and anti-hallucination guardrails for grounded apps.
    *   Query rewriting, metadata-aware retrieval, and domain-specific ranking strategies.
    *   Cost, latency, and quality trade-offs across retrieval depth, reranking, and model size.
*   💼 **Part 2 Capstone Project: "Enterprise Document Q&A Bot with Citations"**
    *   *Context*: Build an internal RAG system over 10,000 legal PDFs with traceable answers.
    *   *Skills*: Prompt design, structured outputs, hybrid retrieval, reranking, citation extraction, and hallucination prevention.

---

## Part 3: The Autonomous Layer (Tools & Agents)

*Focus: Turning chatbots into workflows that can call tools, manage state, recover from failure, and automate real tasks.*

*   [Chapter 8: Function Calling and Tool-Connected LLM Applications](ch8.md)
    *   Giving the LLM "hands" by binding Python functions, SQL queries, and external APIs to the model.
    *   Designing reliable tool schemas and validating tool responses.
    *   Building multi-step tool-calling workflows with retries and fallbacks.
    *   Tool timeouts, idempotency, rate limiting, and safe recovery from partial failures.
    *   Backend integration patterns with FastAPI, async Python, webhooks, and third-party SDKs.
*   [Chapter 9: Agentic Systems (Tool, Memory, State)](ch9.md)
    *   Introduction to ReAct (Reason + Act) and other production-friendly agent patterns.
    *   Managing graph-based state, session memory, and checkpointers with **LangGraph**.
    *   Building controlled agents with state machines and human-in-the-loop checkpoints.
    *   Memory and context management as first-class engineering problems, not prompt afterthoughts.
    *   Scope boundaries, escalation rules, and recovery logic to prevent orchestration drift.
*   [Chapter 10: Multi-Agent Systems](ch10.md)
    *   Orchestrating specialized agent teams such as Planner, Researcher, and Coder.
    *   Using LangGraph, CrewAI, or AutoGen for coordination and delegation.
    *   Handling communication, task boundaries, and failure modes in multi-agent systems.
    *   Orchestrator-subagent patterns, shared memory design, and coordination overhead.
    *   Reliability testing for multi-agent flows: retries, dead ends, and cross-agent evaluation.
*   **💼 Part 3 Capstone Project: "Autonomous E-Commerce Support Agent"**
    *   *Context*: Build an agent that reads emails, queries SQL, and issues refunds with safe escalation paths.
    *   *Skills*: Tool calling, workflow orchestration, state management, retry logic, and agent design.

---

## Part 4: The Customization Layer (Local Models & Fine-Tuning)

*Focus: Taking ownership of models, reducing serving costs, and customizing behavior with local inference and fine-tuning.*

*   [Chapter 11: Running LLMs Locally on your PC](ch11.md)
    *   Using **llama.cpp, Ollama, LM Studio, and Jan**.
    *   Understanding GGUF formats, quantization choices, and hardware constraints.
    *   Optimizing local inference for throughput, latency, and developer workflow.
    *   GPU and VRAM fundamentals: batch size, context length, and what actually fits in memory.
    *   Prompt caching, local model routing, and when local inference beats API calls economically.
*   [Chapter 12: Fine-Tuning LLMs](ch12.md)
    *   **Datasets**: Curating synthetic data, formatting (ShareGPT/Alpaca).
    *   **Finetuning Libraries**: Using Hugging Face **trl**, **Unsloth** (for 2x speed), and **Axolotl**.
    *   **Supervised Fine-Tuning (SFT)**: Parameter-Efficient Fine-Tuning using **LoRA and QLoRA**.
    *   **RLHF Fine-Tuning**: Aligning models using **DPO** and **GRPO**.
    *   Building high-quality training corpora with Hugging Face Datasets and production data hygiene.
    *   Deciding when prompting, RAG, or fine-tuning is the correct intervention.
*   [Chapter 13: Distillation and Continual Pre-training](ch13.md)
    *   Distilling capabilities from frontier models into smaller local models.
    *   Continual pre-training (CPT) to teach an open-weight model a new language, style, or domain syntax.
    *   Knowledge distillation trade-offs, dataset strategy, and long-term model maintenance.
    *   Distributed training basics: DDP, FSDP, DeepSpeed, and when each matters.
    *   Training efficiency tools such as gradient checkpointing, mixed precision, and curriculum design.
*   **💼 Part 4 Capstone Project: "Domain-Specific Medical JSON Extractor"**
    *   *Context*: Fine-tune an open-source 8B model to extract structured patient data into JSON.
    *   *Skills*: QLoRA fine-tuning, quantized local inference, Unsloth optimization, and domain-specific data processing.

---

## Part 5: The Infrastructure Layer (Inference, Serving, & Evals)

*Focus: Making LLM systems measurable, fast, observable, deployable, and secure under real production constraints.*

*   [Chapter 14: Inference Engines for Serving LLMs](ch14.md)
    *   The realities of production: **KV-cache** management, continuous batching, and **quantization** (INT4/INT8/FP8).
    *   Serving models for high throughput using **vLLM**, **SGLang**, and TensorRT-LLM.
    *   Optimizing inference latency, throughput, streaming behavior, and token cost efficiency.
    *   Load balancing, autoscaling, prompt caching, and backpressure under traffic spikes.
    *   Speculative decoding, batching strategy, and model routing for cost-aware serving.
*   [Chapter 15: Evaluation, Benchmark Design, & AI Security](ch15.md)
    *   Why unit tests are not enough for AI systems. Building eval pipelines with DeepEval, Ragas, and LLM-as-a-judge patterns.
    *   Measuring accuracy, hallucinations, regression risk, and output quality over time.
    *   Security and observability: PII masking, prompt injection defense, red teaming, LangSmith, and MLflow.
    *   Shadow mode, canary releases, A/B testing, and rollback strategy for live AI systems.
    *   Monitoring token usage, drift, refusal rates, latency, and cost as operational health signals.
    *   Compliance, bias audits, data consent, and audit trails for enterprise deployments.
*   **💼 Part 5 Capstone Project: "Secure, Cloud-Native LLM Deployment"**
    *   *Context*: Containerize and deploy an open-source LLM on AWS with monitoring, evals, and defensive controls.
    *   *Skills*: Docker, vLLM, cloud deployment, observability, automated evaluation pipelines, and security hardening.

---

## Part 6: Beyond Text (Multimodal & Domains)

*Focus: Shipping portfolio-grade applied AI systems in high-value domains, including multimodal and real-time interfaces.*

*   [Chapter 16: Vision-Language Models (VLMs)](ch16.md)
    *   Processing images and text together using models like LLaVA, Qwen-VL, or GPT-4o.
    *   OCR, layout understanding, and spatial reasoning with multimodal models.
    *   Building vision-enabled applications for document and workflow automation.
    *   Multimodal grounding trade-offs: when to use OCR pipelines, VLMs, or hybrid systems.
    *   Evaluating visual extraction quality, latency, and hallucination risk in production flows.
*   [Chapter 17: Speech Recognition, Text-to-Speech, & Voice Agents](ch17.md)
    *   Building ultra-low-latency voice interfaces using Whisper (STT) and ElevenLabs (TTS).
    *   Integrating speech capabilities into LLM applications with streaming and interruption handling.
    *   Designing conversational voice agents that feel responsive and resilient.
    *   Streaming APIs, WebSockets, SSE, and realtime backend patterns for voice products.
    *   Queueing, retries, and graceful degradation for long-running or failure-prone audio workflows.
*   [Chapter 18: Domain-Specific LLM Applications](ch18.md)
    *   Architectural patterns for **enterprise** copilots, internal knowledge tools, and automation pipelines.
    *   **Healthcare** (compliance and entity extraction), **finance** (automated reporting), and **customer support** (fallback routing).
    *   Domain constraints, graceful degradation, and portfolio-worthy implementation patterns.
    *   End-to-end AI system design: frontend, backend, model layer, tools, storage, and human review.
    *   Cost-aware architecture decisions, escalation paths, and resilience planning for public-facing AI apps.
*   **💼 Part 6 Capstone Project: "Multimodal Real Estate Data Extraction Pipeline"**
    *   *Context*: Build a pipeline that uses VLMs to extract property data from images and auto-populate a database.
    *   *Skills*: Vision-language models, data extraction, SQL integration, workflow orchestration, and end-to-end product packaging.

---

## After the Book: From Learning to Hiring

*Focus: Turning technical skills into visible proof of work.*

*   **Open Source + Portfolio**:
    *   Publish at least one end-to-end app publicly.
    *   Write short architecture docs that explain model choice, retrieval design, eval strategy, and trade-offs.
    *   Record demo walkthroughs and ship eval reports so employers can see how you think, not just what you built.
*   **Target Roles**:
    *   Applied AI Engineer
    *   GenAI Developer
    *   AI Integration Engineer
    *   LLM Solutions Engineer
