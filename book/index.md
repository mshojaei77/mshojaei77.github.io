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

*Focus: Grounding models in custom data and eliminating hallucinations.*

*   [Chapter 4: Prompting, Context Engineering & Structured Outputs](ch4.md)
    *   Moving beyond basic chatting: Chain-of-Thought and Few-Shot prompting.
    *   Forcing LLMs to return strict JSON using Pydantic schemas.
    *   Context engineering strategies for production applications.
*   [Chapter 5: Embeddings, Vector Databases, and Semantic Search](ch5.md)
    *   Turning text into numbers (Dense vs. Sparse vectors).
    *   Setting up Vector Databases (Pinecone, Weaviate, Qdrant, FAISS).
    *   Implementing semantic search for document retrieval.
*   [Chapter 6: Retrieval-Augmented Generation (RAG)](ch6.md)
    *   Document ingestion and chunking strategies (RecursiveCharacter, Semantic).
    *   Context injection and prompt engineering for RAG.
    *   Building a basic RAG pipeline from scratch.
*   [Chapter 7: Advanced RAG Architecture](ch7.md)
    *   Hybrid Search: Combining Vector Search + BM25 using Reciprocal Rank Fusion (RRF).
    *   Reranking: Using Cross-Encoders (Cohere, FlashRank) to sort chunks.
    *   Citations & Freshness: Forcing the LLM to cite exact source pages.
*   [💼 Part 2 Capstone Project: "Enterprise Document Q&A Bot with Citations"**p2.md)
    *   *Context*: Build an internal RAG search engine for 10,000 legal PDFs.
    *   *Skills*: Hybrid search, reranking, citation extraction, and hallucination prevention.

---

## Part 3: The Autonomous Layer (Tools & Agents)

*Focus: Giving LLMs the ability to take action, plan, and use software.*

*   [Chapter 8: Function Calling and Tool-Connected LLM Applications](ch8.md)
    *   Giving the LLM "hands." Binding Python functions, SQL queries, and external REST APIs to the model.
    *   Designing tool schemas and handling tool responses.
    *   Building multi-step tool-calling workflows.
*   [Chapter 9: Agentic Systems (Tool, Memory, State)](ch9.md)
    *   Introduction to ReAct (Reason + Act).
    *   Managing graph-based state, short/long-term memory, and checkpointers using **LangGraph**.
    *   Building autonomous agents with planning capabilities.
*   [Chapter 10: Multi-Agent Systems](ch10.md)
    *   Orchestrating specialized teams of agents (e.g., Planner, Researcher, Coder).
    *   Using LangGraph, CrewAI, or AutoGen for multi-agent coordination.
    *   Handling agent communication and task delegation.
*   **💼 Part 3 Capstone Project: "Autonomous E-Commerce Support Agent"**
    *   *Context*: Build an agent that reads emails, queries SQL databases, and issues refunds autonomously.
    *   *Skills*: Function calling, database integration, API orchestration, and agent design.

---

## Part 4: The Customization Layer (Local Models & Fine-Tuning)

*Focus: Taking ownership of models, reducing API costs, and training.*

*   [Chapter 11: Running LLMs Locally on your PC](ch11.md)
    *   Using **llama.cpp, Ollama, LM Studio, and Jan**.
    *   Understanding GGUF formats and hardware constraints.
    *   Optimizing local inference for performance.
*   [Chapter 12: Fine-Tuning LLMs](ch12.md)
    *   **Datasets**: Curating synthetic data, formatting (ShareGPT/Alpaca).
    *   **Finetuning Libraries**: Using Hugging Face **trl**, **Unsloth** (for 2x speed), and **Axolotl**.
    *   **Supervised Fine-Tuning (SFT)**: Parameter-Efficient Fine-Tuning using **LoRA and QLoRA**.
    *   **RLHF Fine-Tuning**: Aligning models using **DPO** and **GRPO**.
*   [Chapter 13: Distillation and Continual Pre-training](ch13.md)
    *   Transferring reasoning capabilities from massive models (GPT-4) to small local models.
    *   Continual Pre-training (CPT) to teach an open-weight model a completely new language or domain syntax.
    *   Knowledge distillation techniques and best practices.
*   **💼 Part 4 Capstone Project: "Domain-Specific Medical JSON Extractor"**
    *   *Context*: Fine-tune an open-source 8B model to extract structured patient data into JSON.
    *   *Skills*: QLoRA fine-tuning, Unsloth optimization, and domain-specific data processing.

---

## Part 5: The Infrastructure Layer (Inference, Serving, & Evals)

*Focus: MLOps, CI/CD, and deploying LLMs securely at enterprise scale.*

*   [Chapter 14: Inference Engines for Serving LLMs](ch14.md)
    *   The realities of production: **KV-Cache** management, Continuous Batching, and **Quantization** (INT4/INT8/FP8).
    *   Serving models for high throughput using **vLLM**, **SGLang**, and TensorRT-LLM.
    *   Optimizing inference latency and throughput.
*   [Chapter 15: Evaluation, Benchmark Design, & AI Security](ch15.md)
    *   Why unit testing fails for AI. Building "LLM-as-a-judge" pipelines (DeepEval, Ragas).
    *   Security: PII masking, defending against Prompt Injection, and Red-Teaming (OWASP Top 10 for LLMs).
    *   Observability with LangSmith and MLflow.
*   **💼 Part 5 Capstone Project: "Secure, Cloud-Native LLM Deployment"**
    *   *Context*: Containerize and deploy an open-source LLM on AWS with monitoring and evaluation.
    *   *Skills*: Docker, vLLM, cloud deployment, Grafana monitoring, and automated evaluation pipelines.

---

## Part 6: Beyond Text (Multimodal & Domains)

*Focus: The frontier of Generative AI engineering.*

*   [Chapter 16: Vision-Language Models (VLMs)](ch16.md)
    *   Processing images and text together using models like LLaVA, Qwen-VL, or GPT-4o.
    *   OCR and spatial reasoning with multimodal models.
    *   Building vision-enabled applications for document understanding.
*   [Chapter 17: Speech Recognition, Text-to-Speech, & Realtime Voice Agents](ch17.md)
    *   Building ultra-low-latency voice interfaces using Whisper (STT) and ElevenLabs (TTS).
    *   Integrating speech capabilities into LLM applications.
    *   Designing conversational voice agents.
*   [Chapter 18: Domain-Specific LLM Applications](ch18.md)
    *   Architectural patterns for **Enterprise** (internal knowledge bases).
    *   **Healthcare** (compliance & entity extraction), **Finance** (automated reporting), and **Customer Support** (fallback routing).
    *   Industry-specific considerations and best practices.
*   **💼 Part 6 Capstone Project: "Multimodal Real Estate Data Extraction Pipeline"**
    *   *Context*: Build a pipeline that uses VLMs to extract property data from images and auto-populate a database.
    *   *Skills*: Vision-language models, data extraction, SQL integration, and pipeline orchestration.
