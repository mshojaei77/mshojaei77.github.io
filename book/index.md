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
*   **Chapter 3: Building Your First Chatbot**
    *   Connecting to APIs, handling streaming responses, and managing message arrays.
    *   *Enterprise Upgrade*: Wrapping the chatbot in a scalable **FastAPI** backend, handling rate-limiting, and calculating API costs.
*   **💼 Part 1 Capstone Project: "The AI SaaS Backend"**
    *   *Freelance Brief (Upwork)*: "Need a Python backend developer to build a scalable FastAPI service that wraps the OpenAI/Anthropic APIs, streams responses to a Next.js frontend, and implements Redis rate-limiting."

---

## Part 2: The Context Layer (Data Pipelines & RAG)
*Focus: Grounding models in custom data and eliminating hallucinations.*

*   **Chapter 4: Prompting, Context Engineering & Structured Outputs**
    *   Moving beyond basic chatting: Chain-of-Thought and Few-Shot prompting.
    *   Forcing LLMs to return strict JSON using Pydantic schemas.
*   **Chapter 5: Embeddings, Vector Databases, and Semantic Search**
    *   Turning text into numbers (Dense vs. Sparse vectors).
    *   Setting up Vector Databases (Pinecone, Weaviate, Qdrant, FAISS).
*   **Chapter 6: Retrieval-Augmented Generation (RAG)**
    *   Document ingestion, chunking strategies (RecursiveCharacter, Semantic), and context injection.
*   **Chapter 7: Advanced RAG Architecture**
    *   **Hybrid Search**: Combining Vector Search + BM25 using Reciprocal Rank Fusion (RRF).
    *   **Reranking**: Using Cross-Encoders (Cohere, FlashRank) to sort chunks.
    *   **Citations & Freshness**: Forcing the LLM to cite exact source pages and weighting retrieval by document freshness/date.
*   **💼 Part 2 Capstone Project: "Enterprise Document Q&A Bot with Citations"**
    *   *Enterprise Brief (LinkedIn)*: "Build an internal RAG search engine for 10,000 legal PDFs. The bot MUST provide exact page citations, utilize hybrid search, and cannot hallucinate."

---

## Part 3: The Autonomous Layer (Tools & Agents)
*Focus: Giving LLMs the ability to take action, plan, and use software.*

*   **Chapter 8: Function Calling and Tool-Connected LLM Applications**
    *   Giving the LLM "hands." Binding Python functions, SQL queries, and external REST APIs to the model.
*   **Chapter 9: Agentic Systems (Tool, Memory, State)**
    *   Introduction to ReAct (Reason + Act).
    *   Managing graph-based state, short/long-term memory, and checkpointers using **LangGraph**.
*   **Chapter 10: Multi-Agent Systems**
    *   Orchestrating specialized teams of agents (e.g., Planner, Researcher, Coder) using LangGraph, CrewAI, or AutoGen.
*   **💼 Part 3 Capstone Project: "Autonomous E-Commerce Support Agent"**
    *   *Freelance Brief (Arc.dev)*: "Looking for an AI engineer to build an agent that reads emails, queries our SQL database for order status using Function Calling, and uses the Stripe API to issue refunds autonomously."

---

## Part 4: The Customization Layer (Local Models & Fine-Tuning)
*Focus: Taking ownership of models, reducing API costs, and training.*

*   **Chapter 11: Running LLMs Locally on your PC**
    *   Using **llama.cpp, Ollama, LM Studio, and Jan**.
    *   Understanding GGUF formats and hardware constraints.
*   **Chapter 12: Fine-Tuning LLMs**
    *   **Datasets**: Curating synthetic data, formatting (ShareGPT/Alpaca).
    *   **Finetuning Libraries**: Using Hugging Face **trl**, **Unsloth** (for 2x speed), and **Axolotl**.
    *   **Supervised Fine-Tuning (SFT)**: Parameter-Efficient Fine-Tuning using **LoRA and QLoRA**.
    *   **RLHF Fine-Tuning**: Aligning models using **DPO** (Direct Preference Optimization) and **GRPO** (Group Relative Policy Optimization / DeepSeek method).
*   **Chapter 13: Distillation and Continual Pre-training**
    *   Transferring reasoning capabilities from massive models (GPT-4) to small local models.
    *   Continual Pre-training (CPT) to teach an open-weight model a completely new language or domain syntax.
*   **💼 Part 4 Capstone Project: "Domain-Specific Medical JSON Extractor"**
    *   *Freelance Brief (Upwork)*: "GPT-4 is too expensive for our data volume. Fine-tune an open-source 8B model using Unsloth and QLoRA to extract structured patient data into JSON on our local servers."

---

## Part 5: The Infrastructure Layer (Inference, Serving, & Evals)
*Focus: MLOps, CI/CD, and deploying LLMs securely at enterprise scale.*

*   **Chapter 14: Inference Engines for Serving LLMs**
    *   The realities of production: **KV-Cache** management, Continuous Batching, and **Quantization** (INT4/INT8/FP8).
    *   Serving models for high throughput using **vLLM**, **SGLang**, and TensorRT-LLM.
*   **Chapter 15: Evaluation, Benchmark Design, & AI Security**
    *   Why unit testing fails for AI. Building "LLM-as-a-judge" pipelines (DeepEval, Ragas).
    *   Security: PII masking, defending against Prompt Injection, and Red-Teaming (OWASP Top 10 for LLMs).
    *   Observability with LangSmith and MLflow.
*   **💼 Part 5 Capstone Project: "Secure, Cloud-Native LLM Deployment"**
    *   *Enterprise Brief (Accenture)*: "Containerize an open-source LLM with Docker, deploy it via vLLM on an AWS EC2 GPU instance, monitor latency in Grafana, and implement an automated evaluation pipeline."

---

## Part 6: Beyond Text (Multimodal & Domains)
*Focus: The frontier of Generative AI engineering.*

*   **Chapter 16: Vision-Language Models (VLMs)**
    *   Processing images and text together using models like LLaVA, Qwen-VL, or GPT-4o for OCR and spatial reasoning.
*   **Chapter 17: Speech Recognition, Text-to-Speech, & Realtime Voice Agents**
    *   Building ultra-low-latency voice interfaces using Whisper (STT) and ElevenLabs (TTS).
*   **Chapter 18: Domain-Specific LLM Applications**
    *   Architectural patterns for **Enterprise** (internal knowledge bases), **Healthcare** (compliance & entity extraction), **Finance** (automated reporting), and **Customer Support** (fallback routing).
*   **💼 Part 6 Capstone Project: "Multimodal Real Estate Data Extraction Pipeline"**
    *   *Freelance Brief (Jooble)*: "Build a pipeline that takes photos of handwritten property notes and house images, uses a VLM to extract the condition of the house, and auto-populates a SQL database."
