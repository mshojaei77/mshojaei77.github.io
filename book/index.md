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

- The Generation Loop
- Tokens and Subwords
- Decoder-Only Transformers
- Context Windows and Attention
- Logits and Probabilities
- Decoding Parameters
- Generation Failure Modes
- Chat Playgrounds
- Hands-On Exercise

## [Chapter 2: Production Model Selection](ch2.md)

- Model Selection Criteria
- Closed and Open Models
- API, Hosted, and Local Access
- Capability and Task Fit
- Benchmarks and Product Tests
- LLM Leaderboard Pages
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

- Production Prompt Anatomy
- System Prompt Design
- Instruction Hierarchy
- Prompt Security Boundaries
- Prompt Patterns
- Prompt Chaining and Review Loops
- Few-Shot Examples
- Context Assembly
- Token Budgeting
- Prompting for Cost and Latency
- JSON Schema Outputs
- Pydantic Validation Contracts
- Retry and Fallback Logic
- Prompt Versioning
- Prompt Debugging
- Prompt Anti-Patterns
- Production Prompt Checklist
- Hands-On Exercise

## [Chapter 5: Embeddings and Semantic Search](ch5.md)

- Embedding Representations
- Dense and Sparse Signals
- Embedding Model Selection
- Text Preprocessing
- Chunking Strategies
- Metadata Filtering
- Vector Database Design
- Semantic Search Pipelines
- Retrieval Diagnostics
- Index Refresh Workflows
- Hands-On Exercise

## [Chapter 6: Retrieval-Augmented Generation](ch6.md)

- RAG System Architecture
- Document Ingestion Pipelines
- Chunk Retrieval
- Context Injection
- Retrieval-Aware Prompting
- Grounded Answer Generation
- Citation-Friendly Responses
- Missing Evidence Handling
- Retrieval Quality Checks
- RAG Failure Modes
- Hands-On Exercise

## [Chapter 7: Hybrid Retrieval and Reranking](ch7.md)

- Semantic Search Limits
- Keyword Search and BM25
- Hybrid Retrieval Design
- Reciprocal Rank Fusion
- Cross-Encoder Reranking
- Query Rewriting
- Metadata-Aware Ranking
- Freshness Controls
- Retrieval Cost Trade-Offs
- Hallucination-Resistant Retrieval
- Capstone Project

# Part 3: Tools, State, and Agents

## [Chapter 8: Tool-Connected LLM Apps](ch8.md)

- Tool Calling Concepts
- Function Schemas
- Argument Validation
- Python Function Binding
- External API Calls
- Tool Response Handling
- Retry and Timeout Patterns
- Idempotent Tool Design
- FastAPI Tool Integration
- Hands-On Exercise

## [Chapter 9: Stateful Agent Workflows](ch9.md)

- Agent Workflow Patterns
- Reason-Act Loops
- Graph-Based State
- Session Memory
- Long-Term Memory
- Checkpoints and Human Review
- LangGraph Workflows
- Escalation Rules
- Agent Drift Prevention
- Workflow Testing
- Hands-On Exercise

## [Chapter 10: Multi-Agent Systems](ch10.md)

- Multi-Agent Use Cases
- Planner-Worker Patterns
- Researcher-Coder-Reviewer Teams
- Orchestrator Architectures
- Agent Communication
- Shared State Design
- Coordination Failure Modes
- CrewAI, AutoGen, and LangGraph
- Multi-Agent Evaluation
- Capstone Project

# Part 4: Local Models and Customization

## [Chapter 11: Local LLM Inference](ch11.md)

- Local Inference Use Cases
- Open-Weight Model Formats
- GGUF and Quantization
- llama.cpp and Ollama
- LM Studio and Jan
- Hardware and VRAM Limits
- Context Length Trade-Offs
- Throughput and Latency
- Prompt Caching
- Local Model Routing
- Hands-On Exercise

## [Chapter 12: LLM Fine-Tuning](ch12.md)

- Fine-Tuning Decision Criteria
- Prompting, RAG, and Fine-Tuning
- Dataset Design
- Instruction Data Formats
- Supervised Fine-Tuning
- LoRA and QLoRA
- TRL, Unsloth, and Axolotl
- DPO and GRPO
- Fine-Tuning Evaluation
- Hands-On Exercise

## [Chapter 13: Distillation and Pre-Training](ch13.md)

- Knowledge Distillation
- Teacher-Student Design
- Synthetic Training Data
- Continual Pre-Training
- Domain Adaptation
- Distributed Training
- DDP, FSDP, and DeepSpeed
- Training Efficiency
- Model Maintenance
- Capstone Project

# Part 5: Serving, Evaluation, and Reliability

## [Chapter 14: Production LLM Serving](ch14.md)

- Serving Requirements
- KV Cache Management
- Continuous Batching
- Serving Quantization
- vLLM and SGLang
- TensorRT-LLM
- Streaming Architecture
- Load Balancing
- Autoscaling and Backpressure
- Model Routing
- Hands-On Exercise

## [Chapter 15: Evaluation and AI Security](ch15.md)

- AI Evaluation Basics
- Golden Test Sets
- Task-Specific Metrics
- LLM-as-a-Judge
- RAG Evaluation
- Regression Testing
- Observability Traces
- Cost and Latency Monitoring
- Drift and Quality Monitoring
- Prompt Injection Defense
- PII Masking
- Red Teaming
- Canary Releases and Rollbacks
- Capstone Project

# Part 6: Multimodal and Domain Systems

## [Chapter 16: Vision-Language Applications](ch16.md)

- Vision-Language Models
- Image Understanding
- Visual Question Answering
- OCR and Layout Extraction
- Document Automation
- OCR-VLM Hybrid Pipelines
- Multimodal Prompting
- Visual Extraction Evaluation
- Latency and Hallucination Risks
- Hands-On Exercise

## [Chapter 17: Speech and Voice Agents](ch17.md)

- Speech Interface Design
- Speech-to-Text Pipelines
- Whisper Integration
- Text-to-Speech APIs
- Streaming Audio Systems
- WebSocket Backends
- Realtime Turn-Taking
- Interruption Handling
- Voice Agent Reliability
- Audio Failure Recovery
- Hands-On Exercise

## [Chapter 18: Domain LLM Products](ch18.md)

- LLM Product Architecture
- Enterprise Copilots
- Healthcare Applications
- Finance Applications
- Legal Applications
- Customer Support Automation
- Domain Compliance Constraints
- Human Review Workflows
- Cost-Aware Architecture
- Portfolio-Grade Packaging
- Stakeholder Communication
- Capstone Project

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
