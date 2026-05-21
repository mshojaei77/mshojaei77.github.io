---
title: "LLM Engineering In Action"
nav_order: 1
has_children: true
permalink: /book/
description: "A practical, market-aligned guide to building, evaluating, securing, and operating LLM products."
keywords: "LLM, GenAI, RAG, Agents, Fine-Tuning, LLMOps, Evaluation, AI Security, Inference"
author: "Mohammad Shojaei"
date: 2026-05-21
---

# LLM Engineering in Action

This book teaches the production skills that appear repeatedly in 2026 LLM engineering job postings and strong real-world resumes: RAG, agentic workflows, fine-tuning, LLMOps, evaluation, security, cloud deployment, and portfolio-ready product work.

## Market-Aligned Outcomes

By the end of the book, readers should be able to build and explain:

- A document-grounded RAG assistant with citations and retrieval tests
- A hybrid search system with reranking and measurable retrieval quality
- A permission-aware enterprise assistant connected to documents and SQL data
- A tool-connected LLM app with approval gates and audit logs
- A stateful agent workflow with checkpoints, retries, and human review
- A fine-tuned open-source model with before/after evaluation
- A production LLM API with Docker, CI/CD, monitoring, and cost controls
- An evaluation and security harness for RAG and agent systems
- A portfolio capstone with architecture notes, metrics, and trade-off analysis

## Table of Contents

### [Introduction: Modern LLM Engineering](intro.md)

- The LLM Engineer Role
- The Production AI Mindset
- The LLM Application Stack
- Business-to-System Translation
- Reliability, Cost, and Risk
- How to Use This Book

## Part 1: Foundations and First Applications

### [Chapter 1: LLM Text Generation](ch1.md)

- Generation Loop
- Tokenization and Subwords
- Decoder-Only Transformers
- Context Windows and Attention
- Logits and Sampling
- Decoding Controls
- Generation Failure Scenarios
- Chat Playgrounds
- Hands-On Exercise

### [Chapter 2: Production Model Selection](ch2.md)

- Model Selection Criteria
- Closed and Open Models
- API, Hosted, and Local Access
- Capability and Task Fit
- Benchmarks and Product Tests
- LLM Leaderboards
- Cost, Latency, and Reliability
- Model Decision Records
- Hands-On Exercise

### [Chapter 3: Streaming Chatbot Applications](ch3.md)

- Chat API Structure
- Roles and Message History
- Environment and API Keys
- Streaming Response Handling
- Command-Line Chatbot Design
- Context Growth Management
- Runtime Error Handling
- Part 1 Capstone Project

## Part 2: Context, Retrieval, and Enterprise Grounding

### [Chapter 4: Prompting and Structured Outputs](ch4.md)

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
- Prompt Versioning and Debugging
- Prompt Anti-Patterns
- JSON Schema Outputs
- Pydantic Validation Contracts
- Hands-On Exercise

### [Chapter 5: Embeddings and Semantic Search](ch5.md)

- What is an Embedding?
- Practical Chunking Strategies
- Choosing Embedding Models
- Embedding Generation Best Practices
- Vector Databases and Storage Options
- Basic Semantic Search Implementation
- Evaluation Metrics for Retrieval
- Common Pitfalls and Failure Scenarios
- Hands-On Exercise

### [Chapter 6: Retrieval-Augmented Generation](ch6.md)

- Search Is Not an Answer
- The Basic RAG Loop
- Context Construction and Token Budgeting
- Grounded Prompting and "I Don't Know"
- Source Citations as a Product Contract
- FastAPI and Streamlit Interface
- Debugging Retrieval and Context
- Minimal Smoke Tests
- Hands-On Exercise: PDF Q&A Bot with Citations

### [Chapter 7: Hybrid Retrieval, Reranking, and RAG Evaluation](ch7.md)

- Why Basic Vector Search Fails
- BM25 and Sparse Retrieval
- Metadata Filters and Exact-Match Needs
- Hybrid Dense+Sparse Search
- Query Rewriting
- Reciprocal Rank Fusion
- Cross-Encoder Reranking
- Golden Query Sets
- Retrieval Metrics: Recall@k, Precision@k, MRR, NDCG
- RAGAS, DeepEval, and Custom RAG Evaluation
- Hands-On Exercise: Support Search System with Measured Retrieval Lift

### [Chapter 8: Enterprise Data Integration and Permission-Aware RAG](ch8.md)

- Document Ingestion Pipelines
- Parser Selection and Metadata Contracts
- Incremental Indexing and Data Freshness
- SQL and Postgres Grounding
- Permission-Aware Retrieval
- Tenant Isolation and Row-Level Security
- PII Handling and Audit Logs
- Knowledge Graphs and Structured Context
- Access-Control Test Suites
- Hands-On Exercise: Internal Assistant with Documents, SQL Data, and Role Filters

## Part 3: Tools, State, and Agents

### [Chapter 9: Tool-Connected LLM Apps and MCP-Style Interfaces](ch9.md)

- Function Calling and Tool Schemas
- Argument Validation
- Tool-Result Injection
- Read Tools vs Write Tools
- API Safety and Idempotency
- Human Approval for Risky Actions
- MCP-Style Tool Boundaries
- Tool-Call Logging and Audit Trails
- Hands-On Exercise: Customer-Support Copilot with Docs, Order Lookup, and Approval Gates

### [Chapter 10: Stateful Agent Workflows](ch10.md)

- State Machines and Workflow Graphs
- Planning Loops
- Conversation State and User Preferences
- Checkpoints and Resumable Execution
- Retries, Interruptions, and Failure Recovery
- Human-in-the-Loop Review
- Basic Agent Observability
- Hands-On Exercise: Resumable Research Assistant Workflow

### [Chapter 11: Multi-Agent Systems and Agent Evaluation](ch11.md)

- When Multi-Agent Systems Help
- Supervisor and Router Patterns
- Specialist Agents and Handoffs
- Shared State
- Parallel Research
- Conflict Resolution
- Cost Explosion and Latency Risks
- Agent Tracing and Evaluation
- Hands-On Exercise: Multi-Agent Proposal Builder

## Part 4: Open Models and Customization

### [Chapter 12: Local LLM Inference](ch12.md)

- Open-Weight Model Families
- Licensing and Deployment Constraints
- Tokenizers and Context Length
- VRAM/RAM Limits
- Quantization and GGUF
- Ollama, llama.cpp, and Hugging Face Loading
- CPU vs GPU Trade-Offs
- Privacy, Cost, and Offline Use Cases
- Hands-On Exercise: Local Private Document Assistant

### [Chapter 13: LLM Fine-Tuning with LoRA, QLoRA, and PEFT](ch13.md)

- When Fine-Tuning Helps
- When RAG or Prompting Is Better
- Dataset Formatting and Quality
- Train/Validation Splits
- Supervised Fine-Tuning
- LoRA, QLoRA, and PEFT
- Hyperparameters and Overfitting
- Before/After Evaluation
- Model Cards and Dataset Cards
- Hands-On Exercise: Fine-Tuned Support-Response Model

### [Chapter 14: Preference Tuning, Distillation, and Training Judgment](ch14.md)

- DPO, RLHF, and RLAIF Concepts
- Synthetic Data Generation
- Teacher-Student Distillation
- Rationale Transfer
- Continued Pre-Training Awareness
- Tokenizer and Data Quality Decisions
- Compute Budgeting
- Build vs Buy vs Fine-Tune
- Hands-On Exercise: Distilled Classifier or Mini Assistant

## Part 5: Production, Reliability, and Governance

### [Chapter 15: Production LLM Serving and LLMOps](ch15.md)

- API Design and Authentication
- Queues, Rate Limits, and Caching
- Streaming at Scale
- Docker, CI/CD, Secrets, and Rollbacks
- Cloud Deployment Basics
- vLLM, TGI, TensorRT-LLM, and Model Servers
- Continuous Batching and KV Cache
- Autoscaling and Health Checks
- Hands-On Exercise: Productionized RAG/Agent API

### [Chapter 16: Observability, Cost, and Reliability Engineering](ch16.md)

- Structured Logs and Traces
- Prompt and Model Versioning
- Latency Metrics and p95 Debugging
- Token Usage and Cost Tracking
- Model Routing and Fallbacks
- Cache Hit Rate and Queue Depth
- Budget Alerts and Incident Review
- Load Tests and Reliability Reports
- Hands-On Exercise: LLMOps Dashboard

### [Chapter 17: Evaluation, AI Security, and Governance](ch17.md)

- Golden Datasets and Regression Tests
- LLM-as-Judge and Human Review
- RAG Quality and Citation Faithfulness
- Hallucination Checks
- Prompt Injection
- Data Leakage
- Insecure Tool Use and Excessive Agency
- Guardrails, Red Teaming, and Incident Response
- Governance and AI Risk Registers
- Hands-On Exercise: Evaluation and Security Harness

## Part 6: Multimodal, Voice, and Domain Products

### [Chapter 18: Vision-Language and Document Intelligence Applications](ch18.md)

- Image Inputs and Document Screenshots
- OCR vs Vision-Language Models
- Chart and Table Understanding
- Visual Question Answering
- Image-Grounded Extraction
- Multimodal Prompting and RAG
- Visual Evidence Checking
- Hands-On Exercise: Invoice or Form Intelligence Assistant

### [Chapter 19: Speech and Voice Agents](ch19.md)

- Speech-to-Text and Text-to-Speech
- Real-Time Audio Streaming
- Voice Activity Detection
- Turn-Taking and Interruptions
- Latency Budgets
- Voice UX and Failure Recovery
- Tool Use During Calls
- Transcript Storage and Consent
- Hands-On Exercise: Voice Appointment Assistant

### [Chapter 20: Domain LLM Product and Portfolio Capstone](ch20.md)

- Domain Discovery and Workflow Mapping
- User Roles and Data Access
- RAG vs Fine-Tuning Decisions
- Tool Integration and Human Review
- Compliance, Governance, and Rollout
- Product Metrics and Maintenance
- Architecture Documents
- Evaluation Reports
- Cost and Latency Reports
- Resume Bullets and Technical Walkthroughs
- Capstone Project: Complete Domain Copilot

## After the Book: Career and Portfolio

### Portfolio Development

- Project Selection
- Architecture Documents
- Public Demo Applications
- Technical Walkthroughs
- Evaluation Reports
- Security Notes
- Cost and Latency Reports
- Trade-Off Explanations
- Resume Bullets with Measurable Outcomes

### LLM Engineering Careers

- RAG and Knowledge Systems Engineer
- Agentic AI Engineer
- LLM Fine-Tuning Specialist
- LLMOps or AI Platform Engineer
- Applied AI / Full-Stack LLM Engineer
- LLM Evaluation Engineer
- LLM Solutions Engineer
- Forward-Deployed AI Engineer
- Enterprise AI Architect
