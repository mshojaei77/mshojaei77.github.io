---
title: "LLMs: From Foundation to Production"
nav_order: 1
has_children: true
permalink: /book/
description: "A comprehensive guide to Large Language Models from mathematical foundations to production deployment."
keywords: "LLM, Large Language Models, Transformer Architecture, Fine-Tuning, RAG, LLMOps, Deep Learning"
author: "Mohammad Shojaei"
date: 2025-01-01
---

# LLMs: From Foundation to Production

<img width="1024" height="1536" alt="book_cover" src="https://github.com/user-attachments/assets/974f19a6-132f-434b-bf7b-f5fc9826ff0d" />

## [Introduction](intro.md)

## Part I: Foundations
- [Chapter 1: What LLM Engineers Actually Build](ch1.md)
  Understanding the full picture — from models to complete systems, text vs. voice surfaces, and common failure modes.

- **Chapter 2: Tokens, Prompts, Chats, and Structured Outputs**  
  Mastering tokenization, chat templates, system prompts, tool schemas, and why structured output beats ad-hoc parsing.

- **Chapter 3: Using Models Through the Open Ecosystem**  
  Navigating model cards, datasets, checkpoints, inference APIs, and making responsible model choices.

- **Chapter 4: RAG Building Blocks**  
  Embeddings, chunking strategies, indexing, reranking, hybrid retrieval, and citation grounding.

## Part II: Text LLM Systems
- **Chapter 5: RAG System Design and Context Engineering**  
  Building robust context pipelines with metadata, query rewriting, reranking, and source highlighting.

- **Chapter 6: Tools, Function Calling, and Agents**  
  When and how to use agents, tool integration, safety patterns, and human-in-the-loop controls.

- **Chapter 7: Fine-Tuning with PEFT**  
  Supervised fine-tuning, LoRA/QLoRA, adapters, and deciding when adaptation beats prompting or RAG.

- **Chapter 8: Evaluation and Benchmarking for Text Systems**  
  Task vs. system evaluation, rubrics, regression testing, retrieval metrics, and building eval cards.

## Part III: Audio and Realtime Systems
- **Chapter 9: Audio Data and Speech Model Fundamentals**  
  Waveforms, sampling rates, features, VAD, and the differences between ASR, TTS, S2ST, and duplex systems.

- **Chapter 10: ASR Engineering with Whisper and Beyond**  
  Transcription, timestamps, diarization, multilingual support, and quality/latency trade-offs.

- **Chapter 11: TTS Engineering and Voice Quality Control**  
  Speaker conditioning, quality metrics, voice specs, and establishing safety boundaries.

- **Chapter 12: Full-Duplex Speech-to-Speech Systems**  
  Why realtime voice is more than STT → LLM → TTS, handling interruptions, backchannels, and conversational latency.

- **Chapter 13: Multimodal and Voice-Agent Integration**  
  Orchestrating STT, LLM/tools, TTS, transport layers, and observability in voice agents.

## Part IV: Production, Security, and Portfolio
- **Chapter 14: Serving, Latency, and Optimization**  
  Batching, KV/prefix caching, quantization, autoscaling, and local inference formats.

- **Chapter 15: LLMOps, Observability, and Continuous Evaluation**  
  Tracing, prompt versioning, traffic-driven datasets, release gates, and continuous improvement.

- **Chapter 16: Security, Privacy, Governance, and Portfolio Capstones**  
  Prompt injection defense, PII handling, responsible AI practices, and packaging professional portfolio projects.

## Target Audience

- Machine Learning Engineers
- Researchers in Natural Language Processing
- Data Scientists
- Software Engineers implementing AI systems
- Graduate students in Computer Science

## License

This work is licensed under the [MIT License](https://opensource.org/licenses/MIT).

**Citation:**
```bibtex
@book{shojaei2024llms,
  title={LLMs: From Foundation to Production},
  author={Mohammad Shojaei},
  year={2025},
  url={https://mshojaei77.github.io/book/}
}
``` 
