# Portfolio

## Mohammad Shojaei

AI Engineer focused on production LLM agents, retrieval systems, evaluation, and applied machine learning.

[Email](mailto:shojaei.dev@gmail.com) - [LinkedIn](https://www.linkedin.com/in/mshojaei77) - [GitHub](https://github.com/mshojaei77) - [Hugging Face](https://huggingface.co/mshojaei77)

---

## Profile

I build production-oriented AI systems: LLM agents, RAG pipelines, OpenAI-compatible APIs, evaluation workflows, and machine-learning tools for real-world domains such as real estate, finance, healthcare, education, and Persian NLP.

My recent work is centered on agentic systems that need to be reliable outside demos: streaming responses, tool calling, conversation memory, retrieval quality checks, structured outputs, tracing, cost and token tracking, regression checks, and deployment runbooks. I work mostly with Python, FastAPI, LangGraph, Hugging Face, vLLM, vector search, reranking, QLoRA/LoRA fine-tuning, and evaluation frameworks.

I care about AI systems that are measurable, debuggable, and honest about their limits. A good system should not only answer; it should expose its sources, failure modes, latency, cost, and quality signals.

---

## Current Focus

- Building production LLM agents with tool calling, memory, routing, monitoring, and safe fallback behavior.
- Designing RAG and deep-research pipelines with source-grounded answers, reranking, citation filtering, and retrieval evaluation.
- Fine-tuning and evaluating Qwen/Gemma-family models for domain-specific behavior, especially Persian and Russian-domain workflows.
- Turning prototypes into maintainable services with FastAPI, OpenAI-compatible APIs, streaming, structured logs, health checks, systemd deployment, and benchmark suites.

---

## Experience

### AI Engineer, [Sunlife Estate](https://sunlife.estate/)

Remote, Moscow - May 2025 - Present

- Building a Russian real-estate voice agent that integrates ASR, LLM reasoning, XTTS, and LiveKit telephony for property outreach, qualification, objection handling, callback flows, and specialist handoff.
- Developed an OpenAI-compatible backend with streaming responses, tool calling, thread-safe conversation context, token tracking, caching, prompt optimization, detailed monitoring, and persistent memory.
- Designed a LangGraph-based conversation engine with intent classification, dynamic routing, busy-user handling, company-information responses, multilingual property-search tools, and DuckDB full-text search.
- Fine-tuned and evaluated Qwen/Gemma-family models for real-estate operator conversations, improving brevity, instruction adherence, and sales-script consistency.
- Reported sub-250ms p95 latency and about 35% infrastructure cost reduction through streaming, quantization, vLLM serving, model-selection trade-offs, tracing, benchmark suites, and operational runbooks.

### AI Engineer, [No Limit Markets Ltd](https://nolimitmarkets.com/)

Remote, Dubai - Jan 2025 - Dec 2025

- Built AI-powered financial research and market-intelligence agents for stock analysis, financial news monitoring, trading-signal processing, Persian financial content generation, and Telegram-ready reporting.
- Developed an 8-stage FinancialKG pipeline for RSS ingestion, web scraping, topic extraction, web search, financial analysis, post generation, Persian translation, and Telegram publishing.
- Built a Tehran Stock Exchange multi-agent system with dedicated data, analysis, and content agents for technical analysis, fundamental analysis, risk assessment, and structured financial reporting.
- Integrated PyTSETMC, DuckDuckGo Search, Exa Search, RSS feeds, web scraping, Codal/TSETMC-style sources, custom embeddings, and LangGraph tool-calling workflows.
- Added retries, checkpointing, link deduplication, admin commands, daily reports, structured logs, error recovery, schema-guided generation, LLM-as-judge review, and report validation.

### ML Engineer, [Alpha Neuroscience](https://www.alphaica.com)

Tehran - May 2022 - May 2024

- Built ML and signal-processing features for ALPHA, a PyQt5 desktop platform for real-time EEG visualization, recording, analysis, quality monitoring, patient management, clinical review, and report generation.
- Developed EEG preprocessing workflows with MNE-Python, SciPy, NumPy, and Pandas for EDF loading, filtering, notch filtering, artifact handling, segmentation, and frequency-domain analysis.
- Implemented real-time signal-quality and artifact-detection logic covering channel correlation, flatline detection, adaptive thresholds, RMS/PSD checks, eye blinks, EMG, electrode pops, cable movement, baseline drift, and 50/60Hz interference.
- Built spectral-analysis modules for FFT, PSD, band-power extraction, Z-scored power ratios, and topographical brain-map visualizations across Delta, Theta, Alpha, Beta, and High-Beta bands.
- Built medical assistant and report-support workflows that helped technicians and clinicians summarize EEG findings and review patient context.

### Independent AI/LLM Engineer

Remote - 2023 - Present

- Built custom RAG, deep-research, and knowledge-management systems using FAISS, Weaviate, LangChain, LlamaIndex, OpenRouter, Ollama, Streamlit, custom chunking, reranking, memory, citations, and persistent Q&A workflows.
- Developed document-grounded Persian banking and enterprise assistants with source-aware retrieval, citation filtering, streaming answers, reranking, and strict answer-from-documents guardrails.
- Built educational AI systems that convert lecture videos and transcripts into interactive learning materials, including transcript retrieval, summaries, quizzes, and tutorial-style UI generation.
- Delivered VLM/OCR and document-extraction pipelines using Qwen-VL, Florence-2, EasyOCR, PaddleOCR, Tesseract, object detection, salary-document extraction, and OCR quality comparison workflows.

---

## Selected Open-Source Work

- [ReActMCP](https://github.com/mshojaei77/ReActMCP): MCP web-search server/client that connects Exa-powered search to AI assistants.
- [pytsetmc-api](https://github.com/mshojaei77/pytsetmc-api): typed Python client for Tehran Stock Exchange Market Center data retrieval with Pydantic validation, retries, logging, fallbacks, CLI usage, tests, and documentation.
- [Hugging Face](https://huggingface.co/mshojaei77): 14+ models, Persian NLP datasets, and leaderboard spaces focused on low-resource evaluation, tokenizer training, LoRA adapters, model cards, dataset cards, and Persian benchmark workflows.
- Broader GitHub portfolio: 50+ public repositories across LLM tooling, local AI apps, RAG, MCP clients, prompt engineering, web scraping, browser extensions, subtitle translation, VRAM estimation, Persian LLM evaluation, and AI learning resources.

---

## Technical Skills

**LLM and agents:** LangGraph, CrewAI, AutoGen, tool/function calling, multi-step agents, conversation memory, dynamic routing, intent classification, specialist handoff, ReAct-style workflows, MCP, prompt optimization, guardrails, agent evaluation.

**RAG and retrieval:** FAISS, Weaviate, LangChain, LlamaIndex, custom chunking, reranking, citation-aware retrieval, source-grounded Q&A, DuckDB FTS, vector search, RAGAS, DeepEval.

**Model engineering:** Hugging Face Transformers, Qwen, Gemma, QLoRA/LoRA, PEFT, bitsandbytes, Unsloth, W&B, fine-tuning, quantization, model cards, dataset cards, Whisper fine-tuning, TTS.

**Production AI and LLMOps:** FastAPI, OpenAI-compatible APIs, streaming responses, vLLM serving, token/cost tracking, caching, structured logs, tracing, health checks, systemd deployment, operational runbooks, latency optimization, regression checks, Kubernetes, MLflow, CI/CD for ML.

**Data and ML:** Python, NumPy, Pandas, SciPy, MNE-Python, signal processing, feature engineering, anomaly detection, FFT, PSD, EEG preprocessing, PySpark.

**Automation and integrations:** web scraping, RSS ingestion, Telegram bots, MCP clients, SSE/stdio transports, OpenAI function calling, financial-data APIs, OCR/VLM pipelines, structured extraction workflows.

---

## Education and Training

**B.Sc. Computer Engineering**
Higher Education Complex of Bam - 2020 - 2024

- LangChain for LLM Application Development, DeepLearning.AI, 2024
- CS224N: Natural Language Processing with Deep Learning, Stanford, 2024
- Machine Learning Specialization, DeepLearning.AI, 2023
- CS50P: Introduction to Programming with Python, Harvard, 2023

---

## What This Site Is About

This site is where I publish practical learning material on LLMs, RAG, agents, evaluation, and production AI engineering. The goal is to make advanced AI engineering easier to reproduce: clear concepts, concrete code, realistic failure modes, and measurable outcomes.
