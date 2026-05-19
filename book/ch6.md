## RAG System Learning Checklist (2026 Edition)

- [ ] **1. Internalise the goals of a complete RAG system**  
  Ground answers in evidence, cite sources precisely, respect access-control boundaries, mitigate model knowledge‑cutoffs, and degrade gracefully when retrieval fails.

- [ ] **2. Learn the full RAG pipeline (end‑to‑end)**  
  Ingestion → parsing → chunking → embedding → vector store → hybrid retrieval (dense + sparse/keyword) → reranking → context assembly → prompt construction → generation → source attribution → evaluation → feedback loop.

- [ ] **3. Understand the de‑facto 2026 production stack (Milvus‑centric)**  
  Real‑world projects consistently combine:
  - **Vector DB**: Milvus 
  - **Orchestration**: LangChain / LangGraph for structured chunking, retrieval routing, and agentic workflows.
  - **Retrieval pattern**: Hybrid (dense + BM25/sparse) + **GTE‑Rerank** – pure vector search alone is insufficient.
  - **LLM**: Qwen3.6 / DeepSeek (API‑compatible) or Ollama‑hosted open‑weight models for privacy/cost control.
  - **Multi‑hop reasoning**: Vector Graph RAG (entities + relations + passages inside Milvus) without a separate graph DB.
  - **Serving**: FastAPI with streaming; Skill‑first LangGraph agents for tool use and autonomous multi‑step reasoning.

- [ ] **4. Master advanced ingestion & document parsing**  
  Tools: Docling, Marker, Unstructured.io, LlamaParse, Azure Document Intelligence.  
  Techniques: OCR, table extraction, PDF structure recovery, HTML/Markdown cleaning, metadata enrichment, and handling non‑text modalities.

- [ ] **5. Adapt RAG to multilingual and domain‑specific data**  
  Finance, legal, logistics, support – requires domain‑specific chunking, custom embedding fine‑tuning, specialised rerankers, and glossary‑aware retrieval.

- [ ] **6. Architect for large‑scale data (100GB+ and streaming updates)**  
  Incremental vector ingestion, change‑data‑capture pipelines, streaming context assembly, versioned indexes, source provenance tracking, and maintaining low‑latency SLAs under load.

- [ ] **7. Contrast naive vs. advanced RAG patterns**  
  Naive: single‑retrieve → single‑generate.  
  Advanced: query rewriting, HyDE, step‑back prompting, self‑querying, parent‑document retrieval, corrective RAG (CRAG), agentic RAG (tool use), GraphRAG, LightRAG, and context compression.

- [ ] **8. Diagnose common production failures**  
  Recognise that poor chunking, stale embeddings, short‑query under‑retrieval, context‑window overflow, missing access control, and hallucination on un‑retrievable topics are often misattributed to “the model”. Learn to inspect retrieval traces, not just final answers.

- [ ] **9. Implement security, governance & citation**  
  Row‑level / document‑level access control (e.g., OpenFGA‑style), pre‑retrieval filtering (never post‑generation), audit trails, data‑residency constraints, verifiable source citations, and refusal strategies for out‑of‑scope questions.

- [ ] **10. Evaluate, monitor, and operate RAG in production**  
  Frameworks: RAGAS, DeepEval.  
  Build golden evaluation sets (starting with 10–15 real user questions), track faithfulness, answer relevance, latency, and token cost. Set cost/latency budgets, implement canary rollouts, and run continuous regression tests before model or index updates.