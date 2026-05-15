**Chapter 5: Embeddings and Semantic Search**

In Chapter 4, you learned how to control what the model sees. This chapter is about finding the right information to show it.

Most useful LLM products eventually need search. A support bot needs to find the right policy. A coding assistant needs to find the right function. A note app needs to find related ideas. A memory system needs to retrieve old facts without dumping the entire history into the prompt.

Keyword search is still useful, but it is not enough. If a user asks for "refund window," the relevant document might say "returns are accepted within 30 days." A keyword system may miss it because the exact words differ. Semantic search exists to solve that gap.

An **embedding** is a list of numbers that represents meaning. Similar pieces of text should produce vectors that are close together. Once text is represented as vectors, your application can search by meaning instead of only by exact words.

The mental model for this chapter is:

<img width="1084" height="329" alt="image" src="https://github.com/user-attachments/assets/5aefc53f-869c-4fc6-8887-d2bc2274bfa3" />

That one pipeline powers document search, long-term memory, recommendations, duplicate detection, code search, and many retrieval-heavy product features.

<img width="1091" height="135" alt="image" src="https://github.com/user-attachments/assets/5415555f-dce1-4a0b-a622-d6e372511209" />

Boundary note: this chapter stops at **semantic search**. It teaches how to turn documents into searchable vectors and how to evaluate whether search found the right chunks. Chapter 6 turns those chunks into a complete RAG answer with prompt assembly, citations, missing-evidence behavior, and answer evaluation. Chapter 7 upgrades retrieval with keyword search, hybrid search, reranking, HyDE, and more advanced ranking strategies.

## What are Embeddings?

**Improved version with visible example vectors:**

### What is an Embedding?

An **embedding** is a dense vector, a long list of floating-point numbers (typically 384 to 4,096 dimensions) — that captures the semantic meaning of text, images, or other data.

Here’s a simplified **4-dimensional** example (real embeddings are much longer, but this illustrates the idea clearly):

**"How do I reset my password?"**  
```text
[ 0.80,  0.10, -0.30,  0.40 ]
```

**"I forgot my login credentials."**  
```text
[ 0.75,  0.15, -0.25,  0.45 ]
```

**"The invoice is overdue."**  
```text
[-0.20,  0.90,  0.60, -0.10 ]
```

You still don’t interpret these numbers by hand. Instead, you measure how **close** the vectors are using a formula called: **cosine similarity**:
I calculated cosine similarity using 'NumPy' on the exact 4D vectors shown:
- **Password reset** ↔ **Forgot credentials**: **0.99** (very similar) ✅
- **Password reset** ↔ **Invoice overdue**: **-0.28** (unrelated) ❌

The three common distance or similarity choices are:

| Method | Meaning | Practical Use |
| :----- | :------ | :------------ |
| **Cosine similarity** | Compares angle between vectors. | Common for text embeddings. Good default when unsure. |
| **Dot product** | Multiplies and sums vector values. | Fast and equivalent to cosine when vectors are normalized. |
| **Euclidean distance** | Measures straight-line distance. | Useful in some indexes, but less common as the first choice for text search. |

Minimal similarity code:

```python
import numpy as np

def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

query = normalize(np.array([0.2, 0.4, 0.1]))
doc = normalize(np.array([0.1, 0.5, 0.2]))

score = float(np.dot(query, doc))
print(score)
```

If your vector database says it uses cosine distance, check whether it expects normalized vectors or normalizes internally. Do not guess. Read the database and embedding model documentation.

---
**Why this works so well**

Modern embedding models (like `text-embedding-3-large`, `voyage-3`, `snowflake-arctic-embed`, etc.) have learned these representations from massive amounts of text. They understand that “reset password” and “forgot login” share the same underlying intent, even though they share almost no exact words.

<img width="520" height="400" alt="image" src="https://github.com/user-attachments/assets/bc708855-7743-4780-9b3c-7f472947c12d" />

This is useful because users rarely phrase questions exactly like your documents. Semantic search gives your application a better chance of finding relevant text when wording differs.

There are two broad retrieval signals:

| Signal | What it Matches | Strength | Weakness |
| :----- | :-------------- | :------- | :------- |
| **Sparse / keyword** | Exact or near-exact terms, often with BM25. | Great for names, IDs, error codes, exact phrases. | Misses related meaning when words differ. |
| **Dense / embedding** | Semantic similarity through vectors. | Great for paraphrases and conceptual matches. | Can miss exact constraints, rare terms, and identifiers. |

Do not treat dense search as a replacement for keyword search. Treat it as another signal with another use case.

## Choosing Embedding Models

There is no universal best embedding model. There is the best model for your documents, users, language mix, latency target, cost target, and deployment constraints.

Start with these questions:

*   What are you embedding: support docs, code, PDFs, chat history, products, legal text, images, or multilingual content?
*   How long are the inputs?
*   Do you need hosted APIs, local inference, or both?
*   Is data allowed to leave your infrastructure?
*   What latency and cost per indexed document are acceptable?
*   How often will the corpus be refreshed?
*   Can you afford to re-embed everything when changing models?

The current landscape changes quickly, but the main categories are stable:

| Category | Examples | Good Fit | Tradeoff |
| :------- | :------- | :------- | :------- |
| **Hosted general models** | OpenAI `text-embedding-3-small`, Google `gemini-embedding-2` | Fast integration, managed scaling, strong baseline quality. | Data leaves your app boundary; provider pricing and versions can change. |
| **Hosted retrieval-specialized models** | Voyage AI embedding models, Cohere Embed v3. | Search-heavy products, domain-specific retrieval, strong API features. | Another vendor dependency; evaluate on your own data. |
| **Open and local models** | BGE, E5, `nomic-embed-text`, `mxbai-embed-large`, EmbeddingGemma, Qwen Embedding | Privacy, cost control, offline use, customization. | You own serving, batching, hardware, and model upgrades. |
| **Small local baselines** | `sentence-transformers/all-MiniLM-L6-v2`. | Learning, prototypes, small apps, local demos. | Lower quality on hard or domain-specific retrieval. |

For current model names, dimensions, and API details, check the official docs before implementation: [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings), [Voyage AI embeddings](https://docs.voyageai.com/docs/embeddings), [Cohere embeddings](https://docs.cohere.com/docs/embeddings), [Google Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings), and [Sentence Transformers](https://sbert.net/).

Use leaderboards such as [MTEB](https://huggingface.co/spaces/mteb/leaderboard) for shortlisting, not final decisions. MTEB is useful because it compares embedding models across many tasks. It is not your product, your documents, or your users.

Practitioners increasingly compare hosted models against local models for privacy and cost control. That does not mean local is always better. Hosted APIs reduce operational work and are often strong baselines. Local models give control, but you own serving, batching, CPU/GPU sizing, upgrades, and evaluation. For many teams, the useful test is not "hosted or local?" It is "which option meets our retrieval quality bar at the lowest total operational cost?"

For local experiments, tools such as Ollama and `sentence-transformers` make it easy to compare candidates. For production-minded evaluation, keep the test fixed: same documents, same chunks, same queries, same vector store, same metric. Change only the embedding model. Otherwise you will not know whether the improvement came from the model or from chunking, filtering, or ranking.

Engineering consequence: embeddings turn search into a model-dependent system. If the embedding model changes, your vectors change. You cannot safely mix vectors from different embedding models in the same index and expect distances to mean the same thing.

Common mistakes:

*   embedding whole documents instead of useful chunks
*   mixing embedding models in one collection
*   using semantic search for exact IDs, SKUs, filenames, or error codes where keyword search is stronger
*   assuming the closest vector is always the correct answer
*   choosing the model with the highest benchmark score without testing your documents
*   ignoring multilingual or domain-specific needs
*   switching models without rebuilding the index
*   forgetting that larger vectors increase storage and search cost
*   ignoring rate limits and batch limits during ingestion

## Practical Chunking Strategies

Embedding a whole document usually works poorly. A long PDF may contain ten unrelated ideas. One vector cannot represent all of them precisely.

Chunking splits documents into smaller pieces before embedding:

<img width="956" height="179" alt="image" src="https://github.com/user-attachments/assets/8ff176b9-dfa7-435e-af21-64f16beab76c" />

A good chunk is large enough to contain useful context and small enough to stay focused.

Common chunking strategies:

| Strategy | How it Works | Use When | Risk |
| :------- | :----------- | :------- | :--- |
| **Fixed-size with overlap** | Split every N words or tokens, repeat a small overlap. | Quick baseline, simple docs. | Can cut through headings, tables, code, or arguments. |
| **Recursive splitting** | Try paragraphs, then sentences, then words until size fits. | General text. | Still structure-blind for complex docs. |
| **Structural splitting** | Split by headings, sections, Markdown, HTML, or document layout. | Policies, docs, manuals, code docs. | Requires better parsing. |
| **Semantic chunking** | Split when meaning shifts. | Essays, transcripts, mixed-topic text. | Slower and harder to tune. |
| **Agentic chunking** | Use a model to decide chunk boundaries. | High-value corpora where quality matters. | Adds cost, latency, and nondeterminism. |
| **Late chunking** | Use model context over a longer passage before creating chunk vectors. | Long documents where local chunks need broader context. | More complex and model-dependent. |

Useful open-source references exist for experimenting with chunking: `langchain-text-splitters`, LlamaIndex splitters, `superlinked/chunking-research`, Jina AI's late-chunking examples, and smaller semantic chunker packages. Use them to learn and compare strategies, not as a reason to skip evaluation. Chunking libraries make splitting easier; they do not know your product's relevance criteria.

Start simple:

```python
def chunk_words(text: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)

    return chunks
```

This is not the final chunker for a serious document product. It is a baseline. A baseline matters because it gives you something to evaluate before adding complexity.

When documents have structure, preserve it. Headings, page numbers, section titles, URLs, dates, and source IDs often matter as much as the text itself. Store them as metadata:

```python
metadata = {
    "source": "refund_policy.md",
    "section": "Refund Window",
    "chunk_index": 3,
}
```

Chunking tradeoffs:

*   **Small chunks** improve precision but may lose context.
*   **Large chunks** preserve context but can retrieve irrelevant text.
*   **Overlap** prevents boundary loss but increases storage and duplicate retrieval.
*   **Heading-aware chunks** improve citations and debugging.
*   **Semantic chunks** can improve quality but are harder to reproduce.

A practical tuning loop:

1. Start with 150-300 words per chunk and 10-20 percent overlap.
2. Add metadata: source, section, page, date, tenant, permissions.
3. Run retrieval evals.
4. Inspect misses.
5. Adjust chunk size, overlap, and splitting rules.
6. Re-run evals before changing the embedding model.

Common mistakes:

*   optimizing chunking by intuition instead of retrieval tests
*   chunking by characters when token limits matter
*   dropping headings and source metadata
*   using the same chunking strategy for contracts, code, chat logs, and product pages
*   making overlap so large that near-duplicate chunks dominate results
*   splitting code in the middle of functions or classes

## Embedding Generation Best Practices

Embedding generation looks simple: call a model on text and store vectors. Production failures usually come from the details.

Basic hosted API shape:

```python
from openai import OpenAI

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[
        "Refunds are available within 30 days.",
        "Enterprise accounts require manual approval.",
    ],
)

vectors = [item.embedding for item in response.data]
```

Basic local model shape:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

vectors = model.encode(
    [
        "Refunds are available within 30 days.",
        "Enterprise accounts require manual approval.",
    ],
    batch_size=32,
    normalize_embeddings=True,
)
```
CPU embedding inference can be good enough for small jobs and prototypes. GPU inference helps when you have large corpora, frequent refreshes, or high-volume local embedding. Hosted APIs avoid local serving but introduce provider latency, rate limits, and data-handling questions.

Engineering consequence: embedding is an ingestion workload, not just a model call. Treat it like a data pipeline with retries, idempotent writes, progress tracking, and the ability to resume after failure.

Best practices:

*   **Batch requests** to reduce overhead and improve throughput.
*   **Cache embeddings** by model name, model version, dimensions, normalized text, and chunking version.
*   **Normalize consistently** if your distance metric expects it.
*   **Store raw text and metadata** with each vector so results can be inspected.
*   **Use async jobs** for large corpora instead of embedding during user requests.
*   **Track failures**: rate limits, timeouts, truncation, empty chunks, duplicate IDs.
*   **Make re-indexing repeatable** so you can rebuild after model or chunking changes.

## Vector Databases and Storage Options

A vector database stores vectors and searches for nearest neighbors. Some are full databases. Some are libraries. Some are search engines with vector support.

A practical classification:

| Option Type | Examples | Good Fit | Watch For |
| :---------- | :------- | :------- | :-------- |
| **Local libraries** | FAISS, hnswlib, LanceDB. | Experiments, local apps, custom pipelines. | You handle persistence, filtering, serving, and ops details. |
| **Prototype stores** | ChromaDB. | Learning, notebooks, small prototypes. | Validate durability and operational needs before relying on it. |
| **Postgres-based** | pgvector, Supabase vector tooling. | Apps already using Postgres, moderate scale, relational metadata. | Tuning and scale limits depend on workload. |
| **Dedicated vector DBs** | Qdrant, Pinecone, Weaviate, Milvus. | Production retrieval, filtering, scale, managed or self-hosted options. | Vendor choices, pricing, ops, migration complexity. |
| **Search platforms with vectors** | Elasticsearch, OpenSearch, Azure AI Search, MongoDB Atlas Vector Search. | Teams already using those platforms. | Vector search may be one feature among many; understand limits. |
| **Specialized storage** | Astra DB, LEANN and other compressed or serverless options. | Specific platform or compression needs. | Evaluate maturity, ecosystem, and operational fit. |

The practical community pattern is boring but correct: vector storage is becoming a commodity for basic use cases. Start with the database that makes your product easier to operate, then migrate only when a specific bottleneck justifies the complexity.
The database decision is mostly operational:

*   **If you already rely on PostgreSQL**, `pgvector` keeps the stack simple. Use it until you exceed ~50 million vectors or need sub‑10ms latency at high QPS with heavy filtering.
*   **If you’re prototyping or building a learning project**, Chroma DB offers the fastest time‑to‑first‑result. Treat it as a temporary index — plan to move to a production‑grade store before launch.
*   **For most production self‑hosted setups**, Qdrant is the community’s top recommendation. It delivers excellent speed, low memory overhead, and advanced metadata filtering, hitting the sweet spot between performance and operational simplicity.
*   **At extreme scale (billions of vectors, GPU‑accelerated),** Milvus or its managed counterpart Zilliz dominates. It demands solid DevOps skills but delivers the throughput.
*   **When hybrid search (keyword BM25 + vector) is a hard requirement**, Weaviate and Elasticsearch/OpenSearch provide native, best‑in‑class support. Choose Weaviate if your data is graph‑like; choose Elasticsearch/OpenSearch if you’re already on the ELK stack.
*   **Zero‑ops managed services** like Pinecone let you launch quickly, but Reddit experience shows costs spike and filtering limitations hurt at scale. Use it for fast market validation, but keep a migration path open.
*   **FAISS is not a standalone database.** Use it as a low‑level library for embedded/on‑device search or when you need full control over indexing for research and offline workloads.
*   **If storage size is the bottleneck**, newer compressed or recomputation‑oriented systems (like LEANN) are workload‑specific choices — treat them as optimisations, not defaults.

The typical growth path matches the community’s real journeys: start with Chroma/pgvector → scale with Qdrant/Weaviate → or go managed with Zilliz if the team can’t self‑host.

Do not choose a vector database only from a marketing comparison. Test your own workload:

```text
corpus size (vectors)        – <1M? almost any store works; >50M? evaluate Qdrant, Milvus, or tuned pgvector
vector dimensions            – high-dims increase memory; check compression capabilities
metadata filters             – pre-filtering performance varies hugely; Qdrant/pgvector/Weaviate excel
top_k                        – large top_k may expose post-filtering inefficiencies (Pinecone, FAISS)
query volume (QPS)           – low QPS hides management overhead; high QPS demands dedicated stores
latency target (p95)         – sub-50ms at scale often rules out Elasticsearch vector-only, favours Qdrant/Milvus
update frequency             – frequent inserts/deletes hurt rebuild-heavy stores; prefer incremental indexing
backup & HA requirements     – pgvector inherits Postgres; dedicated stores vary widely
tenant isolation             – row-level security (pgvector), namespaces (Pinecone), or collections (Qdrant)
```

Benchmark the top two candidates with your actual data shape, query patterns, and filters. The right answer in production is the one you can operate with confidence, not the one that tops a synthetic benchmark.

FAISS deserves a special note. It is not a full product database. It is a high-performance similarity search library. It is excellent when you want local control over index types. It is less convenient when you need multi-tenant permissions, transactional document updates, backups, query APIs, and operational dashboards. 

Common mistakes:

*   ignoring metadata filters until security or relevance breaks
*   storing vectors without source text
*   using one namespace for all tenants
*   forgetting deletes and document refreshes
*   optimizing ANN parameters before fixing chunk quality
*   treating a vector database benchmark as proof that your retrieval quality is good

## Basic Semantic Search Implementation

Now build the smallest useful semantic search layer: chunk text, embed chunks, store them in ChromaDB, and query with a user question.

Install dependencies:

```bash
pip install sentence-transformers chromadb
```

Index a few documents:

```python
import chromadb
from sentence_transformers import SentenceTransformer

documents = {
    "refund_policy": """
    Customers can request a refund within 30 days of purchase.
    Refunds over 100 dollars require manager approval.
    """,
    "upload_help": """
    If file upload fails, check the file size and supported formats.
    The maximum upload size is 25 MB.
    """,
}


def chunk_words(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
        if words[start:start + chunk_size]
    ]


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(
    name="support_docs",
    metadata={"hnsw:space": "cosine"},
)

ids = []
chunks = []
metadatas = []

for doc_id, text in documents.items():
    for i, chunk in enumerate(chunk_words(text)):
        ids.append(f"{doc_id}:{i}")
        chunks.append(chunk)
        metadatas.append({"source": doc_id, "chunk_index": i})

embeddings = model.encode(
    chunks,
    batch_size=32,
    normalize_embeddings=True,
)

collection.upsert(
    ids=ids,
    documents=chunks,
    metadatas=metadatas,
    embeddings=embeddings.tolist(),
)
```

Query the collection:

```python
def search(query: str, top_k: int = 3) -> list[dict]:
    query_vector = model.encode(query, normalize_embeddings=True)

    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for document, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        matches.append({
            "text": document,
            "source": metadata["source"],
            "distance": distance,
        })

    return matches


for match in search("Can I get my money back after buying?"):
    print(match["source"], match["distance"], match["text"])
```

For this Chroma collection, lower distance means a closer match. Different databases report scores differently, so always check whether the value is a similarity score, distance, or transformed relevance score.

This is a prototype. It proves the search loop works, but it is not production-ready. A production-minded version would add:

*   document loading from files or a database
*   stable document IDs
*   duplicate detection
*   metadata filters
*   ingestion logs
*   retryable embedding jobs
*   evaluation tests
*   delete and refresh workflows

Engineering consequence: semantic search is not one API call. It is a data system. The quality of the search result depends on document parsing, chunking, embedding, metadata, index settings, and query behavior.

## Evaluation Metrics for Retrieval

Do not evaluate retrieval by asking, "Did the chatbot answer well?" That mixes retrieval quality with generation quality. Evaluate retrieval first.

You need a small test set:

```python
eval_cases = [
    {
        "query": "Can I get my money back after buying?",
        "relevant_sources": {"refund_policy"},
    },
    {
        "query": "What is the upload size limit?",
        "relevant_sources": {"upload_help"},
    },
]
```

Useful metrics:

| Metric | Question it Answers |
| :----- | :------------------ |
| **Hit Rate@k** | Did at least one relevant result appear in the top k? |
| **Recall@k** | What fraction of all relevant items appeared in the top k? |
| **MRR** | How high was the first relevant result? |
| **nDCG@k** | Were more relevant results ranked higher? |
| **Latency** | How long did search take at p50 and p95? |
| **Cost** | What did embedding and querying cost per successful search? |
| **Freshness** | Are indexed documents up to date? |
| **Coverage** | What fraction of the corpus is searchable? |

`nDCG@k` is useful when you have graded relevance, not just relevant or irrelevant. For example, a result can be "perfect," "partially useful," or "not useful." That matters because the best retrieval systems do not merely find one relevant chunk; they rank the most useful evidence first.

Minimal Hit Rate@k and MRR:

```python
def hit_rate_at_k(results: list[list[str]], expected: list[set[str]], k: int) -> float:
    hits = 0
    for retrieved_sources, relevant_sources in zip(results, expected):
        if set(retrieved_sources[:k]) & relevant_sources:
            hits += 1
    return hits / len(expected)


def mrr(results: list[list[str]], expected: list[set[str]]) -> float:
    total = 0.0
    for retrieved_sources, relevant_sources in zip(results, expected):
        for rank, source in enumerate(retrieved_sources, start=1):
            if source in relevant_sources:
                total += 1 / rank
                break
    return total / len(expected)
```

Example evaluation loop:

```python
import math

def dcg(scores: list[float]) -> float:
    return sum(
        score / math.log2(rank + 1)
        for rank, score in enumerate(scores, start=1)
    )


def ndcg_at_k(relevance_scores: list[list[float]], k: int) -> float:
    values = []
    for scores in relevance_scores:
        actual = dcg(scores[:k])
        ideal = dcg(sorted(scores, reverse=True)[:k])
        values.append(actual / ideal if ideal else 0.0)
    return sum(values) / len(values)
```

Example evaluation loop:

```python
retrieved = []
expected = []

for case in eval_cases:
    matches = search(case["query"], top_k=5)
    retrieved.append([match["source"] for match in matches])
    expected.append(case["relevant_sources"])

print("hit_rate@3", hit_rate_at_k(retrieved, expected, k=3))
print("mrr", mrr(retrieved, expected))
```

For early projects, 20-50 well-designed eval cases are enough to catch many failures. Include easy questions, paraphrases, exact-term queries, missing-answer queries, ambiguous queries, noisy list-data queries, code-symbol queries, and access-control cases.

Use MTEB and public benchmarks to shortlist models. Use your own eval set to choose the model, chunking strategy, and vector store configuration.

LLM-based evaluation tools can be useful later, but they can become expensive quickly. Retrieval evaluation does not need a judge model at first. If you can label which documents or chunks should be returned, standard information-retrieval metrics are cheaper, more precise, and easier to reproduce. For business-critical systems, add subject-matter expert review and user feedback after the offline metrics are stable.

Use the right evaluation tool for the layer you are testing:

| Tool / Framework | Use in This Chapter | Move to Later Chapters |
| :--------------- | :------------------ | :--------------------- |
| **MTEB** | Shortlist embedding models and inspect benchmark tasks. | Do not treat leaderboard rank as product truth. |
| **BEIR** | Learn classic retrieval evaluation patterns. | Use custom corpora before claiming production quality. |
| **RAGAS / DeepEval** | Mostly out of scope here. | Useful in Chapter 6 when evaluating generated RAG answers. |
| **Vector DB benchmarks** | Estimate speed and memory tradeoffs. | Do not use them as relevance metrics. |

## Common Pitfalls and Failure Scenarios

Embedding search fails in predictable ways.

| Failure | Likely Cause | Fix |
| :------ | :----------- | :-- |
| Good document never appears. | Bad chunking, truncation, wrong model, missing metadata filter. | Inspect chunks, test Recall@k, verify token limits. |
| Results are semantically similar but useless. | Dense search overweights general meaning. | Add keyword search, metadata filters, or reranking. |
| Exact IDs are missed. | Embeddings are weak for exact symbols. | Use keyword filters or hybrid retrieval. |
| Top results are duplicates. | Too much overlap or duplicate documents. | Reduce overlap, deduplicate, diversify results. |
| Retrieval quality slowly degrades. | Embedding drift, stale documents, changed docs not re-indexed. | Track index freshness and model/chunking versions. |
| Search is too slow. | Large vectors, weak index settings, too many filters, no batching. | Tune index, reduce dimensions, use ANN, cache frequent queries. |
| Costs surprise you. | Re-embedding too often, large chunks, no cache. | Cache embeddings, batch jobs, refresh incrementally. |
| Users see unauthorized content. | Missing access-control filters. | Enforce tenant/user permissions before retrieval or inside DB filters. |
| Short queries retrieve vague chunks. | Query signal is too weak. | Add metadata, use better chunk titles, or defer query expansion/HyDE to Chapters 6 and 7. |
| Noisy list data dominates results. | Repeated list structure overwhelms meaningful text. | Split lists structurally, add headers, or index summarized child chunks. |

**Embedding drift** means your retrieval behavior changes because the embedding model, preprocessing, chunking, or source documents changed. It can be silent. The app still returns results, but worse ones.

Track these fields with every index:

```text
embedding_model
embedding_dimensions
distance_metric
normalization
chunker_version
source_document_version
indexed_at
```

Common debugging process:

1. Print the retrieved chunks.
2. Check whether the right document was indexed.
3. Check whether the right text was inside a chunk.
4. Check whether metadata filters removed it.
5. Check whether the query wording needs expansion.
6. Compare dense search with keyword search.
7. Measure on an eval set instead of one example.

## Hands-On Exercise

Build a semantic document search engine over your own notes, Markdown files, or PDFs converted to text.

Goal:

```text
Given a user question, retrieve the most relevant chunks from your document collection.
```

Requirements:

1. Choose 5-20 documents.
2. Split them into chunks.
3. Generate embeddings with `sentence-transformers/all-MiniLM-L6-v2` or another embedding model.
4. Store chunks, vectors, and metadata in ChromaDB, pgvector, FAISS, LanceDB, or another vector store.
5. Implement a `search(query, top_k)` function.
6. Print the retrieved text, source document, chunk ID, and score or distance.
7. Build at least 10 evaluation queries with expected source documents.
8. Measure Hit Rate@3 and MRR.
9. Change one variable: chunk size, overlap, embedding model, or top_k.
10. Compare the results before and after.

Constraints:

*   Keep the first version local.
*   Do not add an LLM answer generator yet.
*   Do not optimize the vector database before evaluating chunk quality.
*   Do not use private or sensitive documents unless you understand where embeddings are sent.

Expected behavior:

*   Queries with different wording should retrieve relevant chunks.
*   Exact terms like product names or file names may still need keyword search.
*   Evaluation should show at least one measurable change when you adjust chunking or model choice.

How you know it works:

*   You can run ingestion repeatedly without duplicate IDs.
*   You can inspect retrieved chunks and understand why they matched.
*   Hit Rate@3 and MRR are computed from a small eval set.
*   You can explain one retrieval failure and make a targeted improvement.

That is the core skill. Once you can build and evaluate semantic search, RAG becomes an engineering problem instead of a guessing game.
