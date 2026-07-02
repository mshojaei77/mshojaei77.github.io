**Chapter 5: Embeddings and Semantic Search**

In Chapter 4, you learned how to control what the model sees and how it formats its output. But as your application grows, you will quickly hit a wall: you cannot fit your entire database, document library, or chat history into the model’s context window. You need a way to find exactly the right information to show it.

Most useful LLM products eventually require search. A support bot needs to find the right policy. A coding assistant needs to find the right function. A memory system needs to retrieve old facts without dumping a gigabyte of history into the prompt. 

Keyword search is a powerful tool, but it is not enough on its own. If a user asks for "refund window," the relevant document might say "returns are accepted within 30 days." A keyword system might miss this completely because the exact words differ. We need a system that understands meaning, not just spelling.

In this chapter, **retrieval** means selecting useful stored information for a current question. **Semantic search** is retrieval by meaning. It tries to match a question with text that says the same thing, even when the vocabulary is completely different. 

The mental model for this chapter looks like this:

<img width="1084" height="329" alt="image" src="https://github.com/user-attachments/assets/5aefc53f-869c-4fc6-8887-d2bc2274bfa3" />

That single pipeline powers document search, long-term memory, recommendations, duplicate detection, and code search. 

<img width="1091" height="135" alt="image" src="https://github.com/user-attachments/assets/5415555f-dce1-4a0b-a622-d6e372511209" />

*Boundary note:* This chapter stops strictly at semantic search. We will learn how to turn documents into searchable numbers and how to evaluate whether our search found the right text. Chapter 6 will use these search results to build a complete **RAG (Retrieval-Augmented Generation)** system, where the application assembles a prompt and cites sources. Chapter 7 will upgrade retrieval by mixing it with keyword search and advanced ranking strategies.

**Code Repository**  
The complete runnable companion project for this chapter is available in the book repository:  
[github.com/mshojaei77/llm-engineering-in-action/chapter-05-semantic-search](https://github.com/mshojaei77/llm-engineering-in-action/tree/main/chapter-05-semantic-search)

The repository contains a local semantic-search workbench with chunking, a persistent FAISS index, evaluation queries, and retrieval metrics. The chapter explains the retrieval mechanics; the repository gives you the runnable local pipeline.

For now, let's look at how software can measure meaning.

---

## What is an Embedding?

To make semantic search work, we need to turn text into a format that software can mathematically compare. 

A **vector** is simply an ordered list of numbers. An **embedding** is a specific type of dense vector,a long list of floating-point numbers (typically 384 to 4,096 dimensions),that captures the semantic meaning of text. "Dense" means that almost every position in the vector carries a useful value. A "dimension" is just one of those positions.

Here is a simplified **4-dimensional** example. Real embeddings are much longer, but this illustrates the concept clearly:

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

You never interpret these numbers by hand. Instead, you measure how **close** the vectors are to each other using a mathematical formula. The most common formula is **cosine similarity**, which measures the angle between two vectors. 

If we calculate the cosine similarity for our example:
- **Password reset** ↔ **Forgot credentials**: **0.99** (very similar) ✅
- **Password reset** ↔ **Invoice overdue**: **-0.28** (unrelated) ❌

There are three common ways to calculate this distance:

| Method | Meaning | Practical Use |
| :----- | :------ | :------------ |
| **Cosine similarity** | Compares the angle between vectors. | The standard for text embeddings. It is the best default when unsure. |
| **Dot product** | Multiplies and sums vector values. | Extremely fast. It is mathematically equivalent to cosine similarity *if* the vectors are normalized. |
| **Euclidean distance** | Measures straight-line distance. | Useful in some specific spatial indexes, but less common as the first choice for text search. |

To **normalize** a vector means to rescale its values so its total length is exactly 1, while preserving its direction. Normalization is a standard engineering trick because it allows you to use the lightning-fast dot product while getting the exact same ranking results as cosine similarity.

Here is the minimal code to prove how this works:

```python
import numpy as np

def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

# Let's say these are our embeddings
query = normalize(np.array([0.2, 0.4, 0.1]))
doc = normalize(np.array([0.1, 0.5, 0.2]))

# Because they are normalized, dot product equals cosine similarity
score = float(np.dot(query, doc))
print(f"Similarity Score: {score:.4f}")
```

If your chosen database says it uses "cosine distance," always check whether it expects you to normalize the vectors beforehand or if it does it internally. Do not guess. Read the documentation.

### Why this works so well

Modern embedding models (like OpenAI's `text-embedding-3-large` or Cohere's `embed-v3`) have learned these numeric representations by analyzing massive amounts of text. They understand that “reset password” and “forgot login” share the same underlying intent, even though they share almost no exact words.

<img width="520" height="400" alt="image" src="https://github.com/user-attachments/assets/bc708855-7743-4780-9b3c-7f472947c12d" />

However, you should not treat embedding-based search as a total replacement for traditional keyword search. Keyword systems often use **BM25**, a ranking formula that heavily rewards exact term matches. 

| Signal | What it Matches | Strength | Weakness |
| :----- | :-------------- | :------- | :------- |
| **Sparse / keyword (BM25)** | Exact or near-exact terms. | Great for names, IDs, error codes, and exact phrases. | Misses related meaning when words differ. |
| **Dense / embedding** | Semantic similarity through vectors. | Great for paraphrases and conceptual matches. | Can miss exact constraints, rare terms, and precise identifiers. |

Treat semantic search as a new signal, not a complete replacement for exact matching. 

---

## Practical Chunking Strategies

Now that we can compare short pieces of text, we run into a practical problem: real products usually search full documents. A long PDF might contain ten completely unrelated ideas. If you embed the entire document into a single vector, the resulting numbers will be a blurry average of all those ideas. It will fail to match specific questions.

**Chunking** is the process of splitting documents into smaller pieces before generating embeddings. This gives the search system a focused unit of text to retrieve.

<img width="956" height="179" alt="image" src="https://github.com/user-attachments/assets/8ff176b9-dfa7-435e-af21-64f16beab76c" />

A good chunk is large enough to contain useful context, but small enough to stay focused on one topic. 

When you tune chunk sizes, remember that models count text in **tokens**, not characters. A token is a sub-word unit that the model uses to process text (roughly 3/4 of an English word). Every embedding model has a maximum token limit it can accept in a single request.

Many chunkers use **overlap**, which means repeating a small amount of text from the end of one chunk at the beginning of the next. Overlap acts as a safety net in case a crucial sentence is accidentally sliced in half at a boundary.

Common chunking strategies:

| Strategy | How it Works | Use When | Risk |
| :------- | :----------- | :------- | :--- |
| **Fixed-size with overlap** | Split every N words or tokens, repeat a small overlap. | Quick baseline, simple docs. | Can ruthlessly cut through headings, tables, or code blocks. |
| **Recursive splitting** | Try paragraphs, then sentences, then words until size fits. | General unstructured text. | Still structure-blind for complex documents. |
| **Structural splitting** | Split strictly by headings, Markdown sections, or HTML. | Policies, manuals, code docs. | Requires robust parsing logic. |
| **Semantic chunking** | Split dynamically when the semantic meaning shifts. | Essays, transcripts, mixed text. | Slower, harder to tune, and nondeterministic. |
| **Agentic chunking** | Use an LLM to decide chunk boundaries. | High-value corpora where quality matters deeply. | High cost, high latency. |
| **Late chunking** | Pass the whole document to a model, then generate vectors for specific spans based on global context. | Long documents where chunks need broader context. | Complex implementation tied to specific model architectures. |

Start with the simplest approach. Here is a minimal baseline chunker:

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

This is not the final chunker for a serious production system, but it gives you a working baseline to evaluate before you add complexity. (For production, you can explore open-source libraries like `langchain-text-splitters`, LlamaIndex splitters, or `superlinked/chunking-research`).

### The Importance of Metadata
Text alone is rarely enough. When a document has structure, you must preserve it. Headings, URLs, source IDs, and permissions matter just as much as the text itself. We store these as **metadata**,structured dictionary fields attached to each chunk. 

In multi-customer systems, a critical piece of metadata is the **tenant** (the account or workspace ID). This controls who is allowed to see the chunk and prevents disastrous data leaks between customers.

```python
metadata = {
    "source_file": "refund_policy.md",
    "section": "Refund Window",
    "tenant_id": "customer_994",
    "chunk_index": 3,
}
```

### Chunking Trade-offs
Before you tune chunk sizes based on intuition, understand the physical trade-offs:
*   **Small chunks** improve precision but risk losing the surrounding context.
*   **Large chunks** preserve context but can dilute the vector, retrieving irrelevant text.
*   **Overlap** prevents boundary loss but increases your storage costs and can result in duplicate retrievals.

---

## Choosing Embedding Models

Once you have defined your chunks, you must choose the model that will turn them into vectors. There is no universal "best" embedding model. There is only the best model for your specific latency targets, cost budget, and data privacy rules.

The primary engineering split is deployment style:
1.  **Hosted APIs:** A provider (like OpenAI or Google) runs the model. You send text; they return vectors. This reduces operational work but means your data leaves your infrastructure.
2.  **Local Inference:** You run the model on your own servers. This gives you total control over privacy and costs, but you own the responsibility for scaling, batching, and hardware provisioning.

Before choosing, clearly define your **corpus** (the full set of documents you plan to search). Are you embedding code, legal text, or multilingual chat logs? How many documents do you have? 

| Category | Examples | Good Fit | Tradeoff |
| :------- | :------- | :------- | :------- |
| **Hosted general models** | OpenAI `text-embedding-3-small`, Google `gemini-embedding-2` | Fast integration, managed scaling, strong baseline quality. | Data leaves your app boundary; provider pricing and versions can change. |
| **Hosted retrieval-specialized models** | Voyage AI embedding models, Cohere Embed v3. | Search-heavy products, domain-specific retrieval. | Vendor dependency; you must evaluate on your own data. |
| **Open and local models** | BGE, E5, `nomic-embed-text`, `mxbai-embed-large`, Qwen | Strict privacy, high volume offline use, cost control. | You manage serving, CPU/GPU scaling, and model upgrades. |
| **Small local baselines** | `sentence-transformers/all-MiniLM-L6-v2`. | Learning, local prototypes, small applications. | Lower quality on difficult or domain-specific queries. |

You will often see practitioners mention the **MTEB** (Massive Text Embedding Benchmark) leaderboard hosted on Hugging Face. Use MTEB to shortlist candidates, but do not make final decisions based on it. MTEB tests general tasks; it does not represent your specific product, vocabulary, or users. 

**Engineering rule:** Embeddings tightly couple your search system to a specific model. If the embedding model changes, the mathematical meaning of your vectors changes. You cannot safely mix vectors from different models in the same database index. If you switch models, you must re-embed your entire corpus.

---

## Embedding Generation Best Practices

Generating an embedding looks like a simple API call, but in production, it is a heavy ingestion workload.

Here is the standard hosted API shape (using OpenAI):

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

# Extract the vectors
vectors = [item.embedding for item in response.data]
```

And here is the open-source local equivalent using the `sentence-transformers` library:

```python
from sentence_transformers import SentenceTransformer

# Downloads and loads the model locally
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

vectors = model.encode(
    [
        "Refunds are available within 30 days.",
        "Enterprise accounts require manual approval.",
    ],
    batch_size=32,
    normalize_embeddings=True, # Forces vectors to length 1
)
```

In production, you will face two primary limits: **rate limits** (how many requests/tokens you can process per minute) and **batch limits** (how many chunks you can send in one single API call).

To handle these gracefully, treat embedding generation like a robust data pipeline:
*   **Batch requests** to improve network throughput and utilize GPU hardware efficiently.
*   **Cache embeddings** based on a hash of the chunk's text, the model version, and the chunking strategy. Never pay to re-embed text that hasn't changed.
*   **Normalize consistently** if your database expects unit-length vectors.
*   **Store raw text alongside the vector.** If you only store numbers, you cannot debug why a chunk was retrieved.
*   **Use async background jobs** for large document uploads instead of blocking the user's web request.

---

## Vector Databases and Storage Options

Once you have generated your vectors, you need a place to store them so they can be queried. A **vector database** stores these lists of numbers and performs nearest-neighbor searches. 

When a user asks a question, you embed their question into a "query vector." The database then scans the stored vectors to find the ones mathematically closest to the query vector. 

At a massive scale, comparing the query against millions of vectors exactly is too slow. Most production vector databases use **ANN (Approximate Nearest Neighbor)** indexes (like HNSW). ANN algorithms trade a tiny fraction of accuracy for a massive increase in search speed. 

Vector stores organize data into **collections** (or namespaces) and heavily rely on **metadata filters**. Metadata filtering allows you to restrict a search *before* calculating vector distances (e.g., "Only search vectors where `tenant_id == customer_994`"). 

This is where an important engineering distinction appears: **FAISS is a vector search library, not a full database**. It is excellent at nearest-neighbor search over dense vectors, but it does not natively solve document storage, metadata filtering, multi-tenant access control, deletes, background reindexing, or operational scaling. If you choose FAISS directly, those concerns become your application's responsibility.

Here is how to think about the current landscape:

| Option Type | Examples | Good Fit | Watch For |
| :---------- | :------- | :------- | :-------- |
| **Local libraries** | FAISS, hnswlib, LanceDB. | Experiments, local apps, on-device search. | You handle persistence, filtering, and scaling manually. |
| **Prototype stores** | ChromaDB. | Learning, notebooks, fast local development. | Validate operational needs before relying on it at high scale. |
| **Postgres-based** | pgvector, Supabase. | Apps already heavily using Postgres, relational metadata. | Performance drops at extreme scales (>50M vectors) without heavy tuning. |
| **Dedicated vector DBs** | Qdrant, Milvus, Weaviate, Pinecone. | Production retrieval, high throughput, massive scale. | Introduces a new database to monitor and maintain. |
| **Search platforms** | Elasticsearch, OpenSearch, MongoDB Atlas. | Teams already using these enterprise platforms. | Vector search might just be an add-on feature; check limitations. |
| **Specialized storage** | Astra DB, LEANN. | Highly specific compression or serverless needs. | Evaluate maturity and ecosystem fit. |

The practical community pattern is boring but correct:
*   **If you want the smallest educational baseline**, use FAISS. It exposes the retrieval mechanics directly: vectors go in, nearest neighbors come out.
*   **If you want offline, embedded, or on-device retrieval**, FAISS is still a strong choice because it is lightweight and fast.
*   **If your app already relies on PostgreSQL**, use `pgvector`. It keeps your infrastructure simple and gives you metadata and operational tooling in one place.
*   **If you need a dedicated, highly scalable production system**, Qdrant is currently the community's top recommendation for balancing performance and operational simplicity.
*   **If you want a convenience layer around local vector search**, ChromaDB can be useful, but it hides some of the lower-level mechanics that are worth understanding first.

What breaks first with FAISS is usually not search quality. It is **operations**. As the corpus grows, you must think about how to rebuild indexes, keep metadata in sync, apply access-control filters safely, and handle updates without turning your ingestion pipeline into a brittle maintenance job.

Do not choose a vector database solely based on a vendor's marketing benchmark. The right answer in production is the one your team can operate with confidence.

---

## Basic Semantic Search Implementation

Let's assemble the smallest working semantic search loop using only local components: chunk the text, embed it, store the vectors in FAISS, and query the index.

FAISS is a good teaching tool because it makes the mechanics visible. You explicitly:

1. create vectors
2. choose a similarity metric
3. add the vectors to an index
4. map search results back to your stored text

That is the core of semantic retrieval. Higher-level vector databases automate parts of this, but they do not change the underlying flow.

FAISS is also a good fit for small local systems because it keeps the moving parts honest. You can see exactly where the vectors live, how similarity is computed, and how results are mapped back to text. That clarity is valuable in Chapter 5. Later, when you move to larger or more dynamic corpora, you can decide whether the extra operational burden is still worth it.

*Prerequisites: `pip install sentence-transformers faiss-cpu`*

```python
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

documents = {
    "refund_policy": "Customers can request a refund within 30 days of purchase. Refunds over 100 dollars require manager approval.",
    "upload_help": "If file upload fails, check the file size and supported formats. The maximum upload size is 25 MB.",
}

def chunk_words(text: str, chunk_size: int = 15, overlap: int = 5) -> list[str]:
    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
        if words[start:start + chunk_size]
    ]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunks = []
for source_file, text in documents.items():
    for chunk_index, chunk in enumerate(chunk_words(text)):
        chunks.append(
            {
                "source_file": source_file,
                "chunk_index": chunk_index,
                "text": chunk,
            }
        )

texts = [item["text"] for item in chunks]
embeddings = model.encode(texts, normalize_embeddings=True)
embeddings = np.asarray(embeddings, dtype="float32")

# IndexFlatIP uses inner product. Because the vectors are normalized,
# this becomes cosine similarity.
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# Persist both the FAISS index and the chunk metadata.
# FAISS stores vectors; your application still has to store the text and metadata.
faiss.write_index(index, "support_docs.index")
Path("support_docs.metadata.json").write_text(
    json.dumps(chunks, indent=2),
    encoding="utf-8",
)

def search(query: str, top_k: int = 2) -> list[dict]:
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")

    scores, indices = index.search(query_vector, k=top_k)

    matches = []
    for score, row_id in zip(scores[0], indices[0]):
        item = chunks[row_id]
        matches.append(
            {
                "source_file": item["source_file"],
                "chunk_index": item["chunk_index"],
                "score": float(score),
                "text": item["text"],
            }
        )
    return matches

print("Query: Can I get my money back after buying?")
for match in search("Can I get my money back after buying?"):
    print(
        f"[{match['source_file']}] "
        f"score={match['score']:.4f} -> {match['text']}"
    )
```

There are four ideas to notice in this code:

*   `model.encode(...)` turns each chunk into a vector.
*   `IndexFlatIP` performs exact nearest-neighbor search using inner product.
*   Because we normalized the vectors first, a **higher** score means a closer semantic match.
*   FAISS stores row positions, so your code needs a second structure to map a row ID back to the original text and metadata.

That last point matters. FAISS does not know that your vectors represent language. It only sees arrays of numbers. If you want cosine-style ranking, you must normalize consistently before both indexing and querying.

For a tiny local system, this is enough. For a larger product, you would add filters, multi-tenant access control, background ingestion, update handling, and possibly an ANN index instead of exact search. But the retrieval logic would still be the same.

---

## Evaluation Metrics for Retrieval

How do you know if your search is actually working? Do not evaluate retrieval by asking an LLM to generate an answer and judging if the text sounds smart. That mixes retrieval quality with generation quality. You must evaluate the retrieval layer independently.

To do this, create a small set of evaluation cases with known good answers:

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

In standard information-retrieval metrics, **`@k`** means "look only at the top `k` returned results." Ranking matters deeply because downstream prompts usually only have room for the first few chunks.

| Metric | Question it Answers |
| :----- | :------------------ |
| **Hit Rate@k** | Did at least *one* relevant chunk appear in the top k results? |
| **Recall@k** | What fraction of *all* relevant items appeared in the top k? |
| **MRR** | (Mean Reciprocal Rank) How high up was the *first* relevant result? |
| **nDCG@k** | Were the *most* relevant results ranked near the very top? |

Here is how you can practically implement Hit Rate and MRR to evaluate your system:

```python
def hit_rate_at_k(results: list[list[str]], expected: list[set[str]], k: int) -> float:
    hits = 0
    for retrieved_sources, relevant_sources in zip(results, expected):
        # Check if there is an intersection between the top K and expected sources
        if set(retrieved_sources[:k]) & relevant_sources:
            hits += 1
    return hits / len(expected)

def mrr(results: list[list[str]], expected: list[set[str]]) -> float:
    total = 0.0
    for retrieved_sources, relevant_sources in zip(results, expected):
        for rank, source in enumerate(retrieved_sources, start=1):
            if source in relevant_sources:
                total += 1 / rank  # The higher the rank, the higher the score (e.g., Rank 1 = 1.0, Rank 2 = 0.5)
                break
    return total / len(expected)
```

**nDCG (Normalized Discounted Cumulative Gain)** is a slightly more advanced metric used when you have graded relevance (e.g., a chunk isn't just "hit or miss," but can be "perfect," "partially useful," or "useless"). It rewards systems that place the absolute highest-value evidence first. 

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
    return sum(values) / len(values) if values else 0.0
```

Start with 20 to 50 well-designed evaluation cases. If you can confidently label which documents should be returned, these mathematical metrics are vastly cheaper, faster, and more precise than using an expensive "LLM-as-a-judge" framework (like RAGAS or DeepEval) at this layer. We will reserve those LLM-based tools for Chapter 6, where we evaluate actual generated text.

---

## Common Pitfalls and Failure Scenarios

Embedding search fails in predictable, concrete ways. One of the most insidious failures is **embedding drift**: retrieval behavior silently changes because you altered your chunking logic or updated the source documents, but forgot to rebuild the vector index. The app still returns results, but they are suddenly much worse.

When building your pipeline, watch out for these standard failures:

| Failure | Likely Cause | Fix |
| :------ | :----------- | :-- |
| **Good document never appears.** | Bad chunking, truncation, or a missing metadata filter blocking it. | Inspect the raw text of your chunks. Verify your model's token limits. |
| **Results are semantically similar but useless.** | Dense search overweights general intent. | Add keyword search (BM25), metadata filters, or a re-ranker (Chapter 7). |
| **Exact IDs/Names are missed.** | Embeddings are weak at recognizing arbitrary symbols and exact SKUs. | Use strict metadata filters or hybrid retrieval. |
| **Top results are all duplicates.** | Too much chunk overlap, or duplicate files in the corpus. | Reduce overlap size. Run a deduplication pass before embedding. |
| **Search is unexpectedly slow.** | Large vectors, weak index settings, or skipping batching. | Ensure you are using an ANN index (like HNSW). Cache frequent queries. |
| **Users see unauthorized content.** | Missing access-control logic. | You must inject `tenant_id` into your database queries to filter securely *before* vector distances are calculated. |
| **Noisy list data dominates results.** | Unstructured tables or repeated lists overwhelm meaningful prose. | Split lists structurally during ingestion, or summarize them before embedding. |

When debugging a bad retrieval, follow this strict checklist:
1. Print the raw text of the chunks that were retrieved.
2. Verify that the document you *expected* to find was actually indexed.
3. Check if a metadata filter accidentally excluded it.
4. Try writing the query differently to see if the semantic signal was too weak.
5. Measure the fix against your entire evaluation set, not just the single failed query.

---

## Hands-On Exercise

**Goal:** Build and evaluate a local semantic document search engine over your own files.

**Requirements:**
1.  **Gather Data:** Choose 5 to 20 text documents (Markdown files, notes, or code READMEs). 
2.  **Chunking:** Write a script to split these documents into chunks. Ensure each chunk retains its source filename as metadata.
3.  **Embedding:** Use the `sentence-transformers/all-MiniLM-L6-v2` model to generate normalized embeddings for each chunk.
4.  **Storage:** Store the chunks, vectors, and metadata in a persistent local vector store (use FAISS, pgvector, or another local option).
5.  **Search Interface:** Implement a `search(query, top_k)` function that prints the retrieved text, the source document name, and the similarity score or distance.
6.  **Evaluation Data:** Create a Python list of at least 10 evaluation queries, pairing each query with the exact source document name you expect it to retrieve.
7.  **Measurement:** Run your 10 queries through the search function and calculate the `Hit Rate@3` and `MRR`.
8.  **Iteration:** Change one single variable in your pipeline (e.g., double the chunk size, or switch the embedding model to `nomic-embed-text`). Re-run your ingestion and evaluation. 
9.  **Compare:** Document whether the `Hit Rate@3` improved or degraded based on your change.

**Constraints:**
*   Keep the entire pipeline local to avoid costs and data privacy concerns.
*   Do not add an LLM answer generator yet. We are exclusively testing retrieval.
*   Do not optimize the vector index settings until your chunking strategy is stable.

By completing this exercise, you will have built the foundational engine of modern AI memory. Once you can reliably chunk, store, and evaluate semantic retrieval, RAG transforms from a guessing game into a predictable engineering problem.
