# Mini Roadmap

This mini-roadmap is a friendly, bite-sized version of our detailed [Full Roadmap](roadmap.md), guiding you straight to the good stuff. Each section is packed with **handy keyword lists** to help you search for tutorials, research papers, code snippets, and other awesome resources across the web. Check out these spots to dig deeper:

- 📚 **Academic Stuff:** Google Scholar, arXiv, Papers With Code
- 🎥 **Videos:** YouTube, Coursera, edX  
- 💻 **Code Examples:** GitHub, Hugging Face, Kaggle
- 📖 **Guides & Reads:** Official docs, tutorials, blogs
- 🤖 **AI Buddies:** Chat with ChatGPT, Claude, or any AI assistant


## 📋 Prerequisites
- **Python (4/5)**, **Git (3/5)**, **Linux (3/5)**, **SQL (2/5)**
- **Linear Algebra (3/5)**, **Probability & Statistics (3/5)**, **Calculus (2/5)**
- **ML Fundamentals (3/5)**, **Deep Learning (2/5)**
- **Tools:** Python 3.9+, CUDA GPU (RTX 3080+), Docker, Jupyter Lab, VSCode

---

# Part 1: LLM Intern 📘
*Foundation building, transformer implementation, data preparation*

## 1. [Tokenization](Tutorial/Tokenization.md)
**Learn:** tokenization, tokens, normalization, pre-tokenization, subword tokenization, byte-pair encoding (BPE), WordPiece algorithm, unigram model, SentencePiece framework, byte-level BPE, vocabulary management, context window optimization, multilingual tokenization, visual tokenization, tokenizer transplantation (TokenAdapt), text preprocessing, OOV handling, byte-level processing, special characters, emojis, code snippets, legal jargon, medical terminology, bilingual applications, token counts, GPT-4 tokenization, Llama tokenization, BERT tokenization, tiktoken, encoding, decoding, vocabulary size, tokenization efficiency  
**Tools:** Hugging Face Tokenizers, SentencePiece, tiktoken, spaCy, NLTK  
**Build:** Custom BPE tokenizer, tokenizer comparison tool, multilingual tokenizer, legal text tokenizer

## 2. [Embeddings](Tutorial/Embeddings.md)
**Learn:** word embeddings, token embeddings, Word2Vec architecture, GloVe embeddings, contextual embeddings, BERT embeddings, RoBERTa, CLIP embeddings, fine-tuning embeddings, semantic search, multimodal embeddings, ALIGN, embedding evaluation metrics, dense retrieval, sparse retrieval, vector similarity, distance metrics, dimensionality reduction, cosine similarity, euclidean distance, dot product, semantic similarity, text similarity, cross-modal search, joint embeddings, embedding spaces, vector representations, financial sentiment, domain adaptation, arXiv papers, e-commerce search, product recommendations  
**Tools:** SentenceTransformers, FAISS, Pinecone, Weaviate, Chroma, Qdrant  
**Build:** Semantic search engine, text similarity API, recommendation system, multimodal search

## 3. [Neural Networks](Tutorial/Neural_Networks.md)
**Learn:** neural networks, backpropagation, activation functions, ReLU, sigmoid, tanh, swish, GELU, gradients, loss functions, cross-entropy, MSE, regularization, L1 regularization, L2 regularization, dropout, batch normalization, layer normalization, optimization algorithms, SGD, Adam, AdamW, RMSprop, learning rate, momentum, weight decay, hyperparameter tuning, AutoML, automatic differentiation, mixed precision training, FP16, BF16, gradient clipping, vanishing gradients, exploding gradients, weight initialization, Xavier initialization, He initialization, MNIST dataset, overfitting, underfitting, validation, early stopping  
**Tools:** PyTorch, JAX, TensorFlow, Weights & Biases, Optuna, Ray Tune  
**Build:** Neural network from scratch, optimization visualizer, regularization experiments, mixed precision training

## 4. [Traditional Language Models](Tutorial/Traditional_LMs.md)
**Learn:** N-gram models, language modeling, smoothing techniques, Laplace smoothing, Good-Turing smoothing, Kneser-Ney smoothing, feedforward neural networks, recurrent neural networks (RNNs), Long Short-Term Memory (LSTM), Gated Recurrent Units (GRUs), bidirectional RNNs, multilayer RNNs, sequence-to-sequence models, encoder-decoder, attention mechanisms, beam search, perplexity, BLEU score, sequence modeling, text generation, machine translation, text summarization, sentiment analysis, time series prediction, character-level modeling, word-level modeling  
**Tools:** Scikit-learn, PyTorch RNN modules, beam search implementations  
**Build:** N-gram language model, text generator, perplexity calculator, RNN text classifier

## 5. [Transformers](Tutorial/Transformers.md)
**Learn:** transformer architecture, self-attention, multi-head attention, scaled dot-product attention, positional encodings, sinusoidal encodings, learned positional encodings, RoPE (Rotary Position Embedding), ALiBi (Attention with Linear Biases), encoder-decoder architecture, decoder-only architecture, layer normalization, residual connections, feed-forward networks, attention heads, attention patterns, Flash Attention, multi-query attention (MQA), grouped-query attention (GQA), KV cache, attention visualization, cross-attention, causal attention, masked attention, attention weights, query, key, value matrices, mini-GPT, nucleus sampling, top-k sampling, text generation, beam search  
**Tools:** PyTorch, JAX, Transformer libraries, Flash Attention implementations  
**Build:** Transformer from scratch, attention visualizer, positional encoding explorer, mini-GPT

## 6. [Data Preparation](Tutorial/Data_Preparation.md)
**Learn:** data collection, web scraping, data cleaning, data filtering, deduplication, MinHash, Locality-Sensitive Hashing (LSH), data quality assessment, contamination detection, synthetic data generation, data augmentation, privacy-preserving processing, PII detection, personally identifiable information, named entity recognition (NER), data decontamination, Common Crawl, C4 dataset, TinyStories, real estate data, quality scoring, data validation, error handling, rate limiting, data structures, incremental indexing, data pipelines, Beautiful Soup, Scrapy, regex patterns, machine learning techniques, sensitive information, training datasets, instruction datasets, quality control, data annotation  
**Tools:** Pandas, Dask, PySpark, Beautiful Soup, Scrapy, Elasticsearch, DVC  
**Build:** Web scraping pipeline, data deduplication tool, data quality scorer, synthetic data generator

---

# Part 2: LLM Scientist ⚙️
*Research-grade model development, novel architectures, theoretical advances*

## 7. [Pre-Training](Tutorial/Pre_Training.md)
**Learn:** unsupervised pre-training, causal language modeling (CLM), masked language modeling (MLM), prefix language modeling (PrefixLM), permutation language modeling (PLM), replaced token detection (RTD), span-based masking, distributed training, data parallelism, model parallelism, pipeline parallelism, training efficiency, scaling laws, curriculum learning, data scheduling, compute optimization, ZeRO optimization, gradient checkpointing, mixed precision, DeepSpeed, FairScale, Megatron-LM, Colossal-AI, Slurm, Kubernetes, multi-node training, checkpoint management, loss monitoring, training instabilities, recovery mechanisms, mathematical reasoning, training throughput, memory usage, convergence speed, hardware configurations  
**Tools:** DeepSpeed, FairScale, Megatron-LM, Colossal-AI, Slurm, Kubernetes  
**Build:** Mini-LLM pre-training, distributed training setup, curriculum learning, training efficiency optimizer

## 8. [Post-Training Datasets](Tutorial/Post_Training_Datasets.md)
**Learn:** instruction datasets, chat templates, conversation formatting, synthetic data generation, quality control, filtering strategies, multi-turn conversations, Hugging Face datasets, Alpaca format, ShareGPT format, data annotation, quality scoring, system prompts, user messages, bot messages, special tokens, role-playing models, internal thoughts, conversation flow, domain-specific patterns, dataset composition, capability-specific optimization, response quality, instruction following, conversation history, context management, data curation, annotation guidelines, validation systems, dataset integrity, bias detection  
**Tools:** Hugging Face Datasets, Alpaca, ShareGPT, data annotation platforms  
**Build:** Instruction dataset creator, chat template designer, synthetic conversation generator, dataset quality evaluator

## 9. [Supervised Fine-Tuning](Tutorial/Supervised_Fine_Tuning.md)
**Learn:** parameter-efficient fine-tuning (PEFT), LoRA (Low-Rank Adaptation), QLoRA, adapters, full fine-tuning, instruction tuning, chat model training, domain adaptation, continual learning, catastrophic forgetting, model merging, model composition, SLERP, TIES-Merging, DARE, multi-task models, CodeLlama, code generation, resource optimization, performance retention, consumer GPU constraints, hyperparameter optimization, memory-efficient training, 4-bit quantization, gradient checkpointing, FSDP (Fully Sharded Data Parallel), specialized models, task-specific models  
**Tools:** PEFT, Hugging Face Transformers, Unsloth, DeepSpeed, FSDP  
**Build:** Parameter-efficient fine-tuning, domain-specific model, instruction-following model, model merging toolkit

## 10. [Preference Alignment](Tutorial/Preference_Alignment.md)
**Learn:** reinforcement learning fundamentals, deep reinforcement learning, policy optimization, Proximal Policy Optimization (PPO), Direct Preference Optimization (DPO), rejection sampling, Reinforcement Learning from Human Feedback (RLHF), reward model training, human preferences, helpfulness, harmlessness, honesty, Constitutional AI, AI feedback, safety evaluation, alignment evaluation, preference datasets, KTO (Kahneman-Tversky Optimization), win rate, safety benchmarks, alignment tax, model capabilities, preference training, automated assessment, human assessment, principle-based alignment, self-critique systems, response revision, defined principles  
**Tools:** TRL (Transformer Reinforcement Learning), Ray RLlib, safety benchmarks  
**Build:** Reward model training, DPO implementation, RLHF pipeline, constitutional AI

## 11. [Model Architectures](Tutorial/Model_Architecture_Variants.md)
**Learn:** Mixture of Experts (MoE), sparse architectures, state space models, Mamba architecture, RWKV, sliding window attention, long context architectures, Longformer, BigBird, hybrid architectures, GraphFormers, graph-based LLMs, novel architectures, efficient architecture search, sparse attention, selective state space, gating network, expert networks, load balancing, memory optimization, computation efficiency, interpolation, extrapolation, long document processing, architecture search frameworks, hybrid components, convolution, attention mechanisms, architecture variants, performance benchmarks  
**Tools:** MoE implementations, Mamba, RWKV, Longformer, architecture search frameworks  
**Build:** MoE model, state space model, long context model, hybrid architecture

## 12. [Reasoning](Tutorial/Reasoning.md)
**Learn:** reasoning fundamentals, System 2 thinking, Chain-of-Thought (CoT), tree-of-thoughts, advanced prompting, reinforcement learning for reasoning (RL-R), process reward models (PRM), step-level rewards, STEP-RLHF, Group Relative Policy Optimization (GRPO), self-reflection, self-consistency loops, deliberation budgets, test-time compute scaling, planner-worker architecture, plan-work-solve decoupling, synthetic reasoning data, bootstrapped self-training, Monte Carlo Tree Search (MCTS), symbolic logic integration, verifiable domains, automatic grading, multi-stage training, curriculum training, DeepSeek-R1, OpenAI o1/o3, Gemini-2.5, thinkingBudget, reasoning chains, mathematical reasoning, code generation, logical reasoning, multi-hop queries, problem decomposition, step-by-step rationales, reasoning traces, confidence scoring, quality filtering  
**Tools:** CoT frameworks, ReAct implementations, reasoning benchmarks (GSM8K, MATH, HumanEval)  
**Build:** Chain-of-thought trainer, tool-using agent, mathematical reasoning model, multi-step reasoning system

## 13. [Evaluation](Tutorial/Evaluation.md)
**Learn:** benchmarking, standardized benchmarks, MMLU, GSM8K, HumanEval, BigBench, human evaluation, crowdsourcing, automated evaluation, LLM-as-judge, bias testing, safety testing, fairness testing, performance monitoring, statistical analysis, accuracy, F1 score, BLEU, ROUGE, win rate, evaluation frameworks, chatbot comparison, quality assessment, annotation guidelines, crowdsourcing mechanisms, comparative evaluation, quality metrics, bias detection, toxicity detection, BOLD dataset, RealToxicityPrompts, fairness frameworks, mitigation recommendations, responsible AI evaluation, custom benchmarks, domain-specific evaluation, benchmark creation, validation  
**Tools:** Evaluation frameworks, statistical analysis, crowdsourcing platforms  
**Build:** Comprehensive evaluation suite, custom benchmark creator, human evaluation platform, model comparison dashboard

---

# Part 3: LLM Engineer 🚀
*Production systems, RAG, agents, deployment, ops & security*

## 14. [Quantization](Tutorial/Quantization.md)
**Learn:** quantization fundamentals, quantization theory, post-training quantization (PTQ), quantization-aware training (QAT), GGUF format, llama.cpp implementation, GPTQ, AWQ, SmoothQuant, ZeroQuant, integer quantization, INT4 quantization, INT8 quantization, calibration, sparsity, hardware-specific optimization, mobile deployment, edge deployment, quantization quality assessment, performance trade-offs, memory optimization, inference acceleration, model compression, BitsAndBytes, ONNX, TensorRT, consumer GPU deployment, resource-constrained environments, quality vs speed analysis  
**Tools:** llama.cpp, GPTQ, AWQ, BitsAndBytes, ONNX, TensorRT  
**Build:** Quantization toolkit, mobile LLM deployer, inference optimizer, quality vs speed analyzer

## 15. [Inference Optimization](Tutorial/Inference_Optimization.md)
**Learn:** Flash Attention, memory optimization, KV cache implementation, KV cache management, test-time preference optimization (TPO), compression methods, speculative decoding, parallel sampling, dynamic batching, continuous batching, multi-GPU inference, multi-node inference, PagedAttention, advanced memory management, vLLM, TensorRT-LLM, DeepSpeed-Inference, Triton, CUDA optimization, high-throughput inference, latency optimization, production systems, draft models, verifiers, speedup gains, quality evaluation, multi-model coordination, memory-efficient attention, long sequences, resource allocation, load balancing, scaling  
**Tools:** vLLM, TensorRT-LLM, DeepSpeed-Inference, Triton, CUDA optimization  
**Build:** High-throughput inference server, dynamic batching system, speculative decoding, multi-model serving

## 16. [Model Enhancement](Tutorial/Model_Enhancement.md)
**Learn:** context window extension, YaRN (Yet another RoPE extensioN), position interpolation, model merging, ensembling, knowledge distillation, model compression, continual learning, adaptation, self-improvement, meta-learning, teacher-student training, context extension methods, long-text data, information recall, model degradation, recovery strategies, model composition, performance improvement, ensembling systems, specialized models, unified systems, compressed models, mobile deployment, user feedback, interactions, ongoing enhancement  
**Tools:** YaRN implementations, merging frameworks, distillation pipelines, continual learning systems  
**Build:** Context window expander, model merger, knowledge distillation system, continual learning pipeline

## 17. [Security & Responsible AI](Tutorial/Securing_LLMs.md)
**Learn:** OWASP LLM Top 10, attack vectors, prompt injection attacks, prompt injection defense, data leaking, prompt leaking prevention, jailbreaking techniques, jailbreak mitigation, training data poisoning, backdoor attacks, model theft prevention, fairness, bias detection, bias mitigation strategies, responsible AI development, personal information masking, reconstruction methods, privacy protection, AI governance, compliance, GDPR, CCPA, AI regulations, input sanitization, output filtering, differential privacy, federated learning, red teaming frameworks, vulnerability assessment, attack simulation, security controls, defensive layers, content moderation, prompt sanitization, security validation, guardrail systems, safety controls  
**Tools:** Input sanitization, Output filtering, differential privacy, federated learning, red teaming frameworks  
**Build:** LLM security scanner, guardrail system, bias detection tool, privacy-preserving LLM

## 18. [Running LLMs](Tutorial/Running_LLMs.md)
**Learn:** LLM APIs, API integration, memory-enabled chatbots, open-source models, prompt engineering, structured outputs, local deployment, interactive demos, production servers, serving LLMs, REST APIs, concurrent users, test-time autoscaling, batching, model deployment, streaming responses, real-time applications, application architecture, scalability, FastAPI, Flask, Streamlit, Gradio, WebSockets, rate limiting, authentication, authorization, conversation history, context management, conversational interfaces, prompt management, containerization, Docker, Kubernetes, load balancing, resource management, monitoring, scaling capabilities, multi-modal applications, unified API services  
**Tools:** FastAPI, Flask, Streamlit, Gradio, Docker, Redis, WebSockets  
**Build:** LLM-powered chatbot, API gateway, streaming response system, multi-modal assistant

## 19. [RAG](Tutorial/RAG.md)
**Learn:** Retrieval Augmented Generation, document ingestion, data sources, chunking strategies, document processing, embedding models, vector representations, vector databases, storage solutions, retrieval implementation, retrieval optimization, RAG pipeline, RAG architecture, Graph RAG, knowledge graphs, knowledge graph construction, knowledge graph optimization, Intelligent Document Processing (IDP), advanced retrieval strategies, hybrid search, BM25, semantic search, reranking algorithms, query enhancement, multi-turn conversational RAG, agentic RAG, query planning, multi-step reasoning, result synthesis, follow-up questions, context management, document updates, incremental indexing, scaling, caching, monitoring, PDFs, web pages, databases, LlamaIndex, LangChain, Haystack, Neo4j, query expansion, evaluation metrics, retrieval accuracy  
**Tools:** LangChain, LlamaIndex, Haystack, Pinecone, Weaviate, Chroma, Neo4j  
**Build:** Enterprise RAG system, multi-modal RAG, conversational RAG, graph RAG implementation

## 20. [Agents](Tutorial/Agents.md)
**Learn:** function calling, tool usage, agent implementation, agent architecture, planning systems, reasoning, agentic RAG integration, multi-agent orchestration, multi-agent coordination, autonomous task execution, safety and control, agent systems, ReAct, planning, tool use, multi-agent systems, LangGraph, AutoGen, CrewAI, function calling APIs, external tool integration, financial market analysis, data collection, sentiment analysis, technical analysis, synthesis, inter-agent communication, task coordination, error handling, safety constraints, smart home devices, device control, natural language command processing, validation, user preferences, ambiguous commands, programming assistant, code generation, debugging, research, web search, documentation lookup, code execution, iterative refinement, multi-step problem solving, workflow automation, business processes, task decomposition, workflow execution, recovery mechanisms, human oversight, approval workflows  
**Tools:** LangGraph, AutoGen, CrewAI, function calling APIs, external tool integration  
**Build:** Multi-agent system, code generation agent, research assistant, workflow automation

## 21. [Multimodal](Tutorial/Multimodal.md)
**Learn:** multimodal LLMs, text input, audio input, audio output, images, transfer learning, pre-trained models, multimodal transformers, vision-language models, CLIP, LLaVA, GPT-4V, multimodal attention, feature fusion, image captioning, Visual Question Answering (VQA), text-to-image generation, multimodal chatbots, multimodal agents, joint image-text representations, audio processing, speech integration, document understanding, OCR, cross-modal attention, OpenCV, Pillow, torchaudio, visual reasoning, image processing, question answering interfaces, document analysis, PDF processing, code screenshot analysis, media preprocessing, Stable Diffusion, prompt engineering, negative prompts, parameter tuning, image generation, quality evaluation, e-commerce chatbots, conversation flows, deployment scenarios  
**Tools:** CLIP, LLaVA, Whisper, OpenCV, Pillow, torchaudio, OCR systems  
**Build:** Vision-language assistant, document analysis system, code screenshot analyzer, audio-visual assistant

## 22. [LLMOps](Tutorial/LLMOps.md)
**Learn:** model versioning, CI/CD pipelines, monitoring, observability, cost optimization, deployment strategies, rollback, Hugging Face Hub integration, model card creation, model sharing, version control, LLM observability tools, debugging techniques, monitoring techniques, Docker, OpenShift, dependency management, containerization, Apache Spark, LLM inference, model registry management, resource management, MLflow, Weights & Biases, Kubeflow, Kubernetes, Terraform, Prometheus, Grafana, custom metrics, GitHub Actions, automated testing, deployment processes, automated workflows, performance metrics, real-time dashboards, alerting systems, performance tracking, A/B testing, experimentation frameworks, statistical analysis, model comparison, prompt optimization, experimentation platforms, data-driven decisions, cost tracking, resource allocation, automatic scaling, demand-based scaling  
**Tools:** MLflow, Weights & Biases, Kubeflow, Docker, Kubernetes, Prometheus, Grafana  
**Build:** ML pipeline automation, model monitoring dashboard, A/B testing framework, cost optimization system
