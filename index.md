---
title: "Home"
nav_order: 0
---
# LLMs: From Foundations to Production

[![Tutorial Status](https://img.shields.io/badge/Status-In_Progress-yellow)](https://img.shields.io/badge/Status-In_Progress-yellow)

A hands-on tutorial series for mastering Large Language Models (LLMs) – featuring practical examples, code implementations, and real-world applications. This series takes you from foundational concepts to building production-ready LLM applications.

## About This Tutorial Series
This comprehensive tutorial series is designed to provide practical, hands-on experience with LLM development and deployment. Each tutorial combines theoretical concepts with practical implementations, real-world examples, and coding exercises.

## Prerequisites
- Basic Python programming
  - [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
  - [Real Python Tutorials](https://realpython.com/)
- Mathematics fundamentals
    - [3Blue1Brown Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
    - [3Blue1Brown Essence of Calculus](https://www.3blue1brown.com/topics/calculus)
    - [MIT Probability Course](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/)
- Basic understanding of machine learning concepts
  - [Stanford CS229: Machine Learning](https://cs229.stanford.edu/)

Don't worry if you're not an expert in these areas - we'll review key concepts as needed throughout the tutorials.

---

# Roadmap

![image](https://github.com/user-attachments/assets/8c634f07-a928-43f0-a79c-5651c12678a1)

- [Intro to Large Language Models](Foundations/Intro.md)
  - Fundamentals of Language Models
  - LLM Capabilities and Applications
- [Tokenization](Foundations/Tokenization.md)
  - Understanding Tokenization Fundamentals
  - BPE Tokenization
  - Working with Hugging Face Tokenizers
  - Building Custom Tokenizers
  - GPT Tokenization Approach
  - Multilingual Tokenization Strategies
- [Embeddings](Foundations/Embeddings.md)
  - Word and Token Embeddings
  - Word2Vec Architecture
  - GloVe Embeddings
  - Contextual Embeddings
  - Fine-tuning LLM Embeddings
  - Semantic Search Implementation

![image](https://github.com/user-attachments/assets/42dd0c30-a1f3-4802-a3cd-0dab17a86122)

- [Neural Network Foundations for LLMs](Neural_Networks/Neural_Networks.md)
  - Neural Network Basics
  - Activation Functions, Gradients, and Backpropagation
  - Loss Functions and Regularization Strategies  
  - Optimization Algorithms and Hyperparameter Tuning
- [Traditional Language Models](Neural_Networks/Traditional_LMs.md)
  - N-gram Language Models and Smoothing Techniques
  - Feedforward Neural Language Models
  - Recurrent Neural Network Language Models
    - Long Short-Term Memory (LSTM) Networks
    - Gated Recurrent Units (GRUs)
    - Bidirectional and Multilayer RNNs
- [The Transformer Architecture](Neural_Networks/Transformers.md)
  - Attention Mechanisms and Self-Attention
  - Multi-Head Attention and Positional Encodings  
  - Transformer Encoder and Decoder Stacks
  - Residual Connections and Layer Normalization
  - Implementing the Transformer from Scratch

![image](https://github.com/user-attachments/assets/b19f699f-87bf-4120-9677-9d459b51ea71)

- [Data Preparation](Training/Data_Preparation.md)
  - LLM Training Data Collection
  - Text Cleaning for LLMs
  - Data Filtering and Deduplication
  - Creating Training Datasets
  - Dataset Curation and Quality Control
  - Dataset Annotation Workflows
  - Hugging Face Hub Dataset Management
  - Data Preparation Techniques for Large-Scale NLP Applications
- [Pre-Training Large Language Models](Training/Pre_Training.md)
  - Model Architecture Selection
  - Unsupervised Pre-Training Objectives
    - Masked Language Modeling (MLM)
    - Permutation Language Modeling (PLM)
    - Replaced Token Detection (RTD)
    - Span-based Masking
    - Prefix Language Modeling
  - Efficient Pre-Training Techniques
    - Dynamic Masking and Whole Word Masking
    - Large Batch Training and Learning Rate Scheduling
    - Curriculum Learning
    - Progressive Training Strategies
  - Training Infrastructure
    - Distributed Training Setup
    - Mixed Precision Training
    - Multi-device Optimization
  - Training Optimization
    - Weight Initialization
    - AdamW Optimizer
    - Learning Rate Scheduling
  - Precision Formats
    - FP16/BF16 Training
    - FP8 Optimization
  - Distributed Training
    - Data Parallel Training
    - ZeRO Optimization
    - Distributed Data Processing
  - Scaling Laws and Model Architecture Variants
- [Post-Training Datasets](Training/Post_Training_Datasets.md)
  - Dataset Storage and Chat Templates
  - Generating Synthetic Training Data
  - Dataset Augmentation Techniques
  - Quality Control and Filtering
- [Supervised Fine-Tuning](Training/Supervised_Fine_Tuning.md)
  - Post-Training Techniques
  - Parameter Efficient Fine-Tuning (PEFT)
  - LoRA Implementation
  - Chat Model Fine-tuning
  - Distributed Fine-tuning
- [Preference Alignment](Training/Preference_Alignment.md)
  - Reinforcement Learning Fundamentals
  - Deep Reinforcement Learning for LLMs
  - Policy Optimization Methods
  - Proximal Policy Optimization (PPO)
  - Direct Preference Optimization (DPO)
  - Rejection sampling
- [Model Architecture Variants](Training/Model_Architecture_Variants.md)
  - Mixture of Experts (MoE)
  - Sparse Architectures
  - Mamba Architecture
  - Sliding Window Attention Models
  - Hybrid Transformer-RNN Architectures
  - GraphFormers and Graph-based LLMs
- [Reasoning](Training/Reasoning.md)
  - Reasoning Fundamentals
  - Chain of Thought
  - Group Relative Policy Optimization (GRPO)
- [Model Evaluation](Training/Evaluation.md)
  - Benchmarking LLM Models
  - Assessing Performance (Human evaluation)
  - Bias and Safety Testing

![image](https://github.com/user-attachments/assets/81b2e913-2b36-4cbe-a2b7-549e04720b6e)

- [Quantization](Deployment_Optimization/Quantization.md)
  - Quantization Fundamentals
  - Post-Training Quantization (PTQ)
  - Quantization-Aware Training (QAT)
  - GGUF Format and llama.cpp Implementation
  - Advanced Techniques: GPTQ and AWQ
  - Integer Quantization Methods
  - Modern Approaches: SmoothQuant and ZeroQuant
- [Inference Optimization](Deployment_Optimization/Inference_Optimization.md)
  - Flash Attention
  - KV Cache Implementation
  - Test-Time Preference Optimization (TPO)
  - Compression Methods to Enhance LLM Performance
- [Running LLMs](Deployment_Optimization/Running_LLMs.md)
  - Using LLM APIs
  - Building Memory-Enabled Chatbots
  - Working with Open-Source Models
  - Prompt Engineering
  - Structured Outputs
  - Deploying Models Locally
  - Creating Interactive Demos
  - Setting Up Production Servers
  - Serving Open Source LLMs in a Production Environment
  - Developing REST APIs
  - Managing Concurrent Users
  - Test-Time Autoscaling
  - Batching for Model Deployment

![image](https://github.com/user-attachments/assets/f37130fc-850b-4e12-a9d0-a765269cddc8)

- [Retrieval Augmented Generation](Advanced/RAG.md)
  - Ingesting documents
  - Chunking Strategies
  - Embedding models
  - Vector databases
  - Retrieval Implementation
  - RAG Pipeline Building
  - Graph RAG Techniques
  - Constructing and Optimizing Knowledge Graphs
  - Intelligent Document Processing (IDP) with RAG
- [Tool Use & AI Agents](Advanced/Agents.md)
  - Function Calling and Tool Usage
  - Agent Implementation
  - Planning Systems
  - Agentic RAG
  - Multi-agent Orchestration
- [Text-to-SQL Systems](Advanced/Text_to_SQL.md)
  - Fundamentals of Text-to-SQL 
  - Few-Shot Prompting Techniques
  - In-Context Learning and Self-Correction
  - Schema-Aware Approaches
  - Fine-Tuning Strategies for SQL Generation
  - Hybrid Neural-Symbolic Methods
  - Benchmarking and Evaluation
- [Multimodal](Advanced/Multimodal.md)
  - Working with Multi-Modal LLMs, Including Text, Audio Input/Output, and Images
  - Transfer Learning & Pre-trained Models
  - Multimodal Transformers
  - Vision-Language Models
  - Multimodal Attention
  - Feature Fusion
  - Image Captioning
  - Visual QA Systems
  - Text-to-Image Generation
  - Multimodal Chatbots
  - Joint Image-Text Representations
- [Securing LLMs](Advanced/Securing_LLMs.md)
  - Prompt Injection Attacks
  - Data/Prompt Leaking
  - Jailbreaking Techniques
  - Training Data Poisoning
  - Backdoor Attacks
  - Model Theft Prevention
  - Fairness in LLMs
  - Bias Detection and Mitigation
  - Responsible AI Development
  - Personal Information Masking
  - Reconstruction Methods
- [Large Language Model Operations (LLMOps)](Advanced/LLMOps.md)
  - Hugging Face Hub Integration
    - Model Card Creation
    - Model Sharing
    - Version Control
  - LLM Observability Tools
  - Techniques for Debugging and Monitoring
  - Docker, OpenShift, CI/CD
  - Dependency Management and Containerization
  - Apache Spark usage for LLM Inference
- [Model Enhancement](Advanced/Model_Enhancement.md)
  - Context Window Expansion
  - Model Merging
  - Knowledge Distillation 


---

## How to Follow Along
1. Follow tutorials sequentially
2. Complete the coding exercises
3. Build the suggested projects
4. Experiment with the provided examples

## Contributing
We welcome contributions! If you'd like to:
- Fix my mistakes
- Improve existing content
- Share your implementations

Please submit a pull request or open an issue.

## Community Support
- Join our Telegram Channel for discussions
- Check out the Issues section for help
- Share your implementations in Discussions

## Acknowledgments
Thanks to all contributors and the AI/ML community for their valuable input and code contributions.

---

Let's start building with LLMs! 🚀
