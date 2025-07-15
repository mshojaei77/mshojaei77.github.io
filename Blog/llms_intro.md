---
title: "Intro to LLMs"
nav_order: 1
parent: Blog
layout: default
---

## Large Language Models - An Up‑to‑Date Pocket Guide

**Expanded summary of "A Comprehensive Overview of Large Language Models"**

Everything an LLM touches is grouped into seven branches - Pre‑Training, Fine‑Tuning, Efficiency, Inference, Evaluation, Applications, and Challenges.

Have you ever wondered how ChatGPT, Gemini, or Claude can write an email, explain a complex topic, or even create a poem in seconds? The magic behind these tools is something called a **Large Language Model**, or **LLM**.

Think of an LLM as a brilliant student who has read almost the entire internet - every Wikipedia page, every book, every blog post. After reading all that, it hasn't just memorized facts; it has learned the patterns of language, logic, and reasoning.

This article, based on the highly detailed research paper "A Comprehensive Overview of Large Language Models," will break down how these models work in simple terms. No background needed! We'll walk through their journey from digital infants to the incredibly capable AI assistants we use today.

---

## The Building Blocks: What Are LLMs Made Of?

Before an LLM can learn, it needs a brain and a way to understand language. Here are the core components that make it all possible.

### 1. Tokenization: Breaking Language into Lego Bricks

Computers don't understand words and sentences the way we do. They need to break them down into numbers. The first step is **Tokenization**.

An LLM uses a "tokenizer" to split text into smaller pieces called tokens. A token can be a whole word, a part of a word, or even just a punctuation mark.

For example, the word "unforgettable" might be broken down into three tokens: `un`, `forget`, and `able`.

This is incredibly efficient. Instead of having to learn millions of individual words, the model can learn the meaning of these smaller building blocks and combine them in new ways.

### 2. Architecture: The Blueprint of an LLM's Brain

Most modern LLMs are built on an architecture called the **Transformer**, introduced in 2017. The Transformer was revolutionary because of a special mechanism called **attention**.

**Attention** is the LLM's ability to focus on what matters. When you read the sentence, "The dog chased the ball across the park," your brain automatically links "dog" with "chased" and "ball" with "chased." The attention mechanism allows an LLM to do the same, weighing the importance of different words when generating a response.

LLM architectures generally come in a few flavors:

- **Encoder-Decoder**: Think of this like a human translator. The encoder reads and understands the input text (e.g., an English sentence). The decoder then uses that understanding to generate the output (e.g., the same sentence in German).

- **Decoder-Only (or Causal Decoder)**: This is the most common architecture for chatbots like ChatGPT. It's a master of prediction. It reads text from left to right and its only job is to predict the very next word (or token). By doing this over and over, it can write entire essays.

- **Mixture-of-Experts (MoE)**: Instead of one giant, monolithic brain, an MoE model is like a team of specialists. When it receives a prompt, a "router" sends the question to the most relevant "experts." For a coding question, it might consult the programming experts; for a history question, the history experts. This makes the model faster and more efficient.

---

## The Life of an LLM: From Birth to Graduation
An LLM goes through several stages of training to become the powerful tool you interact with. We can think of it as a three-step educational journey.

*The training pipeline of an LLM, from pre-training to alignment. (A simplified version of Figure 6 from the paper)*

### Step 1: Pre-training (Reading the Entire Library)

This is the foundational stage where the LLM learns about the world. It's fed a massive amount of text data - terabytes of it from the internet.

During pre-training, its goal is simple: predict the next word. It looks at a sentence like, "The cat sat on the…" and tries to guess the next word. At first, its guesses are random. But every time it's wrong, it adjusts its internal connections. After doing this trillions of times, it develops an incredibly sophisticated understanding of grammar, facts, reasoning, and even style.

A pre-trained model is a powerful generalist but isn't very good at following instructions. It's like a brilliant student who knows everything but doesn't know how to answer specific questions on a test.

### Step 2: Fine-Tuning (Learning to Be a Helpful Assistant)

After pre-training, the model needs to learn how to be useful to humans. This is done through fine-tuning. There are two key types:

#### A. Instruction-Tuning:

Here, we teach the model to follow commands. We give it thousands of examples of instruction -> desired output.

**Example:**
- **Instruction**: "Summarize this article about photosynthesis."
- **Desired Output**: "Photosynthesis is the process…"

After seeing enough examples, the model learns the general pattern of how to respond to user requests, turning it from a simple text-predictor into a helpful assistant.

#### B. Alignment Tuning (Making it Safe and Ethical):

This is one of the most important steps. A pre-trained model might generate text that is unhelpful, biased, or harmful because it learned from the unfiltered internet. Alignment is the process of steering the model to be **Helpful, Honest, and Harmless (HHH)**.

The most common method for this is **Reinforcement Learning from Human Feedback (RLHF)**:

1. **Collect Data**: Developers prompt the model and get it to generate a few different answers.
2. **Rank Responses**: Human reviewers rank these answers from best to worst.
3. **Train a Reward Model**: This human ranking data is used to train another, smaller AI called a reward model. The reward model's only job is to look at a response and give it a score for how "good" it is based on human preferences.
4. **Fine-Tune the LLM**: The LLM is then fine-tuned further. It generates responses and gets "points" from the reward model. It learns to adjust its own behavior to maximize these points, effectively learning to behave in ways that humans prefer.

### Step 3: Inference (Putting the LLM to Work)

This is the final stage, where you, the user, interact with the fully trained LLM. When you type a question into ChatGPT, you are performing **inference**.

The text you provide is called a **prompt**. The art of writing good prompts is called **prompt engineering**. How you phrase your question can dramatically change the quality of the answer. Here are a few simple techniques:

- **Zero-Shot Prompting**: Asking a question directly, with no examples. (e.g., "What is the capital of Australia?")
- **Few-Shot Prompting**: Giving the model a few examples in your prompt to show it the format you want.
- **Chain-of-Thought (CoT) Prompting**: For complex problems, you can simply add "Think step-by-step" to your prompt. This encourages the model to break down its reasoning process, which often leads to a more accurate answer.

---

## Giving LLMs Superpowers: RAG and Tool Use

Even with all this training, LLMs have two major limitations: their knowledge is frozen in time, and they can't interact with the outside world. To solve this, developers have given them "superpowers."

### Retrieval-Augmented Generation (RAG): An Open-Book Exam

An LLM's knowledge is only as current as its training data. RAG solves this by connecting the LLM to a live, external knowledge source (like the internet or a company's internal documents).

Here's how it works:

1. You ask a question (e.g., "What were the key announcements from Apple's event last week?").
2. The RAG system first retrieves relevant, up-to-date information from its connected database.
3. It then feeds this information to the LLM along with your original prompt.
4. The LLM uses this new context to generate a factually accurate and current answer.

*RAG gives an LLM access to external knowledge, making its answers more accurate and up-to-date. (Figure 12 from the paper)*

### Tool Use: Giving LLMs Hands and Eyes

LLMs live inside a computer and can only manipulate text. They can't check the weather, book a flight, or perform a calculation. Tool use gives them the ability to call on other programs (or "tools") to perform actions.

When you ask a tool-enabled LLM, "What's the weather in London and can you add a reminder to my calendar to pack an umbrella?"

1. The LLM identifies it needs two tools: a weather API and a calendar API.
2. It calls the weather tool for "London."
3. It calls the calendar tool to create a reminder.
4. It then synthesizes the results into a single, helpful response for you.

---

## The Road Ahead: Challenges and the Future

Despite their incredible abilities, LLMs are not perfect. The research paper highlights several key challenges that developers are actively working to solve:

- **Hallucinations**: Sometimes, LLMs confidently make things up.
- **Bias**: They can reflect the societal biases present in their vast training data.
- **Cost**: Training and running these models requires enormous computational power and is very expensive.
- **Safety**: Ensuring models are not used for harmful purposes is a constant battle.

The field of LLMs is one of the fastest-moving areas in technology. Researchers are making breakthroughs every day to make these models smarter, safer, and more accessible. From a simple text-predictor to a world-knowledge reasoner, the journey of the LLM is a testament to the power of data and computation. And it's only just beginning.

---

*This article is a simplified, tutorial-style summary of the comprehensive research paper: Naveed, H., Khan, A. U., Qiu, S., et al. (2024). "A Comprehensive Overview of Large Language Models."*