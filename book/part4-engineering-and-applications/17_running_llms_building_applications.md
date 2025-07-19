---
title: "Running LLMs & Building Applications"
nav_order: 17
parent: "Part IV: Engineering & Applications"
grand_parent: "LLMs: From Foundation to Production"
description: "A practical guide to building applications on top of LLMs, covering API usage, prompt engineering, memory, and deploying models with frameworks like FastAPI and LangChain."
keywords: "LLM Applications, API, FastAPI, LangChain, Prompt Engineering, Chatbot Memory, Deployment, Docker"
---

# 17. Running LLMs & Building Applications
{: .no_toc }

**Difficulty:** Intermediate | **Prerequisites:** Web Development, APIs
{: .fs-6 .fw-300 }

This chapter bridges the gap between models and products. We'll cover the practical software engineering skills needed to build real-world applications powered by Large Language Models, whether you're using a commercial API or hosting an open-source model yourself.

---

## 📚 Core Concepts

<div class="concept-grid">
  <div class="concept-grid-item">
    <h4>Using LLM APIs</h4>
    <p>Interacting with commercial LLM providers like OpenAI and Anthropic, including managing API keys, handling rate limits, and monitoring costs.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Prompt Engineering</h4>
    <p>The art and science of designing effective prompts to elicit the desired behavior from a model.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Structured Outputs</h4>
    <p>Techniques like JSON mode and function calling that force a model to produce output in a specific, machine-readable format.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Chatbot Memory</h4>
    <p>Methods for managing conversation history to give chatbots memory, from simple buffers to more complex summarization techniques.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Application Frameworks</h4>
    <p>Using libraries like FastAPI to build robust API backends and frameworks like LangChain or LlamaIndex to orchestrate complex LLM workflows.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Containerization</h4>
    <p>Packaging an LLM application and its dependencies into a Docker container for portable and scalable deployment.</p>
  </div>
</div>

---

## 🛠️ Hands-On Labs

1.  **Build a FastAPI Backend**: Create a simple API endpoint that takes a prompt, sends it to an LLM, and streams back the response.
2.  **Chatbot with Memory**: Use LangChain to build a simple chatbot that remembers the last few turns of the conversation.
3.  **Structured Output with Function Calling**: Use an API that supports function calling (like OpenAI's) to build a tool that can answer questions by calling a simple calculator function.
4.  **Dockerize Your Application**: Write a Dockerfile for your FastAPI application and build a container image.

---

## 🧠 Further Reading

- **[FastAPI Documentation](https://fastapi.tiangolo.com/)**: The official documentation for the FastAPI framework.
- **[LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)**: The documentation for the LangChain library.
- **[OpenAI API Documentation](https://platform.openai.com/docs/overview)**: The API reference for OpenAI models, including details on function calling.
- **[Docker Documentation](https://docs.docker.com/get-started/)**: An introduction to Docker and containerization. 