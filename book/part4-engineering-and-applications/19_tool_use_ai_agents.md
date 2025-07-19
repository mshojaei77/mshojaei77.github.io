---
title: "Tool Use & AI Agents"
nav_order: 19
parent: "Part IV: Engineering & Applications"
grand_parent: "LLMs: From Foundation to Production"
description: "An introduction to AI Agents, which combine LLMs with tools like calculators, search engines, and APIs to solve complex, multi-step problems in the real world."
keywords: "AI Agents, Tool Use, Function Calling, ReAct, Planning, LangGraph, AutoGen, CrewAI"
---

# 19. Tool Use & AI Agents
{: .no_toc }

**Difficulty:** Advanced | **Prerequisites:** Function Calling, Planning
{: .fs-6 .fw-300 }

What happens when you give a language model a set of tools and a goal? You get an AI Agent. This chapter explores the exciting and rapidly-developing field of agentic AI, where LLMs are used as a reasoning engine to decide which tools to use in what order to accomplish complex tasks.

---

## 📚 Core Concepts

<div class="concept-grid">
  <div class="concept-grid-item">
    <h4>Tool Use / Function Calling</h4>
    <p>The ability of an LLM to call external functions or APIs, providing the model with access to real-time information and the ability to take actions.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Agents</h4>
    <p>Systems that use an LLM as a "brain" to reason about a goal, create a plan, and execute a sequence of actions (often involving tool use) to achieve that goal.</p>
  </div>
  <div class="concept-grid-item">
    <h4>ReAct Framework</h4>
    <p>A popular agentic framework where the model follows a "Reason, Act, Observe" loop, iteratively reasoning about what to do next, taking an action with a tool, and observing the result.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Planning & Task Decomposition</h4>
    <p>The ability of an agent to break down a complex, high-level goal into a series of smaller, manageable sub-tasks.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Multi-Agent Systems</h4>
    <p>Sophisticated systems where multiple specialized agents collaborate to solve a problem, each with its own role and set of tools.</p>
  </div>
  <div class="concept-grid-item">
    <h4>Agent Frameworks</h4>
    <p>Libraries like LangGraph, AutoGen, and CrewAI that provide abstractions and tools for building and orchestrating complex, multi-agent workflows.</p>
  </div>
</div>

---

## 🛠️ Hands-On Labs

1.  **Build a Tool-Using Agent**: Use an LLM with function-calling capabilities to build a simple agent that can use a web search API to answer questions about current events.
2.  **Implement a ReAct Agent**: Create a simple ReAct-style agent that can solve a multi-step problem by reasoning about its plan and using a calculator tool.
3.  **Create a Multi-Agent System**: Use a framework like CrewAI or AutoGen to create a two-agent system where one agent researches a topic and the other writes a summary.

---

## 🧠 Further Reading

- **[The AutoGen library](https://microsoft.github.io/autogen/)**: Documentation for Microsoft's framework for multi-agent systems.
- **[The CrewAI library](https://docs.crewai.com/)**: Documentation for a popular, role-based framework for orchestrating multi-agent systems.
- **[The LangGraph library](https://langchain-ai.github.io/langgraph/)**: A library from LangChain for building robust, stateful, multi-agent applications.
- **[Adeo Ressi: "An Introduction to AI Agents"](https://www.adeore.com/posts/an-introduction-to-ai-agents)**: A good conceptual overview of what AI agents are and how they work. 