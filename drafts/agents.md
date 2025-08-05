---
title: "AI Agents"
nav_order: 3
parent: drafts
layout: default
---


# AI Agents: From Foundation to Production

## Front Matter

### Preface – Why “Agentic” Approaches Matter Now

In the rapidly evolving world of Artificial Intelligence, AI agents are emerging as powerful tools that leverage large language models (LLMs) to reason, strategize, and execute complex tasks. While the concept of AI agents has been around for a long time in computer science, the recent popularity stems from autonomous systems based on large language models like GPT, Claude, and Llama. They are a fundamental shift in how we use large language models (LLMs) or multimodal vision language models (VLMs) and open up new possibilities for automation and human-AI interaction.

These sophisticated systems combine advanced reasoning capabilities with memory and the ability to interact with external resources, making them ideal for automating a wide range of processes. Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. This book shares what we’ve learned from working with our customers and building agents ourselves, and gives practical advice for developers on building effective agents.

Success in the LLM space isn't about building the most sophisticated system; it's about building the right system for your needs. We've consistently observed that the most successful implementations use simple, composable patterns rather than complex frameworks.

By the final page, you’ll have production-ready patterns for every item on the subject list: agentic designs, atomic tools, memory, RAG / Agentic RAG, MCP, parallel & multi-agent orchestration, API-call chaining, ReAct, autonomous loops, text-to-SQL, search tools, persistence & streaming, human-in-the-loop, reasoning & chaining, code-exec agents, browser automation, async apps, routing, prompt chaining, orchestrator-worker, and evaluator-optimizer—plus robust testing, deployment, and monitoring schemas.

### How to Use This Book – Suggested Learning Paths for Beginners ↔ Advanced

This book is designed to be a comprehensive guide for a wide range of readers, from product managers exploring use cases to developers building AI agent systems. Whether you have experience with AI and coding or not, this guide will provide both the conceptual background and technical insights you need.

Here is a suggested reading order:

1.  **Parts I–II for foundational mental models.** These sections cover the essential concepts that form the basis of all agentic systems.
2.  **Jump to any Part V chapter when you need a specialised agent.** These chapters are designed to be standalone guides for specific applications.
3.  **Parts VI–VII when you’re ready for production hardening.** These sections focus on the practical aspects of deploying and maintaining agents in real-world environments.
4.  **Keep Appendices handy as quick references during builds.** They contain valuable cheat sheets, templates, and glossaries.

Happy hacking—and may your agents never hallucinate! 🚀

### Prerequisites – Python 3.10+, Basic ML, REST APIs, and Terminal Skills

To get the most out of the hands-on labs and technical sections of this book, you should have the following:

*   **Python 3.10+:** A solid understanding of Python programming is essential.
*   **Basic ML:** Familiarity with fundamental machine learning concepts will be beneficial.
*   **REST APIs:** Experience with using and understanding REST APIs is required for tool integration.
*   **Terminal Skills:** Basic command-line interface (CLI) skills are necessary for running labs and deploying applications.

Don't worry if you don't have experience with AI or coding; the book will try to explain concepts in non-technical terms, and you can feel free to skip the more technical sections around implementation.

---

## Part I · Foundations of Agentic AI

### Chapter 1: What Is an AI Agent?

This chapter introduces the fundamental concepts of AI agents, distinguishing them from other AI systems and exploring the spectrum of their autonomy.

#### 1.1 The Definition of an AI Agent

An AI agent can be defined in several ways. Some define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, all these variations are categorized as agentic systems. An AI agent can be defined as a system that uses a large language model (LLM) to reason through problems, create actionable plans, and execute those plans using a set of tools. Think of an AI agent as a digital assistant capable of understanding complex requests, devising a step-by-step plan, and using various tools to achieve its goals. For example, an AI agent might not only answer questions but also book flights, manage calendars, or analyze data.

Although there are many definitions, you can still find some repeating patterns — given a goal, an AI agent can plan and try to achieve it by accessing external info or using tools autonomously. An oversimplification to help you understand AI agents might be that AI agents replace traditional programming control flows with an LLM that has access to info & tools.

##### Key Aspects of AI Agents:

*   **Agency or Autonomy:** As the name suggests, agents have agency, which means that they can act independently to achieve a goal. The agency can have many different levels, from simply routing control flow based on users' intents to function execution to multi-step flows and eventually to complex multi-agent cooperation. When we're designing AI agent products, we can define AI agents with varied agency levels to match the complexity of the product.
*   **Access to External Info/Tools:** Unlike isolated LLMs or VLMs, agents can interact with the external environment to access information or use tools to help them achieve their goals. This allows them to make informed decisions based on more updated relevant data that is beyond the data in their pre-training. They can use APIs or functions to access the internet, modify databases, or even do things in the physical world.
*   **Reasoning, Planning, and Execution:** Another core aspect is that when given a goal, AI agents can reason about which actions to take, plan a list of steps, and then execute those steps autonomously, even learning and adapting along the way. This makes AI agents different from previous predefined workflows to accommodate a wide range of cases, even the ones the agent product designers might not have thought about.

#### 1.2 Agent vs. LLM: The Autonomy Spectrum

It is important to draw an architectural distinction between workflows and agents:

*   **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
*   **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

AI agents work in a continuous reasoning-acting loop. At a high level, AI agents work just like what we do when given a task: we think, act, and learn from what happened—then we repeat until the job is done. The core of this process is a continuous ReAct (reasoning-acting) loop. The LLM reasons about the current situation based on context & message history, it makes a function call, the glue code parses the function call, executes the function, and puts the result into the context, and the LLM can then observe the result and reason again based on the new information to update the plan. This loop continues until the goal is achieved or a stopping condition is met.

#### 1.3 When and When Not to Use Agents

When building applications with LLMs, it is recommended to find the simplest solution possible and only increase complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

For simple, well-defined tasks and flows, traditional programmatic flows may work much more efficiently and reliably than AI agents. For tasks that require LLM capabilities but don't require complex reasoning, simple LLM-based workflows may be enough. It is when the task becomes complex, open-ended, and requires dynamic decision-making that AI agents become the better choice.

Here is a simple flow chart to help determine if you need to implement AI agents:

1.  Does the task require LLM capabilities (e.g., handling unstructured data)?
    *   No -> Use programmatic flows.
    *   Yes -> Proceed to 2.
2.  Is the task complex, open-ended, and requires dynamic decision-making?
    *   No -> Use simple LLM-based workflows (e.g., prompt chaining).
    *   Yes -> Use AI agents.

**Pros of using AI agents:**

*   **Handling complex tasks:** They can handle open-ended problems and make dynamic decisions.
*   **Flexibility:** They can adapt to cover edge cases and gather missing information on their own.
*   **Resource reduction:** They can automate tasks previously done by humans, reducing costs.
*   **Real-world interaction:** They can use tools to perform tasks beyond text generation.

**Potential risks:**

*   **Complexity:** They may add overhead like vector databases, async calls, and error handling.
*   **Latency:** The need for generated tokens for reasoning and iterative steps can cause delays.
*   **Cost:** Extra tokens for reasoning can quickly increase LLM API costs.
*   **Potential errors:** They may be prone to errors, which can compound in multi-step tasks.
*   **Lack of interpretability & hard to debug:** Their dynamic nature can make debugging difficult.

#### 1.4 Use-Cases for AI Agents

AI agents are being applied across various industries. Some promising applications include:

*   **Customer Support:** This combines chatbot interfaces with enhanced capabilities through tool integration. Support interactions naturally follow a conversational flow while requiring access to external information and actions like pulling customer data, issuing refunds, or updating tickets. Success can be clearly measured through user-defined resolutions.
*   **Coding Agents:** The software development space has shown remarkable potential, with capabilities evolving from code completion to autonomous problem-solving. Agents are effective because code solutions are verifiable through automated tests, they can iterate on solutions using test results as feedback, the problem space is well-defined, and output quality can be measured objectively. In some implementations, agents can solve real GitHub issues based on the pull request description alone.

#### 1.5 Hands-On Lab: Build a “Hello-Agent” CLI Loop

This lab will guide you through building a simple command-line interface (CLI) loop that serves as a basic "Hello, World!" for AI agents. You will implement a simple reasoning loop to demonstrate the core concept of an agent.

### Chapter 2: Agentic Design Patterns

This chapter explores common patterns for agentic systems, starting with the foundational building block—the augmented LLM—and progressively increasing complexity from simple compositional workflows to autonomous agents.

#### 2.1 The Augmented LLM as a Building Block

The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Current models can actively use these capabilities by generating their own search queries, selecting appropriate tools, and determining what information to retain. The implementation should focus on tailoring these capabilities to the specific use case and ensuring they provide an easy, well-documented interface for the LLM. One approach is through a Model Context Protocol, which allows developers to integrate with third-party tools with a simple client implementation.

You can think of an agent as a smart digital worker with a brain, tools, and memory. These components work together to enable the agent to plan, perform actions, adapt based on observations, and complete tasks.

*   **The Brain:** The brains of this wave of agents are LLMs or VLMs. These models can understand instructions, process information, draft plans, evaluate them, and generate responses or structured function calls to use a tool. It doesn't have to be a single LLM; different models can be used for different steps. For example, a capable model like GPT-4 for reasoning and smaller, specialized models for tasks like code generation.
*   **The Hands (Tools):** These are external functions provided to the agent. They can range from a simple calculator to complex APIs for accessing databases, sending emails, or even code sandboxes. Tool definitions are provided in the LLM prompt, and the LLM can choose which to use with the appropriate parameters.
*   **Memory:** Agents need memory to maintain context for learning and adapting. This allows them to recall past interactions, store information about goals and tool usage, and adapt future steps. Memories can be short-term (for the current session) or long-term (preserved across sessions, typically in a vector database).
*   **Glue Code:** This links the brain, tools, and memory together. It defines tools, parses LLM actions, and serves information into the prompts.

#### 2.2 ReAct (Reason+Act)

The ReAct pattern enhances the capabilities of agents by integrating the use of tools. It allows agents to observe the outcome of each step and make a decision on how to proceed, rather than blindly following a sequence. ReAct interleaves reasoning and action, with the agent analyzing observations at each step to determine the next course of action. This continuous loop of reasoning, acting, and observing enables the agent to improve its plan and approach iteratively.

#### 2.3 Plan-and-Execute

In this pattern, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results. This is well-suited for complex tasks where you can’t predict the subtasks needed (e.g., coding). The key difference from parallelization is its flexibility—subtasks aren't pre-defined but are determined by the orchestrator based on the specific input. This is also known as the Orchestrator-Workers workflow.

#### 2.4 Evaluator-Optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop. This is effective when there are clear evaluation criteria and iterative refinement provides measurable value. This is analogous to the iterative writing process a human writer might go through. Other names for this pattern include Self-Refine and CRITIC.

*   **Self-Refine:** This approach allows agents to improve their outputs by reflecting on their prior attempts, correcting errors, and improving future outputs.
*   **CRITIC:** This method uses an evaluator to score the outcome of a previous action, and then a reflection model to analyze what went wrong. This approach is more robust and the output is more refined.
*   **Reflexion:** This builds on basic reflection, incorporating reinforcement learning, and includes an evaluator to check outputs against external data.

#### 2.5 Orchestrator-Worker

As mentioned in the Plan-and-Execute section, this workflow involves a central LLM (the orchestrator) that dynamically breaks down tasks and delegates them to worker LLMs. The orchestrator then synthesizes their results. This pattern is well-suited for complex tasks where subtasks cannot be predicted in advance. Examples include coding products that make complex changes to multiple files or search tasks that involve gathering and analyzing information from multiple sources.

#### 2.6 Hands-On Lab: Compare Patterns on a Trivia Task

This lab will involve implementing and comparing the ReAct, Plan-and-Execute, and Evaluator-Optimizer patterns on a trivia-based task to understand their respective strengths and weaknesses.

### Chapter 3: Reasoning, Chaining & Dynamic Composition

This chapter delves into the techniques agents use for complex reasoning, sequencing actions, and dynamically choosing tools.

#### 3.1 Chain-of-Thought (CoT)

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. Programmatic checks ("gates") can be added on any intermediate steps to ensure the process is on track. This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy by making each LLM call an easier task. When you ask an LLM to “think step by step,” you are asking it to decompose a task.

Examples:
*   Generating marketing copy, then translating it into a different language.
*   Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

#### 3.2 Tree-of-Thought (ToT)

Tree-of-Thought (ToT) enables agents to explore multiple reasoning paths in parallel, allowing for a more thorough exploration of potential solutions. This is a form of multi-plan selection where several possible plans are generated and the most promising one is selected. This strategy is particularly useful when the optimal path is not immediately clear.

#### 3.3 Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for the separation of concerns and the building of more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs. This works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately.

Examples:
*   Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes.
*   Routing easy questions to smaller models like Claude 3.5 Haiku and hard questions to more capable models like Claude 3.5 Sonnet to optimize cost and speed.

#### 3.4 Tool Selection

Agents use tools to interact with the world. These tools can be read-only (e.g., retrieving information) or write (e.g., changing data or initiating actions). Write actions can be used to automate workflows but require careful security consideration. The mechanism for using tools is often function calling.

**Function Calling:** This is how an LLM uses tools. Each tool has an execution entry point, parameters, and documentation. It is important to specify what tools the agent can use for a given query. The LLM generates a structured output (typically JSON) with the function name and parameters, which is then parsed and executed by the application code.

**Prompt Engineering Your Tools:** Tool definitions and specifications should be given as much prompt engineering attention as the overall prompts.
*   Give the model enough tokens to "think" before it writes itself into a corner.
*   Keep the format close to what the model has seen naturally occurring in text on the internet.
*   Make sure there's no formatting "overhead."
*   Put yourself in the model's shoes: Is it obvious how to use this tool?
*   Change parameter names or descriptions to make things more obvious.
*   Test how the model uses your tools and iterate.
*   Poka-yoke your tools: Change the arguments so that it is harder to make mistakes. For example, requiring absolute filepaths instead of relative ones.

#### 3.5 Hands-On Lab: Visualise Reasoning Traces with LangGraph

In this lab, you will use LangGraph to build an agent that uses Chain-of-Thought and tool selection, and then visualize the reasoning traces to understand its decision-making process.

---

## Part II · Tools, Memory & Persistence

### Chapter 4: Atomic Tools & API Call Chaining

This chapter covers the fundamentals of equipping agents with tools, from single function calls to sequencing multiple API interactions.

#### 4.1 Function Calling and JSON Schemas

Function calling is the primary mechanism for an agent to use a tool. Modern LLMs can be instructed to produce a JSON object containing the name of a function to call and the arguments to provide. The application code then parses this JSON, executes the corresponding function, and feeds the result back to the agent as an observation. JSON schemas are used to define the structure of the expected output, ensuring reliability.

#### 4.2 REST/GraphQL APIs

Agents can be given tools that interact with external services via REST or GraphQL APIs. This allows them to fetch real-time data (e.g., weather, stock prices) or perform actions (e.g., sending an email, creating a calendar event).

#### 4.3 Error Handling

Robust error handling is crucial when dealing with external tools. The agent's workflow must be able to gracefully handle API failures, invalid responses, or timeouts. The error information should be passed back to the agent as an observation, allowing it to reflect on the failure and potentially retry the action or try a different approach.

#### 4.4 API Call Chaining

Many tasks require a sequence of tool calls, where the output of one call is the input for the next. This is known as API call chaining. For example, to find the weather in the capital of a country, an agent would first need a tool to find the capital, and then use that result with a weather API tool.

#### 4.5 Hands-On Lab: Weather-API & PDF-Summariser Agent

This lab will guide you through building an agent that uses two distinct tools: one that calls a weather API and another that summarizes the content of a PDF file. You will implement logic for the agent to choose the correct tool based on the user's query.

### Chapter 5: ReAct-Style Tool Use

This chapter builds upon the ReAct pattern, focusing on the practical implementation of the action-observation loop for tool use.

#### 5.1 The Action-Observation Loop

The core of ReAct-style tool use is the iterative loop:
1.  **Thought:** The agent reasons about the task and decides on an action.
2.  **Action:** The agent invokes a tool.
3.  **Observation:** The result from the tool is observed.
This loop repeats, allowing the agent to dynamically build a plan based on real-time feedback from its environment.

#### 5.2 Self-Reflection in the Loop

Within the loop, the agent can engage in self-reflection. After an observation, it can assess whether the action was successful, whether the result is what it expected, and how this new information affects its overall plan. This reflective step is key to handling errors and adapting to unexpected outcomes.

#### 5.3 Retries and Error Correction

When a tool call fails or produces an unexpected result, a ReAct agent can use its reasoning ability to attempt a fix. It might retry the call, perhaps with different parameters, or it might decide that the chosen tool is not appropriate and select a different one. This iterative error correction makes the agent more robust.

#### 5.4 Hands-On Lab: Build a Math + Web-Search ReAct Agent

In this lab, you will build an agent that can solve mathematical problems and answer general knowledge questions. It will be equipped with two tools: a calculator and a web search API. You will implement the ReAct loop to allow the agent to decide which tool to use and to combine their outputs to answer complex queries.

### Chapter 6: Agent Memory & Persistence

This chapter explores how to give agents memory, overcoming the stateless nature of LLMs to enable continuity and learning.

#### 6.1 The Challenge of Statelessness

LLMs are inherently stateless; they have no memory of past interactions beyond the current context window. Agent memory architecture is the critical component designed to overcome this challenge, allowing agents to maintain coherent conversations and learn from experience.

#### 6.2 Short-Term vs. Long-Term Memory

*   **Short-Term Memory:** This is the agent's working memory, holding the context for the current task. It includes recent dialogue history and intermediate thoughts. This is often managed within the LLM's context window.
*   **Long-Term Memory:** This is a persistent knowledge store for information that needs to be recalled across sessions. It's crucial for personalization and historical knowledge.

#### 6.3 Episodic vs. Semantic Memory

Long-term memory can be further categorized:
*   **Episodic Memory:** Memory of specific events or past interactions, like a diary. For example, "the user previously asked about flights to Paris."
*   **Semantic Memory:** Memory of general facts, concepts, and knowledge. For instance, a company's product specifications.
*   **RAG-based Memory:** This uses retrieval-augmented generation to incorporate past experiences, retrieving relevant information from past interactions to inform current planning.
*   **Embodied Memory:** This involves fine-tuning LLMs with experiential samples to improve planning abilities.
*   **LLM Cache:** Caching previously generated responses can significantly reduce response times and costs. This can be implemented using keyword or semantic caching.

#### 6.4 Vector Stores and Persistence Layers

Long-term memory is typically implemented using external databases.
*   **Vector Stores (e.g., Chroma, Pinecone):** These are specialized databases for storing and querying high-dimensional vector embeddings, which represent the semantic meaning of text. They are ideal for semantic and episodic memory retrieval.
*   **Persistence Layers:** For other types of state, traditional databases are used. In-memory databases like Redis are good for fast-access short-term memory, while SQL or NoSQL databases can provide durable long-term storage for user profiles or conversation logs.

#### 6.5 MemGPT

MemGPT is a system that intelligently manages different memory tiers, deciding when to move information from the limited context window to external storage, inspired by virtual memory in traditional operating systems.

#### 6.6 Hands-On Lab: Implement a Redis + Chroma Memory Backend

This lab will involve implementing a memory system for an agent using Redis for short-term memory and ChromaDB for long-term semantic memory, allowing the agent to recall information from previous conversations.

### Chapter 7: Streaming & Async Applications

This chapter focuses on building responsive and scalable agent applications that can handle real-time data and long-running tasks.

#### 7.1 Tokens vs. Chunks

Understanding the different units of data in streaming is important. **Tokens** are the smallest units of text generated by an LLM. **Chunks** are larger blocks of data, which might contain multiple tokens or structured information about the agent's state.

#### 7.2 Back-Pressure

Back-pressure is a mechanism to handle situations where a data stream is producing information faster than the consumer can process it, preventing system overload.

#### 7.3 Server-Sent Events (SSE) and WebSockets

These are two common technologies for implementing real-time communication between a server (the agent backend) and a client (the user interface).
*   **Server-Sent Events (SSE):** A simpler, one-way communication protocol where the server can push data to the client.
*   **WebSockets:** A more complex, two-way communication protocol that allows for full-duplex interaction.

#### 7.4 `asyncio` for Asynchronous Applications

For applications that need to handle multiple tasks concurrently (like managing several user conversations or performing multiple tool calls in parallel), Python's `asyncio` library is essential. It allows for non-blocking I/O operations, making the application more efficient and responsive.

#### 7.5 Hands-On Lab: Live-Coding an Async Chat & Streaming Agent

In this lab, you will build a chat application from scratch. The backend will use `asyncio` to handle multiple connections, and the agent's responses, including its intermediate thoughts, will be streamed to the user's browser in real-time using SSE or WebSockets.

---

## Part III · Retrieval & Planning

### Chapter 8: Retrieval-Augmented Generation (RAG)

This chapter provides a detailed look at RAG, a foundational technique for connecting LLMs to external knowledge.

#### 8.1 The RAG Pipeline: Indexing, Chunking, Embeddings

The standard RAG pipeline involves several key steps:
*   **Indexing/Ingestion:** An offline process where documents are prepared.
    *   **Loading:** Documents are loaded from sources (PDFs, websites, etc.).
    *   **Chunking:** Documents are split into smaller, manageable chunks. The way documents are broken down affects the relevance of results. Techniques include splitting by token count, semantic meaning, or paragraphs.
    *   **Embeddings:** Each chunk is converted into a numerical vector (embedding) that captures its semantic meaning. These are stored in a vector database.
*   **Retrieval:** When a user asks a query, the query is embedded, and the vector database is searched for the most similar document chunks.
*   **Generation:** The retrieved chunks are added to the prompt along with the original query, and the LLM generates an answer based on this augmented context.

#### 8.2 Improving RAG: Rerankers and Advanced Techniques

To improve the quality of retrieved information, several advanced techniques can be used:
*   **Rerankers:** After an initial retrieval, a reranker model can be used to re-order the results, prioritizing the most relevant chunks.
*   **Query Expansion/Transformation:** The original query can be expanded with synonyms or related terms, or transformed (e.g., rephrased) to better align with the documents. HyDE (hypothetical document embeddings) generates a hypothetical answer which is then used for the embedding search.
*   **Hybrid Retrieval:** Combining different methods like keyword search, graph search, and vector search can improve results.

#### 8.3 Hands-On Lab: Build a Basic Pinecone RAG Pipeline

This lab will guide you through the process of building a complete RAG pipeline. You will ingest a set of documents, create embeddings, store them in Pinecone, and build a query engine to answer questions based on the documents.

### Chapter 9: Agentic RAG

This chapter explores how to infuse the RAG process with agentic intelligence, moving from a static pipeline to a dynamic, reasoning-driven workflow.

#### 9.1 The Need for Agentic RAG

While powerful, the standard RAG pipeline is often too simplistic for complex queries that require synthesizing information from multiple sources or steps. Agentic RAG uses an agent to orchestrate a more dynamic retrieval process.

#### 9.2 Multi-Step Retrieval and Query Planning

An agent can perform multi-step retrieval. It can break a complex query into a series of sub-queries, execute a retrieval for each one, and then synthesize the results. This is also known as query planning.

#### 9.3 Self-Instruction and Reflection

An agentic RAG system can reflect on the quality of its retrieved information. If the initial retrieval is poor, the agent can decide to reformulate the query and try again, a process sometimes referred to as self-instruction.

#### 9.4 Hands-On Lab: Agentic RAG for Software Docs

In this lab, you will build an agent that can answer complex questions about a software library by navigating its documentation. The agent will use query planning and multi-step retrieval to find information spread across multiple pages.

### Chapter 10: Multi-Context Planning (MCP) and Advanced Planning

This chapter covers advanced planning techniques that enable agents to tackle large, complex tasks.

#### 10.1 Task Decomposition

Task decomposition is the process of breaking down a complex task into smaller, more manageable sub-tasks. This is crucial for handling intricate problems efficiently.
*   **Decomposition-First Methods:** The task is fully decomposed into sub-goals before any execution begins.
*   **Interleaved Decomposition:** This method dynamically combines decomposition and planning, often based on feedback from executed steps.

#### 10.2 Multi-Plan Selection

This strategy involves generating several possible plans and then selecting the most promising one. This is useful when the optimal path is not clear.
*   **Self-consistency:** Generating multiple outputs and selecting the most frequent or consistent one.
*   **Tree-of-Thought (ToT):** Exploring multiple reasoning paths in parallel.

#### 10.3 External Planner-Aided Planning

This involves leveraging external tools to enhance planning.
*   **Symbolic Planners:** Using established models like PDDL to generate formal plans.
*   **Neural Planners:** Using deep learning techniques to improve planning efficiency.
*   **Search Tools:** Augmenting LLMs with search tools to access public information.
*   **State Tracking Systems:** Helping the agent keep track of the environment's state.

#### 10.4 Memory-Augmented Planning

Incorporating memory mechanisms enhances planning by allowing agents to retain and utilize past experiences. This can involve RAG-based memory to retrieve relevant steps from similar past tasks.

#### 10.5 Context Windows and Hierarchical Planning

For very large tasks, like summarizing a 100-page document, the entire context won't fit into the LLM's context window. Hierarchical planning involves creating a high-level plan first, then creating more detailed sub-plans for each step. This allows for a more structured and manageable approach.

#### 10.6 Plan-Representation-Execution Cycles

Multi-Context Planning (MCP) often involves cycles where the agent:
1.  **Plans:** Decomposes the task.
2.  **Represents:** Creates a representation of the sub-task to be worked on.
3.  **Executes:** Acts on that sub-task.
This cycle repeats, allowing the agent to systematically work through a large problem.

#### 10.7 Hands-On Lab: Q&A over 100-page PDFs with MCP Planner

This lab will task you with building an agent that can answer detailed questions about a very long PDF document. You will implement a Multi-Context Planning agent that uses hierarchical planning and iterative execution to navigate the document.

---
## Part IV · Execution & Orchestration

### Chapter 11: Code Execution Agents

This chapter covers agents that can write and execute their own code, a powerful but potentially risky capability.

#### 11.1 The Power of Code Execution

Giving an agent the ability to execute code transforms it into a powerful computational tool. It can perform data analysis, create visualizations, automate software tasks, and much more.

#### 11.2 Sandboxing and Security

The ability to execute code comes with significant security risks. It is absolutely essential to run the code in a secure, sandboxed environment (e.g., a Docker container or a dedicated service) that is isolated from the host system and has restricted permissions.

#### 11.3 The Python REPL as a Tool

A common way to enable code execution is to give the agent a tool that provides access to a Python Read-Eval-Print Loop (REPL). The agent can then write and execute Python code snippets to perform calculations or manipulate data.

#### 11.4 Evaluation and Self-Correction

A powerful application of code execution is self-correction. An agent can write code to solve a problem, run it against a set of tests, and if the tests fail, it can analyze the error message and attempt to debug and fix its own code.

#### 11.5 Hands-On Lab: Auto-Grader That Fixes Its Own Code

In this lab, you will build an agent that is given a programming problem and a set of unit tests. The agent's task is to write Python code to solve the problem. It will then run its code against the tests and, if they fail, enter a loop to debug and correct its code until all tests pass.

### Chapter 12: Prompt Chaining & Routing

This chapter delves deeper into the workflow patterns of prompt chaining and routing, which are essential for building complex, multi-step applications.

#### 12.1 Conditional Prompts and Graph Flows

Prompt chaining involves sequencing LLM calls, but this can be enhanced with conditional logic. Instead of a linear chain, the workflow can be represented as a graph where the path taken depends on the output of previous steps. This allows for more dynamic and flexible applications.

#### 12.2 Guardrails

Guardrails are checks and balances built into the workflow to ensure the agent's behavior stays within desired bounds. This can involve screening for inappropriate content, validating tool outputs, or ensuring compliance with regulations.

#### 12.3 Routing Between Models

The routing pattern can be used to direct a user's query to the most appropriate model for the job. For example, a router might identify that a query is asking for an image and route it to an image generation model like DALL-E, while routing text-based queries to a model like GPT-4.

#### 12.4 Hands-On Lab: Route Queries Between Image & Text Models

This lab will involve building a router agent. The agent will analyze incoming user queries and decide whether to send them to a large language model for a text-based answer or to an image generation model to create a picture.

### Chapter 13: Parallel Agents & Async Orchestration

This chapter explores how to improve the efficiency and speed of agentic systems by running tasks and agents in parallel.

#### 13.1 Fan-Out/Fan-In Pattern

This is a common parallelization pattern where a task is fanned out to multiple workers that run concurrently, and their results are then fanned back in and aggregated or synthesized.

#### 13.2 Concurrency and Asynchronous Processing

Leveraging concurrency is key to building high-performance agents. This involves using asynchronous programming techniques to run multiple operations (like API calls or model inferences) at the same time without blocking the main execution thread.

#### 13.3 Frameworks for Orchestration: LangGraph and crewAI

*   **LangGraph:** A library from LangChain for building stateful, multi-agent applications. Its graph-based structure is well-suited for defining complex workflows with cycles and parallel execution.
*   **crewAI:** A framework for orchestrating role-playing, autonomous AI agents. It simplifies the process of creating "crews" of agents that collaborate on tasks.

#### 13.4 Examples of Parallelization

*   **Sectioning:** Breaking a task into independent subtasks that run in parallel. For example, one model instance could process a user query while another screens it for safety.
*   **Voting:** Running the same task multiple times with different prompts or models to get diverse outputs, which can then be compared or voted on to increase confidence.

#### 13.5 Hands-On Lab: Parallel Research Agent with Summariser

In this lab, you will build a research agent that takes a topic and spawns multiple worker agents. Each worker agent will research a different aspect of the topic in parallel. A final summariser agent will then aggregate the findings from all workers into a single, coherent report.

### Chapter 14: Multi-Agent Systems

This chapter focuses on the design and implementation of systems where multiple agents interact to solve a problem.

#### 14.1 Roles, Coordination, and Communication

In a multi-agent system, it's crucial to define:
*   **Roles:** Each agent should have a specialized role or expertise (e.g., Planner, Critic, Executor).
*   **Coordination:** A mechanism is needed to coordinate the actions of the agents. This is often done by a "supervisor" or "orchestrator" agent.
*   **Communication Channels:** Agents need a way to share information and results with each other.

#### 14.2 The "Crew" Concept

The idea of a "crew" of agents, popularized by frameworks like crewAI, involves assembling a team of agents with complementary skills to work together on a task, much like a human team.

#### 14.3 Common Architectures

*   **Orchestrator-Workers:** A central orchestrator breaks down tasks and delegates them to specialized workers. This is a flexible, hierarchical approach.
*   **Parallelization:**
    *   **Sectioning:** A task is divided into independent subtasks, which are run in parallel by different agents.
    *   **Voting:** The same task is given to multiple agents, and their diverse outputs are used to generate a more robust result.

#### 14.4 Hands-On Lab: Build a "Crew" with Planner & Worker Agents

This lab will use a framework like crewAI or LangGraph to build a multi-agent system. You will create a Planner agent that decomposes a task and a set of Worker agents that execute the sub-tasks, all coordinated to achieve a common goal.

### Chapter 15: Orchestrator-Worker & Evaluator-Optimizer Loops

This chapter provides a deeper look into two of the most powerful and sophisticated multi-agent design patterns.

#### 15.1 Supervisor Patterns

The Orchestrator-Worker (or Supervisor-Worker) pattern is a flexible architecture where a central agent manages a team of specialized agents. This is particularly useful for complex tasks where the exact sequence of steps is not known in advance. The orchestrator can dynamically decide which worker to call next based on the results of the previous step.

#### 15.2 Self-Critique and Iterative Refinement

The Evaluator-Optimizer loop is a pattern for iterative refinement. One agent (the Optimizer or Worker) produces an output, and another agent (the Evaluator or Critic) provides feedback. This is highly effective when clear evaluation criteria can be defined.

#### 15.3 The Role of Feedback (RLHF)

The feedback process in these loops is analogous to Reinforcement Learning from Human Feedback (RLHF), a technique used to align large language models. The same principle can be applied at the agent level, using feedback from either a human or another AI to guide the agent's learning and improve its performance.

#### 15.4 Hands-On Lab: AutoGen Reflection Loop on Coding Tasks

This lab will use the AutoGen framework to implement a reflection loop for a coding task. You will set up a Worker agent that writes code and a Critic agent that reviews the code and provides feedback, allowing the Worker to iteratively improve its solution.

---

## Part V · Specialised Agents

### Chapter 16: Text-to-SQL & Database Agents

This chapter focuses on agents designed to interact with structured data in databases.

#### 16.1 The Challenge of Schema Linking

For an agent to generate correct SQL, it must understand the database schema—the tables, columns, and relationships. Schema linking is the process of correctly mapping the entities mentioned in a natural language query to the corresponding parts of the schema.

#### 16.2 SQL Generation

The core task of a Text-to-SQL agent is to take a natural language question and generate a syntactically correct and semantically equivalent SQL query.

#### 16.3 The Execute-Validate-Fix Cycle

A robust Text-to-SQL agent often uses a correction loop:
1.  **Generate:** The agent generates a SQL query.
2.  **Execute & Validate:** The system attempts to execute the query.
3.  **Fix:** If the execution fails, the error message is fed back to the agent, which then tries to debug and fix its own query.

#### 16.4 Hands-On Lab: Natural-Language Dashboard Generator

In this lab, you will build an agent that can connect to a SQL database. Users will be able to ask questions in natural language (e.g., "Show me the total sales by region for the last quarter"), and the agent will generate and execute the SQL to produce the answer, effectively creating a natural-language-powered dashboard.

### Chapter 17: Web Browser Automation

This chapter explores how to build agents that can interact with and extract information from websites.

#### 17.1 DOM Parsing and Interaction

To automate a web browser, an agent needs tools that can parse the Document Object Model (DOM) of a webpage. This allows it to identify and interact with elements like buttons, links, and forms.

#### 17.2 Screenshot Feedback

A powerful technique for web automation agents is to provide them with screenshot feedback. After an action (like a click), the agent can be shown a screenshot of the resulting page, allowing it to "see" the outcome of its action and plan its next move.

#### 17.3 Tools like AgentGPT and AgentQ

This section will discuss existing tools and frameworks specifically designed for building web automation agents, analyzing their approaches and capabilities.

#### 17.4 Hands-On Lab: Book-Price Comparer That Clicks & Scrapes

In this lab, you will build an agent that is given the title of a book. The agent's task is to navigate to several online bookstores, search for the book, extract the price from each site, and then present a comparison to the user. This will involve using tools for navigation, clicking, typing, and scraping data from webpages.

### Chapter 18: Agentic Search Tools

This chapter covers the design of sophisticated search agents that go beyond simple keyword lookups.

#### 18.1 Limitations of Traditional Search

Traditional search engines are powerful, but an agentic approach can provide more synthesized and context-aware answers by performing multi-step research.

#### 18.2 Using SERP APIs

Instead of scraping search engine results pages (SERPs), agents can use dedicated SERP APIs that provide structured data from search results, which is easier and more reliable to work with.

#### 18.3 MCTS for Search Strategy

Monte Carlo Tree Search (MCTS) is an advanced algorithm that can be used by a search agent to explore the "tree" of possible search queries and follow-on links, helping it to build a more effective and comprehensive search strategy.

#### 18.4 Citations and Reliability

A key feature of a good search agent is providing citations for its claims, linking back to the sources of its information. This increases the reliability and trustworthiness of the agent's output.

#### 18.5 Hands-On Lab: Build an Academic-Paper Search Assistant

This lab will involve building an agent that helps users find and understand academic papers. The agent will use a search API (like Google Scholar or Semantic Scholar) to find relevant papers, extract key information like abstracts and authors, and potentially find full-text PDFs.

---
## Part VI · Human Interaction & Governance

### Chapter 19: Human-in-the-Loop (HITL)

This chapter addresses the critical need to incorporate human oversight and intervention into agentic systems.

#### 19.1 The Importance of Human Oversight

For high-stakes or sensitive tasks, it is often necessary to have a human in the loop to approve actions, provide guidance, or handle exceptions. This combines the speed of automation with the judgment of a human expert.

#### 19.2 Approval Gates and Uncertainty Triggers

HITL can be implemented through explicit approval gates, where the agent must pause and wait for human confirmation before proceeding with a critical action (e.g., sending an email or making a purchase). Intervention can also be triggered when the agent's own confidence in its decision falls below a certain threshold.

#### 19.3 UI Patterns for HITL

This section will explore effective user interface patterns for presenting information to a human reviewer and capturing their input in a clear and efficient way.

#### 19.4 Hands-On Lab: HITL Content-Moderation Pipeline

In this lab, you will build a content moderation system. An agent will first classify a piece of content. If the agent is highly confident, it will take action automatically. If its confidence is low, it will flag the content for review and route it to a simple web interface for a human moderator to make the final decision.

### Chapter 20: Evaluation, Guardrails & Optimisation

This chapter covers the essential practices for testing, ensuring the safety of, and improving agent performance.

#### 20.1 Unit Tests and Trace-Based Evals

Just like traditional software, agentic systems need to be rigorously tested. This includes:
*   **Unit Tests:** Testing individual tools and components.
*   **Trace-Based Evals:** Evaluating the entire reasoning process of an agent by analyzing its execution trace (the sequence of thoughts and actions).

#### 20.2 LLM-as-Judge

A powerful evaluation technique is to use another, often more powerful, LLM to act as a "judge." The judge LLM is given the agent's output and a set of criteria and is asked to score the agent's performance.

#### 20.3 Cost Dashboards and Performance Monitoring

In production, it's crucial to monitor the performance and cost of agents. This involves setting up dashboards to track key metrics like task success rate, latency, and token consumption.

#### 20.4 Hands-On Lab: Add Automatic Evals & Slack Alerts

In this lab, you will take a previously built agent and add an automated evaluation pipeline using the LLM-as-Judge pattern. You will also integrate it with a monitoring system to send an alert to a Slack channel if the agent's performance drops or its costs exceed a certain budget.

---

## Part VII · Deployment & Scaling

### Chapter 21: Persistence & Streaming Architectures

This chapter revisits persistence and streaming from an architectural perspective, focusing on building scalable and resilient production systems.

#### 21.1 Durable Queues and Event Sourcing

For asynchronous, long-running tasks, a robust architecture often involves:
*   **Durable Queues:** Using message queues (like RabbitMQ or SQS) to manage tasks, ensuring that they are not lost even if a component fails.
*   **Event Sourcing:** A pattern where all changes to the application state are stored as a sequence of events. This provides a complete audit trail and allows the state to be rebuilt at any time.

#### 21.2 State Snapshots

For very long-running agents, it can be inefficient to replay the entire event history to restore state. Taking periodic snapshots of the agent's state provides a more efficient recovery mechanism.

#### 21.3 Hands-On Lab: Kubernetes-Ready Streaming Agent Service

This lab will focus on packaging an agent application into a containerized service that is ready for deployment on Kubernetes. You will implement a durable state management system using a database and message queue.

### Chapter 22: Deployment Patterns

This chapter covers the common patterns and technologies for deploying agentic applications in the cloud.

#### 22.1 Serverless (e.g., AWS Lambda)

For simple, event-driven agents, a serverless architecture can be a cost-effective and scalable deployment option.

#### 22.2 GPU Inference Servers (e.g., RunPod)

When using open-source models, you need to host them on servers with GPUs. This section will discuss platforms and services for GPU inference.

#### 22.3 Optimizing Inference with vLLM and Ray Serve

*   **vLLM:** A library for fast LLM inference and serving.
*   **Ray Serve:** A scalable model serving library for building online inference APIs.

#### 22.4 Hands-On Lab: Deploy on RunPod & AWS Lambda

This lab will provide a practical comparison of two deployment models. You will deploy a simple agent using a serverless approach on AWS Lambda and a more complex agent with a self-hosted model on a GPU server using a service like RunPod.

### Chapter 23: Monitoring & Observability

This chapter covers the tools and techniques for monitoring the health and performance of agents in production.

#### 23.1 Tracing with OpenTelemetry

OpenTelemetry is an open-source observability framework that can be used to capture detailed traces of an agent's execution, showing the flow of a request through different components and services.

#### 23.2 Metrics, Logging, and Alerting

*   **Metrics:** Collecting key performance indicators (KPIs) like latency, error rates, and token usage.
*   **Logging:** Recording detailed information about the agent's operations for debugging.
*   **Alerting:** Setting up automated alerts to notify developers of problems.

#### 23.3 Hands-On Lab: Grafana Dashboard for Agent Metrics

In this lab, you will instrument an agent application to export metrics and traces using OpenTelemetry. You will then set up a Grafana dashboard to visualize these metrics in real-time, providing a comprehensive view of the agent's health and performance.

---
## Part VIII · Case Studies & Blueprints

### Chapter 24: Autonomous Research Assistant

*   **Use-Case:** An agent designed to conduct comprehensive research on a given topic.
*   **Highlights:** This system will be built using a trio of agents: a Planner that breaks down the research topic, multiple Researchers that gather information in parallel using Agentic RAG and web search tools, and a Critic that synthesizes and refines the final report.

### Chapter 25: Customer-Support RAG Bot

*   **Use-Case:** An advanced chatbot for customer support.
*   **Highlights:** This bot will use tool-calling to access customer data and knowledge bases. It will feature parallel FAQ retrieval to quickly find answers to common questions and a Human-in-the-Loop (HITL) escalation path to seamlessly hand over complex issues to a human agent.

### Chapter 26: Enterprise Workflow Orchestration

*   **Use-Case:** An agent system for automating complex business processes within a large enterprise.
*   **Highlights:** This system will be based on an Orchestrator-Worker architecture, with an Evaluator-Optimizer loop for quality control. It will also include robust compliance logging to create a complete audit trail of all actions taken by the agents.

---
## Appendices

### A. Framework Cheat-Sheets
This appendix will provide quick-reference cheat sheets for the key APIs and concepts in popular agentic frameworks, including:
*   LangChain
*   LangGraph
*   AutoGen
*   crewAI
*   Semantic-Kernel

### B. Configuration Templates
This appendix will contain ready-to-use configuration templates for common tools and deployment environments:
*   `.env` files for environment variables
*   `YAML` configuration files
*   `Docker Compose` files for local development
*   `Helm charts` for Kubernetes deployment

### C. Glossary of Agentic Terms
A comprehensive glossary defining the key terms and concepts used throughout the book, from "Agentic RAG" to "Vector Store."

### D. Further Reading & Community Resources
A curated list of resources for continued learning:
*   **Key Papers:** Links and DOIs for influential research papers in the field.
*   **Blogs:** Recommended blogs from industry leaders and researchers.
*   **Discords & Newsletters:** Pointers to active online communities and newsletters to stay up-to-date with the latest developments.

---

### Pedagogical Features

Throughout the book, you will find several features designed to enhance your learning experience:

*   **Concept Deep-Dives:** Short essays that provide a deeper theoretical understanding of key concepts (e.g., comparing Chain-of-Thought vs. Tree-of-Thought).
*   **Code Breakdowns:** Side-by-side annotated code listings that explain the implementation details of key patterns and techniques.
*   **Pro Tips:** Practical advice and tricks for improving performance, managing costs, and enhancing the security of your agents.
*   **Exercises:** "Fix-this-agent" and design challenges to test your understanding and problem-solving skills.
*   **Review Checklists:** A checklist at the end of each major part to ensure you have mastered the key concepts before moving on.

### Frameworks: A Note on Simplicity and Abstraction

There are many frameworks that make agentic systems easier to implement, including LangGraph from LangChain, Amazon Bedrock's AI Agent framework, Rivet, and Vellum. These frameworks are useful for getting started, as they simplify tasks like calling LLMs, defining tools, and chaining calls together.

However, they often create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice. It is suggested that developers start by using LLM APIs directly, as many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of error. Some recommend building your own AI agent to avoid the overhead of frameworks. After all, an AI agent is a prompt with a goal, tool definition, and context with some code to link LLM calls, execute functions, and plug in the relevant context.

When building, we try to follow three core principles:
1.  **Maintain simplicity in your agent's design.** Start with simple prompts and add multi-step agentic systems only when simpler solutions fall short.
2.  **Prioritize transparency** by explicitly showing the agent’s planning steps.
3.  **Carefully craft your agent-computer interface (ACI)** through thorough tool documentation and testing.

By following these principles, you can create agents that are not only powerful but also reliable, maintainable, and trusted by their users.

### Acknowledgements

This work draws upon the experiences of building agents at Anthropic and the valuable insights shared by our customers, for which we're deeply grateful. Written by Erik Schluntz and Barry Zhang. Additional content and structure adapted from Harsh Pathak's "Mastering AI Agent Planning" and Guodong (Troy) Zhao's "Comprehensive guide to AI agents in 2025".