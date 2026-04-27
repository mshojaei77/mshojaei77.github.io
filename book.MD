**Introduction**

Large language models have completely changed what software can do. Just a few years ago, they were mostly research experiments. Today, they're at the heart of apps that summarize long documents with proper citations, answer questions using your own data, automate messy workflows, run intelligent agents, and even handle natural, back-and-forth voice conversations in real time. Users now expect these AI capabilities as a normal part of any decent product, and the engineers who can actually deliver them reliably are in serious demand.

But there's a big difference between firing off a quick prompt to an LLM API and building something solid that people can trust in the real world. A clever prompt might impress in a demo, but production systems need a lot more: smart retrieval so the model stays grounded in real information, careful tool use with proper safeguards, solid ways to test and evaluate what you've built, efficient running and optimization, good observability when things go wrong, and serious attention to security and privacy. That's where real engineering comes in — turning flashy prototypes into dependable products.

**Hands-on AI Engineering** is for Python developers who already know how to ship software and now want to get really good at working with foundation models. I wrote it with a practical, application-first mindset: very little math, and a heavy focus on skills you can use right away. You'll learn to move comfortably between open-source models and hosted APIs, build RAG pipelines that actually stick to the source material instead of hallucinating, create agents that use tools safely (with human approval where it matters), figure out whether your system is truly solving the problem, optimize for speed and cost, and handle the extra headaches that come with realtime voice and multimodal setups.

Throughout the book, I treat LLMs like any other serious piece of production software — something that should be observable, testable, scalable, and maintainable. Every chapter gives you clear explanations paired with code you can run immediately on your laptop or in the cloud, all built around the Hugging Face tools and ecosystem that real teams actually use. I call out the trade-offs honestly: when a good prompt is enough, when you really need retrieval, when doing some targeted fine-tuning with PEFT makes sense, and when it's better to go with a cascaded speech stack versus a native one. The goal is to help you make smart decisions for whatever problem you're actually trying to solve.

The book follows a logical flow that matches how modern LLM systems are usually put together:

- **Foundations** helps you build solid mental models around tokenization, prompting, structured outputs, and the open model world.  
- **Text LLM Systems** takes you from basic prompting to full applications that use retrieval, tools, and proper evaluation.  
- **Audio and Realtime Systems** dives into speech — ASR, TTS, full-duplex conversations, handling interruptions, diarization, and multimodal voice agents — where the same engineering ideas show up but with new kinds of headaches around latency and natural turn-taking.  
- **Production, Security, and Portfolio** shows you how to move from working demos to systems you can actually ship, covering serving, quantization, observability, security, privacy, and a set of capstone projects you can put in your portfolio.

Whether you're an independent developer who wants to add smart features to your own projects, a software engineer moving into AI work, or a team lead responsible for delivering these capabilities, this book gives you a practical, evaluation-first way of working. By the time you finish, you'll have real artifacts — tested pipelines, clear failure analyses, deployment plans, voice system designs, and solid portfolio pieces — that show you can think and build like a proper AI engineer.

The field moves fast, with new models dropping all the time. But the core principles — build something, measure how it actually performs, understand where it breaks before you try to optimize it, and always treat the whole thing as real software — don't change. This book is designed to help you keep up while building systems that are worth keeping around.

Welcome to *Hands-on AI Engineering*. Let's get to work. 🚀

---

This version feels more natural and human — warmer tone, occasional contractions, simpler phrasing, and a personal touch ("I wrote it", "I call out the trade-offs") without sounding overly casual or salesy. It should read like something written by an experienced engineer sharing practical advice rather than generated corporate text. 

Want any small tweaks, or shall we move on to the next section (e.g., "How to Use This Book", Preface, or Chapter 1)?