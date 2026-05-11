- Book Style Guide

## 1. Book Identity

This is a hands-on technical book for practitioners.

The book should help readers build real systems, not just understand concepts. It should be practical, clear, structured, and realistic.

The tone should feel like an experienced engineer guiding a capable junior developer: direct, helpful, pragmatic, and honest about trade-offs.

Avoid hype, overpromising, academic heaviness, and unnecessary complexity.

---

## 2. Teaching Philosophy

The book follows a hands-on learning style:

1. Start with motivation and intuition.
2. Explain the core concept clearly.
3. Build a minimal working version.
4. Improve it with practical engineering considerations.
5. Discuss trade-offs, limitations, and failure modes.
6. End with exercises or a project.

Theory should support implementation. It should not dominate the chapter.

Readers should always understand:

- what problem is being solved,
- why the technique matters,
- how to implement it,
- what can go wrong,
- when to use it,
- when not to use it.

---

## 3. Book Progression

The book should increase complexity gradually.

Early chapters should establish foundations and simple workflows. Later chapters should add more advanced techniques, architecture, production concerns, and specialization.

Patterns introduced early should return later in more advanced forms. This makes the book feel like one coherent workshop instead of disconnected tutorials.

---

## 4. Reusable Chapter Template

Most chapters should follow this flexible structure:

```markdown
# Chapter [X]: [Title]
[A brif introduction here]
## [sub title]
## ...
## ...
## ...
## Exercise
```

---

## 5. Chapter Scope

Each chapter should focus on one main topic.

Avoid teaching too many new ideas at once. Mention related future topics only briefly when needed.

A chapter should leave the reader with:

- one clear mental model,
- one useful working pattern,
- one understanding of what breaks,
- one practical way to continue.

---

## 6. Code Policy

Code should be practical and runnable.

In the book:

- show only the code needed to explain the concept,
- avoid long file dumps,
- explain why each snippet matters,
- keep examples readable,
- prefer clarity over cleverness.

In the companion repository:

- provide complete runnable projects,
- include setup instructions,
- include dependency files,
- include environment examples,
- include troubleshooting notes when useful.

The book explains the ideas. The repository carries the full implementation.

---

## 7. Repository Policy

The repository is part of the learning experience.

It should closely match the book and make it easy for readers to run, modify, and extend the projects.

Keep repository structure clean, consistent, and easy to navigate.

Each project should include:

- clear setup instructions,
- required dependencies,
- safe handling of secrets or configuration,
- expected output or behavior,
- notes for common setup problems.

---

## 8. Diagrams and Visuals

Use visuals to explain systems, flows, and relationships.

Diagrams should be simple and purposeful. Every diagram should clarify something that prose alone would make harder to understand.

Avoid decorative diagrams.

---

## 9. Tables and Checklists

Use tables and checklists for decisions, trade-offs, comparisons, and readiness checks.

Do not use tables for long explanations.

Keep tables short and easy to scan.

---

## 10. Failure Modes

Every practical chapter should explain what commonly goes wrong.

Failure modes should be concrete and useful. They should help readers debug, make better design choices, and avoid fragile implementations.

When possible, include common errors, likely causes, and fixes.

---

## 11. Exercises and Projects

Every chapter should end with practical work.

Exercises should reinforce the chapter’s main skill and increase in difficulty when appropriate.

Projects should feel realistic, scoped, and testable.

Each project should define:

- the goal,
- the constraints,
- the expected behavior,
- what is out of scope,
- how the reader knows it works.

---

## 12. Production Mindset

The book should distinguish between:

- a prototype,
- a production-minded prototype,
- a production-ready component,
- a production-grade system.

Do not call something production-grade unless it handles reliability, security, monitoring, failure recovery, and operational concerns.

Encourage readers to build the simplest useful version first, then upgrade only when the problem requires it.

---

## 13. Tone Rules

Prefer direct, practical language.

Use phrases that focus on engineering judgment:

- what works,
- what breaks,
- what costs money,
- what adds complexity,
- what is good enough for now,
- what requires an upgrade.

Avoid hype words, vague claims, and exaggerated promises.

The writing should feel calm, useful, and grounded.

---

## 14. References

Use reliable sources.

Prefer:

- official documentation,
- original research papers,
- official benchmark descriptions,
- high-quality engineering references.

Avoid weak sources, SEO-driven articles, and unsupported claims.

References should support durable concepts, not temporary hype.

---

## 15. Final Principle

The book should feel like one coherent hands-on workshop.

Keep the structure consistent, the progression gradual, the examples practical, and the tone honest.

Consistency builds trust.  
Progression builds skill.  
Practicality keeps the reader moving.