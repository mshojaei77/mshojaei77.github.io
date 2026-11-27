from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ResearchState(TypedDict, total=False):
    question: str
    protocol: dict[str, object]
    status: str
    reviewer_notes: str


def draft_protocol(state: ResearchState) -> dict[str, object]:
    """Create a placeholder protocol; replace this with domain-specific logic."""
    return {
        "protocol": {
            "question": state["question"],
            "sources": [],
            "eligibility_criteria": [],
            "analysis_plan": [],
        },
        "status": "awaiting_human_review",
    }


def review_protocol(state: ResearchState):
    """Pause until a human approves, revises, or rejects the protocol."""
    response = interrupt(
        {
            "protocol": state["protocol"],
            "allowed_decisions": ["approve", "revise", "reject"],
        }
    )
    decision = response.get("decision")
    notes = response.get("notes", "")

    if decision == "approve":
        return Command(
            update={"status": "approved", "reviewer_notes": notes},
            goto="literature_search",
        )
    if decision == "revise":
        return Command(
            update={"status": "needs_revision", "reviewer_notes": notes},
            goto=END,
        )
    return Command(
        update={"status": "rejected", "reviewer_notes": notes},
        goto=END,
    )


def literature_search(state: ResearchState) -> dict[str, str]:
    """Placeholder for a bounded, source-aware search adapter."""
    return {"status": "ready_for_literature_search"}


def build_graph():
    builder = StateGraph(ResearchState)
    builder.add_node("draft_protocol", draft_protocol)
    builder.add_node("review_protocol", review_protocol)
    builder.add_node("literature_search", literature_search)
    builder.add_edge(START, "draft_protocol")
    builder.add_edge("draft_protocol", "review_protocol")
    return builder.compile(checkpointer=InMemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "research-example-1"}}
    paused = graph.invoke(
        {"question": "How should a review of X define eligible studies?"},
        config,
    )
    assert "__interrupt__" in paused

    resumed = graph.invoke(
        Command(resume={"decision": "approve", "notes": "Protocol reviewed."}),
        config,
    )
    assert resumed["status"] == "ready_for_literature_search"
    print(resumed["status"])


if __name__ == "__main__":
    main()
