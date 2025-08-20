from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from .rag_chain import build_rag

class State(TypedDict):
    question: str
    draft: str
    final: Optional[str]
    attempts: int

def build_app():
    rag = build_rag()

    def answer_node(state: State) -> State:
        out = rag.invoke({"question": state["question"]})
        text = out if isinstance(out, str) else str(out)
        return {**state, "draft": text, "attempts": state["attempts"] + 1}

    def critic_node(state: State) -> State:
        draft = (state["draft"] or "").strip()
        ok = ("[doc" in draft) and (len(draft) > 40)
        return {**state, "final": draft if ok else None}

    def revise_node(state: State) -> State:
        stricter = state["question"] + (
            "\n\nIMPORTANT: If context is insufficient, respond exactly 'I don't know'. "
            "Always include [docN] citations."
        )
        out = rag.invoke({"question": stricter})
        text = out if isinstance(out, str) else str(out)
        return {**state, "draft": text}

    def route(state: State) -> str:
        if state["final"] is not None:
            return "finish"
        return "revise" if state["attempts"] < 2 else "finish"

    g = StateGraph(State)
    g.add_node("answer", answer_node)
    g.add_node("critic", critic_node)
    g.add_node("revise", revise_node)

    g.set_entry_point("answer")
    g.add_edge("answer", "critic")
    g.add_conditional_edges("critic", route, {"revise": "revise", "finish": END})
    g.add_edge("revise", "critic")
    return g.compile()
