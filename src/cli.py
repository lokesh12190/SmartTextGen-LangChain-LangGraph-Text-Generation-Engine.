from .graph_app import build_app

def main():
    app = build_app()
    print("Smart Research Assistant (local CPU). Ask about your PDFs. Type 'quit' to exit.\n")
    while True:
        q = input("Q> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        if not q:
            continue

        state = {"question": q, "draft": "", "final": None, "attempts": 0}
        # Optional extra safety: recursion limit
        out = app.invoke(state, config={"recursion_limit": 12})

        answer = out.get("final") or out.get("draft", "")
        # Guard against meta-echo
        if not answer or "cite sources" in answer.lower():
            answer = "I don't know."

        print("\n--- Answer ---")
        print(answer, "\n")

if __name__ == "__main__":
    main()
