from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnableParallel, RunnableLambda
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from .ingest import build_retriever
from .config import GEN_MODEL_ID, CONTEXT_CHARS, MAX_TOTAL_CONTEXT

PROMPT = PromptTemplate.from_template(
    """Answer the user using ONLY the context.
If the context is not enough, answer exactly: I don't know.

Question: {question}

Context:
{context}

Rules:
- Cite sources as [docN] where N is the context order.
- Return ONLY the final answer (no meta-instructions, no headings).

Answer:"""
)

def _format_context(docs):
    # Shorten each chunk and cap the total length to stay within model limits
    parts, total = [], 0
    for i, d in enumerate(docs):
        snippet = (d.page_content or "")[:CONTEXT_CHARS]
        parts.append(f"[doc{i+1}] {snippet}")
        total += len(snippet)
        if total >= MAX_TOTAL_CONTEXT:
            break
    return "\n\n".join(parts)

def load_llm():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    # keep inputs bounded
    tok.model_max_length = 512
    tok.truncation_side = "right"
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_ID)
    text2text = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=200,
        truncation=True,       # enforce truncation if inputs get long
    )
    return HuggingFacePipeline(pipeline=text2text)

def build_rag():
    retriever = build_retriever()
    llm = load_llm()

    # IMPORTANT: pass only the string question into the retriever
    gather = RunnableParallel(
        docs=lambda x: retriever.invoke(x["question"]),
        question=lambda x: x["question"],
    )

    to_prompt = RunnableLambda(
        lambda x: {"question": x["question"], "context": _format_context(x["docs"])}
    )

    return gather | to_prompt | PROMPT | llm
