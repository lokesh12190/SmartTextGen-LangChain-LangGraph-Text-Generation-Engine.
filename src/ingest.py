import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from .config import DOCS_PATH, DB_PATH, EMB_MODEL, RETRIEVER_K

def build_retriever(k: int | None = None):
    # 1) Load PDFs
    docs = []
    for pdf in glob.glob(str(DOCS_PATH / "*.pdf")):
        docs.extend(PyPDFLoader(pdf).load())

    if not docs:
        raise RuntimeError(f"No PDFs found in {DOCS_PATH}. Put at least one PDF there.")

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    # 3) Embed + store in Chroma
    embedder = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    vectordb = Chroma.from_documents(
        splits,
        embedding=embedder,
        persist_directory=str(DB_PATH),
    )
    vectordb.persist()

    # 4) Return retriever interface
    return vectordb.as_retriever(search_kwargs={"k": k or RETRIEVER_K})
