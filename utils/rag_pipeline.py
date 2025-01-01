# utils/rag_pipeline.py
import openai
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def retrieve_and_generate(news_texts: list[str], query: str) -> str:
    """
    Demo function using LangChain to retrieve from local texts and then generate an answer.
    """

    # 1. Load documents
    # For demonstration, convert each news_text to a doc
    docs = [TextLoader.from_text(t) for t in news_texts]

    # 2. Create embeddings, store in FAISS
    embeddings = OpenAIEmbeddings(openai_api_key="YOUR_OPENAI_API_KEY")
    vectordb = FAISS.from_documents(docs, embeddings)

    # 3. Setup retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.7, openai_api_key="YOUR_OPENAI_API_KEY"),
        retriever=vectordb.as_retriever()
    )

    # 4. Query and return
    return qa_chain.run(query)
