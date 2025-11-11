from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import pymupdf4llm
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

file = "data/Hands-On_Large_Language_Models.pdf"

doc = pymupdf4llm.to_markdown(file)

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
)

chunks = text_splitter.split_text(doc)

print(chunks)

embeddings = OllamaEmbeddings(model = 'mxbai-embed-large')

db_location = './chroma_langchain_db'
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, text in enumerate(chunks):
        document = Document(
            page_content=text,
            id = str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="text_book",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids = ids)

retriever = vector_store.as_retriever(
    search_kwargs = {"k": 1}
)