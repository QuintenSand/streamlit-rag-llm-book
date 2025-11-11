from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model='llama3.2')

template = """
You are an expert in answering questions about a book

Here is some text from the book: {book}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n------------------------")
    question = input("Ask your question (q to quit): ")
    if question == "q":
        break
    
    book = retriever.invoke(question)
    result = chain.invoke({"book": book, "question": question})
    print(result)