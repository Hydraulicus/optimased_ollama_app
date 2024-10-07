import time
from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from utils.caching import cache_retrieval_chain

app = Flask(__name__)

folder_path = "db"
# cached_llm = Ollama(model="llama3.2")
cached_llm = Ollama(model="llama3")

embedding = FastEmbedEmbeddings()
embedding._model = embedding.model_dump().get("_model")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """
[INST] {input}
        Context: {context}
        Answer: 
[/INST] 
"""
)


# raw_prompt = PromptTemplate.from_template(
#     """
# <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
# [INST] {input}
#         Context: {context}
#         Answer:
# [/INST]
# """
# )



@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)

    # print("/ai response", response)

    response_answer = {"answer": response}
    return response_answer


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    start_time = time.time()  # Record the start time

    print(f"query: {query}")

    # TODO try to cache it too
    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    # TODO try to cache it too
    print("Creating chaine")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    # TODO try to cache it too
    document_chaine = create_stuff_documents_chain(cached_llm, raw_prompt)
    # chaine = create_retrieval_chain(retriever, document_chaine)
    chaine = cache_retrieval_chain(retriever, document_chaine)

    result = chaine.invoke({"input": query})

    end_time = time.time()  # Record the end time
    latency = end_time - start_time  # Calculate the latency
    print(f"Latency: {latency:.2f} seconds")  # Print the latency

    # print(result)

    sources = []
    for doc in result["context"]:
        # print("doc:")
        # print(doc.page_content)
        # print("Source: ", doc.metadata["source"])
        sources.append(
            {"source": doc.metadata["source"], "page_context": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
