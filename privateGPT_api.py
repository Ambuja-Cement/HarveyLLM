#!/usr/bin/env python3
from constants import CHROMA_SETTINGS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os
import argparse
import time
from flask import Flask, request, jsonify

app = Flask(__name__)

model = os.environ.get("MODEL", "mistral")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

# Initialize components outside of the Flask routes
args = parse_arguments()
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
llm = Ollama(model=model, callbacks=[StreamingStdOutCallbackHandler()])
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Empty or missing 'query' field"})

    # Get the answer from the chain
    start = time.time()
    res = qa(query)
    answer, docs = res['result'], [
        ] if args.hide_source else res['source_documents']
    end = time.time()

    # Format the response
    response = {
        "query": query,
        "answer": answer,
        "source_documents": [
            {"source": document.metadata["source"], "content": document.page_content}
            for document in docs
        ],
        "processing_time": end - start
    }
    return jsonify(response)



if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)

