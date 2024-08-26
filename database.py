import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import os
import shutil
import nltk
nltk.download('averaged_perceptron_tagger')

DATASET_PATH = r"C:\Users\leduc\OneDrive\Desktop\NLP\UIT-chatbot\RAG_data"
CHROMA_PATH = "chroma_database"
sample_path = r"C:\Users\leduc\OneDrive\Desktop\NLP\UIT-chatbot\sample_RAG_data"


def main():
    docs = load_docs()
    chunks = splitting_text(docs)
    save_database(chunks)



def split_into_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def load_docs():
    try:
        documents = []
        for filename in os.listdir(DATASET_PATH):
            if filename.endswith(".txt"):
                file_path = os.path.join(DATASET_PATH, filename)
                index = int(filename.split(".")[0].split("_")[1])
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append(Document(page_content=content, metadata={"source": file_path, "ids": index}))
        return documents
    except Exception as e:
        print(f"An error occurred while loading documents: {e}")
        return []

def splitting_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split  {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def save_database(chunks):
    if(os.path.exists(CHROMA_PATH)):
        shutil.rmtree(CHROMA_PATH)

    
    model_name="hiieu/halong_embedding"
    model_kwargs = {
    'device': 'cuda',
    'trust_remote_code':True
    }

    embedding = HuggingFaceEmbeddings(model_name = model_name, model_kwargs = model_kwargs)

    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding, collection_metadata={"hnsw:space": "cosine"})

    max_batch_size = 5461


    for batch in split_into_batches(chunks, max_batch_size):
        db.add_documents(batch)

    
    # db = Chroma.from_documents(
    #     chunks,
    #     embedding,
    #     persist_directory=CHROMA_PATH
    # )

    # db.persit()
    print(f"Saved {len(chunks)} to db {CHROMA_PATH}")

if __name__ == "__main__":
    main()

