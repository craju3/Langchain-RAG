# from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import FAISS
# import faiss
# import numpy as np
# import os

# # Initialize SentenceTransformer model
# model_name = "distilbert-base-nli-stsb-mean-tokens"  # Use a SentenceTransformers model
# embedding_model = SentenceTransformer(model_name)

# # Directory containing your text files
# data_dir = "data/"
# files = [
#     "code_of_conduct.txt",
#     "it_security_policy.txt",
#     "leave_policy.txt",
#     "remote_work_policy.txt"
# ]

# # Read text files and create documents
# documents = []
# docstore = {}
# for i, filename in enumerate(files):
#     with open(os.path.join(data_dir, filename), 'r') as file:
#         text = file.read()
#         docstore[i] = {"text": text, "metadata": {"source": filename}}
#         documents.append(text)

# # Compute embeddings
# embeddings = np.array(embedding_model.encode(documents))

# # Create FAISS index
# dimension = embeddings.shape[1]  # Assuming the embeddings have shape (n_documents, dimension)
# index = faiss.IndexFlatL2(dimension)
# index.add(embeddings)

# # Create index-to-docstore ID mapping
# index_to_docstore_id = {i: i for i in range(len(documents))}

# # Define embedding function
# def embedding_function(texts):
#     return np.array(embedding_model.encode(texts))

# # Initialize FAISS with the created components
# vector_store = FAISS(index=index, embedding_function=embedding_function, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# # Example usage
# def ask_question(query):
#     # Embed query and search in FAISS
#     query_embedding = embedding_function([query])
#     D, I = vector_store.index.search(np.array(query_embedding), k=1)
#     doc_id = vector_store.index_to_docstore_id[I[0][0]]
#     return vector_store.docstore[doc_id]["text"]

# # Example call
# if __name__ == "__main__":
#     question = "What is the leave policy?"
#     answer = ask_question(question)
#     print(answer)

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import faiss
import numpy as np
import os
import openai

# Initialize SentenceTransformer model
model_name = "distilbert-base-nli-stsb-mean-tokens"  # Use a SentenceTransformers model
embedding_model = SentenceTransformer(model_name)

# Directory containing your text files
data_dir = "data/"
files = [
    "code_of_conduct.txt",
    "it_security_policy.txt",
    "leave_policy.txt",
    "remote_work_policy.txt"
]

# Read text files and create documents
documents = []
docstore = {}
for i, filename in enumerate(files):
    with open(os.path.join(data_dir, filename), 'r') as file:
        text = file.read()
        docstore[i] = {"text": text, "metadata": {"source": filename}}
        documents.append(text)

# Compute embeddings
embeddings = np.array(embedding_model.encode(documents))
print(f"Document embeddings shape: {embeddings.shape}")

# Create FAISS index
dimension = embeddings.shape[1]  # Assuming the embeddings have shape (n_documents, dimension)
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Create index-to-docstore ID mapping
index_to_docstore_id = {i: i for i in range(len(documents))}

# Define embedding function
def embedding_function(texts):
    return np.array(embedding_model.encode(texts))

# Initialize FAISS with the created components
vector_store = FAISS(index=index, embedding_function=embedding_function, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

# OpenAI API key
openai.api_key = 'your-openai-api-key'

# Define the question answering function
def ask_question(query):
    # Embed query and search in FAISS
    query_embedding = embedding_function([query])
    print(f"Query embedding shape: {query_embedding.shape}")

    D, I = vector_store.index.search(query_embedding, k=5)  # Retrieve top 5 documents
    print(f"Distances: {D}")
    print(f"Indexes: {I}")

    # Filter out invalid indexes
    valid_indexes = [i for i in I[0] if i in vector_store.index_to_docstore_id]
    print(f"Valid indexes: {valid_indexes}")

    if not valid_indexes:
        return "No relevant documents found."

    # Retrieve and rank documents
    ranked_docs = []
    for i in valid_indexes:
        doc_id = vector_store.index_to_docstore_id[i]
        ranked_docs.append(vector_store.docstore[doc_id]['text'])

    # Combine documents into a context for the generative model
    context = "\n\n".join(ranked_docs).strip()
    if not context:
        return "No relevant documents found."

    # Generate the answer using OpenAI's GPT model
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use the desired model
        prompt=f"Based on the following documents:\n{context}\n\nAnswer the question: {query}",
        max_tokens=200,
        temperature=0.7
    )

    # Return the generated answer
    return response.choices[0].text.strip()

# Example call
if __name__ == "__main__":
    question = "What is the leave policy?"
    answer = ask_question(question)
    print(answer)



# Example call
if __name__ == "__main__":
    question = "What is the leave policy?"
    answer = ask_question(question)
    print(answer)
