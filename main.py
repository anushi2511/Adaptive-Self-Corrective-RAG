
from app.retrieval import retrieve
from app.generation import generate_response

if __name__ == "__main__":

    query = "What is the main topic of the document?"
    retrieved_docs = retrieve(query)

    answer = generate_response(query, retrieved_docs)
    print("Answer:", answer)