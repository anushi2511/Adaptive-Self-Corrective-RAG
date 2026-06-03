
from app.retrieval import retrieve
from app.generation import generate_response
from app.evaluation import evaluate_response, needs_rewrite, print_evaluation

if __name__ == "__main__":

    query = "What is the main topic of the document?"
    retrieved_docs = retrieve(query)

    result = generate_response(query, retrieved_docs)
    answer = result["answer"]
    context = result["context"]

    evaluation_result = evaluate_response(query, context, answer)

    print_evaluation(evaluation_result)
    print("\n---\n")

    if needs_rewrite(evaluation_result):
        print("The response needs correction.")
    else:
        print("The response is satisfactory. No rewrite needed.")

    print (answer)
    