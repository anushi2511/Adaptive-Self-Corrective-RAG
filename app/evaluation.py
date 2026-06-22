import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

from app.retrieval import retrieve
from app.generation import generate_response

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

def evaluate_response(query, context, generated_response):
    prompt = f"""Evaluate the following response based on the context provided.\n\nContext:\n{context}\n\nQuestion: {query}\nResponse: {generated_response}\n\nEvaluate the answer based on 1. Factual correctness\n2. Grounding in the retrieved context\n3. Completeness\n4. Hallucination risk\n
    Return ONLY valid JSON in the following format:
    {{
        "confidence": integer,
        "hallucination": true/false,
        "rewrite_needed": true/false,
        "reason": "short explanation"
    }}

    Rules:
    confidence must be between 0 and 100.
    hallucination = true if answer contains unsupported claims.
    rewrite_needed = true if retrieval should be improved.
    Return ONLY JSON."""

    try:
        evaluation = model.generate_content(prompt)
        text = evaluation.text.strip()
        text = text.replace("```json", "").replace("```", "")
        text = text.strip()
        result = json.loads(text)
        return result

    except Exception as e:
        print("Error during evaluation:", e)
        return {
            "confidence": 0,
            "hallucination": True,
            "rewrite_needed": True,
            "reason": "Evaluation failed due to error."
        }

def needs_rewrite(evaluation_result):
    return (
        evaluation_result["confidence"] < 80 or
        evaluation_result["hallucination"] or
        evaluation_result["rewrite_needed"]
    )

def print_evaluation(evaluation_result):
    print("Evaluation Result:")
    print(f"Confidence: {evaluation_result['confidence']}")
    print(f"Hallucination: {evaluation_result['hallucination']}")
    print(f"Rewrite Needed: {evaluation_result['rewrite_needed']}")
    print(f"Reason: {evaluation_result['reason']}")

def rewrite_query(query):

    prompt = f"""
    Rewrite the following query to improve document retrieval.

    Original Query:
    {query}

    Return only the rewritten query.
    """

    response = model.generate_content(prompt)

    return response.text.strip()