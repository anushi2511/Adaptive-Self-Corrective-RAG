import os
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key = GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

def build_context(query, retrieved_chunks):
    return "\n\n".join(retrieved_chunks) + "\n\n" + query

def generate_response(query, retrieved_chunks):
    context = build_context(query, retrieved_chunks)
    prompt = f"Use only the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text