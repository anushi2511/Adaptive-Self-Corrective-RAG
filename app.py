import streamlit as st

from app.retrieval import retrieve
from app.generation import generate_response
from app.evaluation import (
    evaluate_response,
    needs_rewrite,
)

st.set_page_config(
    page_title="Adaptive Self-Corrective RAG",
    layout="wide"
)

st.title(
    body="Adaptive Self-Corrective RAG",
    text_alignment="center"
)
st.write(
    "Ask questions based on your indexed documents."
)

query = st.text_input(
    "Enter your question:"
)

if st.button("Ask"):

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Retrieving documents..."):
        docs = retrieve(query)

    with st.spinner("Generating response..."):
        result = generate_response(
            query,
            docs
        )

    answer = result["answer"]
    context = result["context"]

    with st.spinner("Evaluating response..."):
        evaluation = evaluate_response(
            query,
            context,
            answer
        )

    st.subheader("Generated Response")
    st.write(answer)

    st.subheader("Evaluation Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Confidence",
            f"{evaluation['confidence']}%"
        )

    with col2:
        st.metric(
            "Hallucination",
            str(evaluation['hallucination'])
        )

    with col3:
        st.metric(
            "Rewrite Needed",
            str(evaluation['rewrite_needed'])
        )

    st.info(
        evaluation["reason"]
    )

    if needs_rewrite(evaluation):
        st.warning(
            "Self-correction recommended."
        )
    else:
        st.success(
            "Response accepted."
        )

    with st.expander(
        "Retrieved Context"
    ):
        for i, doc in enumerate(docs, start=1):
            st.markdown(
                f"**Chunk {i}:**"
            )
            st.write(doc)
            st.divider()