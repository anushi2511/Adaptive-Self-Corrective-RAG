import streamlit as st

import os

from app.retrieval import retrieve, ingest_docs
from app.generation import generate_response
from app.evaluation import (
    evaluate_response,
    needs_rewrite,
    rewrite_query,
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

with st.sidebar:

    st.header("Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        os.makedirs("data", exist_ok=True)

        file_path = os.path.join(
            "data",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Index Document"):

            with st.spinner(
                "Processing and indexing document..."
            ):
                try:
                    chunk_count = ingest_docs(file_path)
                    st.success(
                        f"{uploaded_file.name} indexed successfully, ({chunk_count} chunks uploaded.)"
                    )
                except Exception as err:
                    st.error("Indexing failed")
                    st.exception(err)

    st.divider()

    st.subheader("Indexed Documents")

    if os.path.exists("data"):

        files = [
            f for f in os.listdir("data")
            if f.endswith(".pdf")
        ]

        if files:
            for file in files:
                st.write(f"{file}")
        else:
            st.write("No documents uploaded.")

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

    original_confidence = evaluation["confidence"]

    if needs_rewrite(evaluation):

        st.warning(
            "Self-correction required."
        )

        current_query = query
        max_iterations = 2

        for iteration in range(max_iterations):

            with st.spinner(
                f"Self-Correction Iteration {iteration + 1}"
            ):

                # Rewrite Query
                rewritten_query = rewrite_query(
                    current_query
                )

                # Retrieve New Context
                docs = retrieve(
                    rewritten_query
                )

                # Generate New Answer
                result = generate_response(
                    rewritten_query,
                    docs
                )

                answer = result["answer"]
                context = result["context"]

                # Evaluate Again
                evaluation = evaluate_response(
                    rewritten_query,
                    context,
                    answer
                )

                current_query = rewritten_query

                # Stop if confidence is acceptable
                if not needs_rewrite(
                    evaluation
                ):
                    break

        st.success(
            "Self-correction completed."
        )

        st.subheader(
            "Self-Correction Details"
        )

        st.markdown(
            f"""
            **Original Query**

            {query}

            **Refined Query**

            {current_query}
            """
        )

        st.subheader(
            "Confidence Improvement"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Before",
                f"{original_confidence}%"
            )

        with col2:
            st.metric(
                "After",
                f"{evaluation['confidence']}%",
                delta=
                evaluation["confidence"]
                - original_confidence
            )

        # Final Verified Answer
        st.subheader(
            "Final Verified Response"
        )

        st.write(
            answer
        )

        # Final Evaluation Report
        st.subheader(
            "Final Evaluation Report"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Confidence",
                f"{evaluation['confidence']}%"
            )

        with col2:
            st.metric(
                "Hallucination",
                str(
                    evaluation["hallucination"]
                )
            )

        with col3:
            st.metric(
                "Rewrite Needed",
                str(
                    evaluation["rewrite_needed"]
                )
            )

        st.info(
            evaluation["reason"]
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