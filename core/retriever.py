from typing import List

import pandas as pd
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def get_retriever(job_postings: pd.DataFrame) -> VectorIndexRetriever:
    "Composes the original job postings to a retriever that enables lookup of postings similar to a query."

    # The job postings are converted into LamaIndex document objects.
    # More about it here: https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/
    documents = []
    for _, row in job_postings.iterrows():
        documents.append(
            Document(
                text=row["job_desc"],
                # type: ignore[call-arg] # metadata argument is present in the documentation:
                # https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/
                metadata={
                    "job_title": row["job_title"],
                    "company": row["company"],
                    "location": row["location"],
                },
                metadata_seperator="::",
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
        )

    # We selected this embedings mostly based on thier efficiency. The embeding is required for indexing process
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Indexing is required to enable a fast retrival of documents (job offers) matching the user query.
    # Indexing process includes embedding the text of the documents, and then putting them into the organized index structure, which enables fast search.
    # To build the index and store it we use VectorStoreIndex class from LamaIndex: https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    # We also declare a retriver which will be later used to compares the job description embedding against the job offers embeddings in the index to find a few job offers,
    # based on which the recomendation will be generated: https://ts.llamaindex.ai/api/classes/VectorIndexRetriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    return retriever


def nodes_to_postings(nodes: List[NodeWithScore]) -> pd.DataFrame:
    "A helper function transforming job postings nodes to the original representation."
    return pd.DataFrame(
        (
            (
                node.__dict__["node"].__dict__["metadata"]["job_title"],
                node.__dict__["node"].__dict__["metadata"]["company"],
                node.__dict__["node"].__dict__["text"],
                node.__dict__["node"].__dict__["metadata"]["location"],
                node.__dict__["score"],
            )
            for node in nodes
        ),
        columns=("job_title", "company", "job_desc", "location", "score"),
    )
