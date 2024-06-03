from typing import List

import pandas as pd
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def get_retriever(job_postings: pd.DataFrame) -> VectorIndexRetriever:
    documents = []
    for index, row in job_postings.iterrows():
        documents.append(
            Document(
                text=row["job_desc"],
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

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=10,
    )

    return retriever


def nodes_to_postings(nodes: List[NodeWithScore]) -> pd.DataFrame:
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
