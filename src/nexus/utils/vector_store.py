from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    def __init__(self, collection_name="semantic_memory", persist_directory="./data/nexus/demo"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)

    def episodic_similarity_search(self, task_description, user_id, k=4, threshold=1.0):
        print(f"Episodic Search with filter - {user_id}")

        # 1. Get raw tuples (Doc, Score)
        docs_and_scores = self.vector_store.similarity_search_with_score(
            task_description,
            filter={"user_id": user_id},
            k=k
        )

        print("Episodic results before filtering:", docs_and_scores)

        # 2. Filter by threshold (Lower distance = Better match)
        # We unpack the tuple directly in the list comprehension
        filtered_results = [
            (doc, score) for doc, score in docs_and_scores
            if score <= threshold
        ]

        print(f"Episodic results matching threshold: {len(filtered_results)}")
        return filtered_results

    def semantic_similarity_search(self, description, k=4, threshold=3):
        print("Semantic Search with filter")

        # Get raw tuples
        docs_and_scores = self.vector_store.similarity_search_with_score(description, k=k)
        print("Semantic results before filtering:", docs_and_scores)

        # Filter by threshold
        filtered_results = [
            (doc, score) for doc, score in docs_and_scores
        ]

        return filtered_results
