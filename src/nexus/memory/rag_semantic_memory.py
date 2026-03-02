import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cloud_services import LlamaParse

from src.nexus.utils.constants import CHUNK_SIZE, CHUNK_OVERLAP, SEMANTIC_COLLECTION
from src.nexus.utils.vector_store import VectorStore


class MultimodalSemanticMemory:
    def __init__(self):
        # 1. Initialize LlamaParse (The Cloud Parser)
        # It handles OCR, tables, and images automatically in the cloud.
        self.parser = LlamaParse(
            api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
            parsing_instruction="This is a professional document. Extract all text, maintain table formatting, and describe any charts or images in detail.",
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt4o"
        )

        # 2. Setup Embeddings & Vector Store
        self.vector_store = VectorStore(SEMANTIC_COLLECTION)

        # 3. Setup Chunking Logic
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def ingest_file(self, file_path: str):

        # Step A: Parse file in the cloud
        # This one line replaces all the complex local Docling/RapidOCR setup
        llama_docs = self.parser.load_data(file_path)

        all_chunks = []

        for l_doc in llama_docs:
            # Step B: Split the markdown content into overlapping chunks
            page_text = l_doc.text
            metadata = {
                "source": file_path,
                "page_no": l_doc.metadata.get("page_number", "unknown")
            }
            # Create LangChain documents from the split text
            chunks = self.text_splitter.create_documents([page_text], metadatas=[metadata])
            all_chunks.extend(chunks)

        # Step C: Save to persistent storage
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            print(f"Successfully indexed {len(all_chunks)} chunks.")
        else:
            print("No content extracted from file.")

    def semantic_similarity_search(self, query: str, k: int = 5, threshold: float = 1.2):
        return self.vector_store.semantic_similarity_search(query, k=k, threshold=threshold)
