import os
import config
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # New Import
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

class VectorDbManager:
    __client: QdrantClient
    __dense_embeddings: GoogleGenerativeAIEmbeddings # Updated Type Hint

    def __init__(self):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if qdrant_url and qdrant_api_key:
            print("Connecting to Qdrant Cloud...")
            self.__client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            print(f"Connecting to Local Qdrant at: {config.QDRANT_DB_PATH}")
            self.__client = QdrantClient(path=config.QDRANT_DB_PATH)

        # Offload embedding math to Google's API (Saves ~500MB RAM)
        self.__dense_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

    def create_collection(self, collection_name):
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            
            # Simplified to Dense-only to ensure stability on Free Tier
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=768, # Google embedding-001 size
                    distance=qmodels.Distance.COSINE
                )
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            print(f"✓ Collection already exists: {collection_name}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                client=self.__client,
                collection_name=collection_name,
                embedding=self.__dense_embeddings,
                retrieval_mode=RetrievalMode.DENSE # Switched from HYBRID to DENSE
            )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")
