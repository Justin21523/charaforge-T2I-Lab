# backend/core/vector_store.py
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    """FAISS-based vector storage for RAG"""

    def __init__(
        self, model_name: str = "BAAI/bge-base-en-v1.5", cache_dir: str = None
    ):
        self.cache_dir = (
            cache_dir or os.getenv("AI_CACHE_ROOT", "/tmp") + "/models/embeddings"
        )
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        # Load embedding model
        self.embedding_model = SentenceTransformer(
            model_name, cache_folder=self.cache_dir, device="auto"
        )
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Storage for collections
        self.collections: Dict[str, Dict] = {}
        self.storage_path = Path(self.cache_dir) / "rag_collections"
        self.storage_path.mkdir(exist_ok=True)

        self._load_collections()

    def _load_collections(self):
        """Load existing collections from disk"""
        for collection_dir in self.storage_path.iterdir():
            if collection_dir.is_dir():
                try:
                    self._load_collection(collection_dir.name)
                except Exception as e:
                    logging.warning(
                        f"Failed to load collection {collection_dir.name}: {e}"
                    )

    def _load_collection(self, collection_name: str):
        """Load a specific collection from disk"""
        collection_path = self.storage_path / collection_name

        # Load FAISS index
        index_path = collection_path / "index.faiss"
        if not index_path.exists():
            return

        index = faiss.read_index(str(index_path))

        # Load metadata
        metadata_path = collection_path / "metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.collections[collection_name] = {
            "index": index,
            "texts": metadata["texts"],
            "sources": metadata["sources"],
            "document_ids": metadata["document_ids"],
        }

        logging.info(
            f"Loaded collection '{collection_name}' with {index.ntotal} vectors"
        )

    def _save_collection(self, collection_name: str):
        """Save collection to disk"""
        if collection_name not in self.collections:
            return

        collection_path = self.storage_path / collection_name
        collection_path.mkdir(exist_ok=True)

        collection = self.collections[collection_name]

        # Save FAISS index
        faiss.write_index(collection["index"], str(collection_path / "index.faiss"))

        # Save metadata
        metadata = {
            "texts": collection["texts"],
            "sources": collection["sources"],
            "document_ids": collection["document_ids"],
        }
        with open(collection_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def add_documents(
        self, collection_name: str, document_id: str, texts: List[str], source: str
    ):
        """Add documents to collection"""
        if collection_name not in self.collections:
            # Create new collection
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
            self.collections[collection_name] = {
                "index": index,
                "texts": [],
                "sources": [],
                "document_ids": [],
            }

        collection = self.collections[collection_name]

        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        collection["index"].add(embeddings)

        # Store metadata
        collection["texts"].extend(texts)
        collection["sources"].extend([source] * len(texts))
        collection["document_ids"].extend([document_id] * len(texts))

        # Save to disk
        self._save_collection(collection_name)

        logging.info(f"Added {len(texts)} chunks to collection '{collection_name}'")

    def search(
        self, collection_name: str, query: str, top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """Search for relevant documents"""
        if collection_name not in self.collections:
            return []

        collection = self.collections[collection_name]

        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )
        query_embedding = query_embedding.astype(np.float32)

        # Search
        scores, indices = collection["index"].search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                text = collection["texts"][idx]
                source = collection["sources"][idx]
                results.append((text, source, float(score)))

        return results

    def get_collections_info(self) -> Dict[str, int]:
        """Get information about all collections"""
        info = {}
        for name, collection in self.collections.items():
            info[name] = collection["index"].ntotal
        return info
