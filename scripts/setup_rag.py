# scripts/setup_rag.py
#!/usr/bin/env python3
"""
RAG系統設置腳本
下載必要模型並進行初始化
"""

import os
import sys
from pathlib import Path
import subprocess


def setup_shared_cache():
    """Setup shared cache environment"""
    import pathlib, torch

    AI_CACHE_ROOT = os.getenv("AI_CACHE_ROOT", "/mnt/ai_warehouse/cache")

    for k, v in {
        "HF_HOME": f"{AI_CACHE_ROOT}/hf",
        "TRANSFORMERS_CACHE": f"{AI_CACHE_ROOT}/hf/transformers",
        "HF_DATASETS_CACHE": f"{AI_CACHE_ROOT}/hf/datasets",
        "HUGGINGFACE_HUB_CACHE": f"{AI_CACHE_ROOT}/hf/hub",
        "TORCH_HOME": f"{AI_CACHE_ROOT}/torch",
    }.items():
        os.environ[k] = v
        pathlib.Path(v).mkdir(parents=True, exist_ok=True)

    # Create RAG-specific directories
    rag_dirs = [
        f"{AI_CACHE_ROOT}/models/embeddings",
        f"{AI_CACHE_ROOT}/rag_collections",
        f"{AI_CACHE_ROOT}/datasets/rag_docs",
    ]
    for p in rag_dirs:
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)

    print(f"[RAG Setup] Cache root: {AI_CACHE_ROOT}")
    print(f"[RAG Setup] GPU available: {torch.cuda.is_available()}")
    return AI_CACHE_ROOT


def download_embedding_models():
    """Download embedding models for RAG"""
    models = [
        "BAAI/bge-base-en-v1.5",  # English + some multilingual
        "BAAI/bge-base-zh-v1.5",  # Chinese optimized (optional)
    ]

    print("Downloading embedding models...")
    try:
        from sentence_transformers import SentenceTransformer

        for model_name in models:
            print(f"  → {model_name}")
            try:
                model = SentenceTransformer(model_name, device="cpu")
                print(f"    ✅ Downloaded: {model_name}")
            except Exception as e:
                print(f"    ❌ Failed {model_name}: {e}")

    except ImportError:
        print("❌ sentence-transformers not installed")
        print("Run: pip install sentence-transformers")
        return False

    return True


def test_rag_system():
    """Test RAG system functionality"""
    print("\nTesting RAG system...")

    try:
        # Test imports
        from backend.core.vector_store import VectorStore
        from backend.core.document_processor import DocumentProcessor
        from backend.core.rag_pipeline import RAGPipeline

        # Test basic functionality
        vector_store = VectorStore()
        processor = DocumentProcessor()
        pipeline = RAGPipeline(vector_store)

        # Test with a simple document
        test_text = "This is a test document about artificial intelligence and machine learning."
        chunks = processor.split_text(test_text)

        vector_store.add_documents(
            collection_name="test",
            document_id="test_doc",
            texts=chunks,
            source="test.txt",
        )

        # Test query
        result = pipeline.query("What is this document about?", collection_name="test")

        print("✅ RAG system test successful")
        print(f"   Test query result: {result['answer'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False


def main():
    """Main setup function"""
    print("🔧 Setting up RAG system...\n")

    # Setup cache
    cache_root = setup_shared_cache()

    # Download models
    if not download_embedding_models():
        print("❌ Model download failed")
        return False

    # Test system
    if not test_rag_system():
        print("❌ System test failed")
        return False

    print("\n✅ RAG system setup complete!")
    print(f"   Cache root: {cache_root}")
    print("   Ready to use /rag/upload and /rag/ask endpoints")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
