# scripts/test_rag_smoke.py
#!/usr/bin/env python3
"""
RAG系統煙霧測試
"""

import requests
import tempfile
import os
import json


def test_rag_endpoints(api_base_url="http://localhost:8000/api/v1"):
    """Test RAG API endpoints"""

    print("🧪 Testing RAG endpoints...\n")

    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{api_base_url.replace('/api/v1', '')}/api/v1/health")
        if response.status_code == 200:
            print("   ✅ Health check passed")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

    # Test 2: RAG status
    print("2. Testing RAG status...")
    try:
        response = requests.get(f"{api_base_url}/rag/status")
        if response.status_code == 200:
            status = response.json()
            print(
                f"   ✅ Status: {status['total_documents']} docs, {status['total_chunks']} chunks"
            )
        else:
            print(f"   ❌ Status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Status error: {e}")
        return False

    # Test 3: Document upload
    print("3. Testing document upload...")
    try:
        # Create a test document
        test_content = """
        Artificial Intelligence (AI) is a branch of computer science that aims to create
        intelligent machines that can perform tasks that typically require human intelligence.

        Machine Learning is a subset of AI that focuses on the development of algorithms
        that can learn and make decisions from data without being explicitly programmed.

        Natural Language Processing (NLP) is another important area of AI that deals with
        the interaction between computers and human language.
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(test_content)
            temp_file_path = f.name

        try:
            with open(temp_file_path, "rb") as f:
                files = {"file": f}
                data = {
                    "collection_name": "test_collection",
                    "chunk_size": 200,
                    "chunk_overlap": 20,
                }

                response = requests.post(
                    f"{api_base_url}/rag/upload", files=files, data=data
                )

                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ Upload success: {result['total_chunks']} chunks")
                else:
                    print(
                        f"   ❌ Upload failed: {response.status_code} - {response.text}"
                    )
                    return False
        finally:
            os.unlink(temp_file_path)

    except Exception as e:
        print(f"   ❌ Upload error: {e}")
        return False

    # Test 4: RAG query
    print("4. Testing RAG query...")
    try:
        query_data = {
            "question": "What is artificial intelligence?",
            "collection_name": "test_collection",
            "top_k": 2,
            "max_length": 100,
            "temperature": 0.7,
        }

        response = requests.post(f"{api_base_url}/rag/ask", json=query_data)

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Query success")
            print(f"   Answer: {result['answer'][:100]}...")
            print(f"   Confidence: {result['confidence']:.1%}")
        else:
            print(f"   ❌ Query failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Query error: {e}")
        return False

    print("\n✅ All RAG tests passed!")
    return True


def main():
    """Main test function"""
    print("🧪 RAG Smoke Tests\n")

    # Test API endpoints
    if test_rag_endpoints():
        print("✅ RAG system is working correctly!")
        return True
    else:
        print("❌ RAG system has issues")
        return False


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
