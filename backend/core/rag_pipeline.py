# backend/core/rag_pipeline.py
from typing import List, Tuple
import logging
from backend.core.vector_store import VectorStore
from backend.core.pipeline_loader import get_chat_pipeline


class RAGPipeline:
    """RAG query pipeline combining retrieval and generation"""

    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()
        self.chat_pipeline = None

    def _get_chat_pipeline(self):
        """Lazy load chat pipeline"""
        if self.chat_pipeline is None:
            self.chat_pipeline = get_chat_pipeline()
        return self.chat_pipeline

    def query(
        self,
        question: str,
        collection_name: str = "default",
        top_k: int = 3,
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> dict:
        """Perform RAG query"""

        # 1. Retrieve relevant documents
        search_results = self.vector_store.search(collection_name, question, top_k)

        if not search_results:
            return {
                "question": question,
                "answer": "I couldn't find relevant information to answer your question.",
                "relevant_chunks": [],
                "sources": [],
                "confidence": 0.0,
                "model_used": "rag-pipeline",
            }

        # 2. Prepare context
        relevant_chunks = [result[0] for result in search_results]
        sources = [result[1] for result in search_results]

        context = "\n\n".join([f"Document: {chunk}" for chunk in relevant_chunks])

        # 3. Generate answer using context
        prompt = f"""Based on the following context, please answer the user's question. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

        chat_pipeline = self._get_chat_pipeline()

        try:
            response = chat_pipeline(
                prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=chat_pipeline.tokenizer.eos_token_id,
            )

            # Extract answer from response
            full_response = response[0]["generated_text"]
            answer = full_response.split("Answer:")[-1].strip()

            # Calculate confidence based on search scores
            avg_score = sum(result[2] for result in search_results) / len(
                search_results
            )
            confidence = min(avg_score, 0.95)  # Cap at 95%

            return {
                "question": question,
                "answer": answer,
                "relevant_chunks": relevant_chunks,
                "sources": sources,
                "confidence": confidence,
                "model_used": "rag-pipeline",
            }

        except Exception as e:
            logging.error(f"Error generating RAG response: {e}")
            return {
                "question": question,
                "answer": f"Error generating response: {str(e)}",
                "relevant_chunks": relevant_chunks,
                "sources": sources,
                "confidence": 0.0,
                "model_used": "rag-pipeline",
            }
