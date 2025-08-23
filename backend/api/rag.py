# backend/api/rag.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path
import logging

from backend.schemas.rag import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGStatusResponse,
)
from backend.core.document_processor import DocumentProcessor
from backend.core.vector_store import VectorStore
from backend.core.rag_pipeline import RAGPipeline

router = APIRouter()

# Global instances
vector_store = VectorStore()
rag_pipeline = RAGPipeline(vector_store)


@router.post("/rag/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = "default",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    """Upload and process document for RAG"""
    try:
        # Validate file type
        if not file.filename.lower().endswith((".txt", ".pdf")):
            raise HTTPException(
                status_code=400, detail="Only TXT and PDF files are supported"
            )

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Process document
            processor = DocumentProcessor(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            document_id, chunks = processor.process_document(tmp_file_path)

            # Add to vector store
            vector_store.add_documents(
                collection_name=collection_name,
                document_id=document_id,
                texts=chunks,
                source=file.filename,
            )

            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                total_chunks=len(chunks),
                collection_name=collection_name,
            )

        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)

    except Exception as e:
        logging.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/ask", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """Query the RAG system"""
    try:
        result = rag_pipeline.query(
            question=request.question,
            collection_name=request.collection_name,
            top_k=request.top_k,
            max_length=request.max_length,
            temperature=request.temperature,
        )

        return RAGQueryResponse(**result)

    except Exception as e:
        logging.error(f"Error processing RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rag/status", response_model=RAGStatusResponse)
async def rag_status():
    """Get RAG system status"""
    try:
        collections_info = vector_store.get_collections_info()

        total_documents = (
            len(
                set().union(
                    *[
                        vector_store.collections[name]["document_ids"]
                        for name in collections_info.keys()
                    ]
                )
            )
            if collections_info
            else 0
        )

        total_chunks = sum(collections_info.values())

        return RAGStatusResponse(
            collections=list(collections_info.keys()),
            total_documents=total_documents,
            total_chunks=total_chunks,
        )

    except Exception as e:
        logging.error(f"Error getting RAG status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rag/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a document collection"""
    try:
        if collection_name in vector_store.collections:
            # Remove from memory
            del vector_store.collections[collection_name]

            # Remove from disk
            collection_path = vector_store.storage_path / collection_name
            if collection_path.exists():
                import shutil

                shutil.rmtree(collection_path)

            return {
                "status": "success",
                "message": f"Collection '{collection_name}' deleted",
            }
        else:
            raise HTTPException(
                status_code=404, detail=f"Collection '{collection_name}' not found"
            )

    except Exception as e:
        logging.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
