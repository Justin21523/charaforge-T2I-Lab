# backend/schemas/rag.py
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import UploadFile


class DocumentUploadRequest(BaseModel):
    collection_name: str = Field(
        default="default", description="Document collection name"
    )
    chunk_size: int = Field(default=512, description="Text chunk size")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size")


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    collection_name: str
    status: str = "success"


class RAGQueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    collection_name: str = Field(default="default", description="Collection to search")
    max_length: int = Field(default=200, description="Max response length")
    top_k: int = Field(default=3, description="Number of relevant chunks to retrieve")
    temperature: float = Field(default=0.7, description="Response creativity")


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    relevant_chunks: List[str]
    sources: List[str]
    confidence: float
    model_used: str


class RAGStatusResponse(BaseModel):
    collections: List[str]
    total_documents: int
    total_chunks: int
    status: str = "ready"


# backend/core/document_processor.py
import os
import hashlib
from typing import List, Tuple
from pathlib import Path
import logging

try:
    import fitz  # PyMuPDF

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyMuPDF not available, PDF processing disabled")


class DocumentProcessor:
    """Process documents and split into chunks"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif file_path.suffix.lower() == ".pdf":
            if not PDF_AVAILABLE:
                raise ValueError("PDF processing not available, install PyMuPDF")
            return self._extract_pdf_text(file_path)

        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyMuPDF"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to split at sentence or paragraph boundary
            if end < len(text):
                # Look for sentence endings within the last 100 chars
                for i in range(min(100, self.chunk_size // 4)):
                    if text[end - i] in ".!?。！？":
                        end = end - i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def process_document(self, file_path: str) -> Tuple[str, List[str]]:
        """Process document and return document_id and chunks"""
        text = self.extract_text_from_file(file_path)
        chunks = self.split_text(text)

        # Generate document ID from content hash
        content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        filename = Path(file_path).stem
        document_id = f"{filename}_{content_hash}"

        return document_id, chunks
