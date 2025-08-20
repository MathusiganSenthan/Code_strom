# =============================================================================
# services/memory_vector_store.py - In-Memory Vector Store (No Redis Required)
# =============================================================================

import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import hashlib
import pickle

logger = logging.getLogger(__name__)

class MemoryVectorStore:
    """In-memory vector store that doesn't require Redis - perfect for development."""
    
    def __init__(self):
        """Initialize in-memory storage."""
        self.documents = {}  # document_id -> document_info
        self.chunks = {}     # chunk_key -> chunk_data
        self.embeddings = {} # chunk_key -> embedding_vector
        logger.info("✅ In-memory vector store initialized")
    
    def store_document_chunks(self, 
                            document_id: str,
                            filename: str,
                            chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks with embeddings in memory."""
        
        try:
            # Store document metadata
            self.documents[document_id] = {
                "filename": filename,
                "chunk_count": len(chunks),
                "created_at": datetime.now().isoformat(),
                "document_id": document_id
            }
            
            # Store each chunk with its embedding
            stored_count = 0
            for i, chunk in enumerate(chunks):
                chunk_key = f"{document_id}:chunk:{i}"
                
                # Store chunk data
                self.chunks[chunk_key] = {
                    "content": chunk["content"],
                    "chunk_index": i,
                    "document_id": document_id,
                    "filename": filename,
                    "metadata": chunk.get("metadata", {})
                }
                
                # Store embedding if provided
                if "embedding" in chunk:
                    self.embeddings[chunk_key] = np.array(chunk["embedding"])
                    stored_count += 1
            
            logger.info(f"✅ Stored {stored_count} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store chunks: {e}")
            return False
    
    def search_similar_chunks(self, 
                            query_embedding: List[float],
                            top_k: int = 5,
                            document_id: Optional[str] = None,
                            min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity."""
        
        try:
            query_vec = np.array(query_embedding)
            results = []
            
            # Search through all stored embeddings
            for chunk_key, embedding in self.embeddings.items():
                chunk_data = self.chunks.get(chunk_key, {})
                
                # Filter by document_id if specified
                if document_id and chunk_data.get("document_id") != document_id:
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vec, embedding)
                
                if similarity >= min_similarity:
                    results.append({
                        "content": chunk_data.get("content", ""),
                        "similarity_score": float(similarity),
                        "chunk_index": chunk_data.get("chunk_index", 0),
                        "document_id": chunk_data.get("document_id", ""),
                        "filename": chunk_data.get("filename", ""),
                        "metadata": chunk_data.get("metadata", {})
                    })
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            logger.info(f"✅ Found {len(results)} similar chunks, returning top {top_k}")
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document information."""
        return self.documents.get(document_id)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all stored documents."""
        return list(self.documents.values())
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            # Remove document metadata
            if document_id in self.documents:
                del self.documents[document_id]
            
            # Remove all chunks for this document
            chunks_to_remove = [
                key for key, data in self.chunks.items() 
                if data.get("document_id") == document_id
            ]
            
            for chunk_key in chunks_to_remove:
                if chunk_key in self.chunks:
                    del self.chunks[chunk_key]
                if chunk_key in self.embeddings:
                    del self.embeddings[chunk_key]
            
            logger.info(f"✅ Deleted document {document_id} and {len(chunks_to_remove)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "total_embeddings": len(self.embeddings),
            "storage_type": "in_memory"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the vector store is healthy."""
        try:
            stats = self.get_stats()
            return {
                "status": "healthy",
                "type": "memory_vector_store", 
                "stats": stats
            }
        except Exception as e:
            return {
                "status": "error",
                "type": "memory_vector_store",
                "error": str(e)
            }
