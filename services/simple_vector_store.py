# =============================================================================
# services/simple_vector_store.py - Simple Redis Vector Store for Testing
# =============================================================================

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import redis
import logging
from datetime import datetime
import hashlib
import pickle

logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """Simple Redis-based vector store for testing embedding functionality."""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6379, 
                 db: int = 0,
                 password: Optional[str] = None):
        """Initialize Redis connection."""
        
        try:
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                db=db, 
                password=password,
                decode_responses=False,  # Keep binary for vectors
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {host}:{port}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def store_document_chunks(self, 
                            document_id: str,
                            filename: str,
                            chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks with embeddings in Redis using simple keys."""
        try:
            stored_count = 0
            
            # Store document metadata
            doc_meta_key = f"doc_meta:{document_id}"
            doc_metadata = {
                "filename": filename,
                "chunk_count": len(chunks),
                "timestamp": int(datetime.now().timestamp())
            }
            self.redis_client.hset(doc_meta_key, mapping=doc_metadata)
            
            # Store each chunk
            for i, chunk in enumerate(chunks):
                chunk_key = f"chunk:{document_id}:{i}"
                
                # Prepare chunk data
                chunk_data = {
                    "content": chunk["content"],
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": i,
                    "metadata": json.dumps(chunk.get("metadata", {}))
                }
                
                # Store chunk metadata
                self.redis_client.hset(chunk_key, mapping=chunk_data)
                
                # Store embedding separately (as binary)
                embedding_key = f"embedding:{document_id}:{i}"
                embedding_bytes = pickle.dumps(chunk["embedding"])
                self.redis_client.set(embedding_key, embedding_bytes)
                
                stored_count += 1
            
            # Add to document list
            self.redis_client.sadd("documents", document_id)
            
            logger.info(f"✅ Stored {stored_count} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store document chunks: {e}")
            return False
    
    def search_similar_chunks(self, 
                            query_embedding: List[float], 
                            top_k: int = 5,
                            document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Simple similarity search using cosine similarity."""
        try:
            results = []
            query_vector = np.array(query_embedding)
            
            # Get all document IDs to search
            if document_id:
                doc_ids = [document_id]
            else:
                doc_ids = [doc_id.decode() for doc_id in self.redis_client.smembers("documents")]
            
            # Search through all chunks
            for doc_id in doc_ids:
                # Get document metadata
                doc_meta = self.redis_client.hgetall(f"doc_meta:{doc_id}")
                if not doc_meta:
                    continue
                
                chunk_count = int(doc_meta.get(b"chunk_count", 0))
                
                for i in range(chunk_count):
                    try:
                        # Get chunk data
                        chunk_key = f"chunk:{doc_id}:{i}"
                        chunk_data = self.redis_client.hgetall(chunk_key)
                        
                        # Get embedding
                        embedding_key = f"embedding:{doc_id}:{i}"
                        embedding_bytes = self.redis_client.get(embedding_key)
                        
                        if not chunk_data or not embedding_bytes:
                            continue
                        
                        # Deserialize embedding
                        chunk_embedding = pickle.loads(embedding_bytes)
                        chunk_vector = np.array(chunk_embedding)
                        
                        # Calculate cosine similarity
                        similarity = np.dot(query_vector, chunk_vector) / (
                            np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector)
                        )
                        
                        # Add to results
                        results.append({
                            "content": chunk_data[b"content"].decode(),
                            "document_id": chunk_data[b"document_id"].decode(),
                            "filename": chunk_data[b"filename"].decode(),
                            "chunk_index": int(chunk_data[b"chunk_index"]),
                            "similarity_score": float(similarity)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing chunk {doc_id}:{i}: {e}")
                        continue
            
            # Sort by similarity and return top_k
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            results = results[:top_k]
            
            logger.info(f"✅ Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            chunks = []
            
            # Get document metadata
            doc_meta = self.redis_client.hgetall(f"doc_meta:{document_id}")
            if not doc_meta:
                return []
            
            chunk_count = int(doc_meta.get(b"chunk_count", 0))
            
            for i in range(chunk_count):
                chunk_key = f"chunk:{document_id}:{i}"
                chunk_data = self.redis_client.hgetall(chunk_key)
                
                if chunk_data:
                    chunks.append({
                        "content": chunk_data[b"content"].decode(),
                        "chunk_index": int(chunk_data[b"chunk_index"]),
                        "metadata": json.loads(chunk_data[b"metadata"].decode())
                    })
            
            return sorted(chunks, key=lambda x: x["chunk_index"])
            
        except Exception as e:
            logger.error(f"❌ Failed to get document chunks: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # Get chunk count
            doc_meta = self.redis_client.hgetall(f"doc_meta:{document_id}")
            if not doc_meta:
                return True
            
            chunk_count = int(doc_meta.get(b"chunk_count", 0))
            
            # Delete all chunks and embeddings
            deleted_count = 0
            for i in range(chunk_count):
                chunk_key = f"chunk:{document_id}:{i}"
                embedding_key = f"embedding:{document_id}:{i}"
                
                self.redis_client.delete(chunk_key, embedding_key)
                deleted_count += 1
            
            # Delete document metadata
            self.redis_client.delete(f"doc_meta:{document_id}")
            
            # Remove from document set
            self.redis_client.srem("documents", document_id)
            
            logger.info(f"✅ Deleted {deleted_count} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        try:
            document_count = self.redis_client.scard("documents")
            
            # Count total chunks
            total_chunks = 0
            doc_ids = [doc_id.decode() for doc_id in self.redis_client.smembers("documents")]
            
            for doc_id in doc_ids:
                doc_meta = self.redis_client.hgetall(f"doc_meta:{doc_id}")
                if doc_meta:
                    total_chunks += int(doc_meta.get(b"chunk_count", 0))
            
            return {
                "total_chunks": total_chunks,
                "unique_documents": document_count,
                "vector_dimension": 768,
                "distance_metric": "COSINE",
                "storage_type": "Redis Simple"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health."""
        try:
            # Test basic connection
            self.redis_client.ping()
            
            return {
                "redis_connected": True,
                "storage_ready": True,
                "status": "healthy",
                "storage_type": "Redis Simple"
            }
            
        except Exception as e:
            return {
                "redis_connected": False,
                "storage_ready": False,
                "error": str(e),
                "status": "unhealthy"
            }

def create_vector_store() -> SimpleVectorStore:
    """Create and return a simple vector store instance."""
    return SimpleVectorStore(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD")
    )
