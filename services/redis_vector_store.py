# =============================================================================
# services/redis_vector_store.py - Redis Vector Database Service
# =============================================================================

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import redis
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

class RedisVectorStore:
    """Redis-based vector store for document embeddings and metadata."""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6379, 
                 db: int = 0,
                 password: Optional[str] = None):
        """Initialize Redis connection and vector store."""
        
        try:
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                db=db, 
                password=password,
                decode_responses=False,  # Keep binary for vector data
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {host}:{port}")
            
            # Initialize index
            self.index_name = "document_vectors"
            self.doc_prefix = "doc:"
            self._create_index()
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def _create_index(self):
        """Create vector search index if it doesn't exist."""
        try:
            # Check if index exists
            try:
                self.redis_client.ft(self.index_name).info()
                logger.info(f"✅ Vector index '{self.index_name}' already exists")
                return
            except:
                pass  # Index doesn't exist, create it
            
            # Define schema
            schema = [
                TextField("content"),
                TextField("document_id"),
                TextField("filename"),
                TextField("chunk_id"),
                NumericField("chunk_index"),
                NumericField("timestamp"),
                VectorField("embedding", 
                           "FLAT", 
                           {
                               "TYPE": "FLOAT32",
                               "DIM": 768,  # Google text-embedding-004 dimension
                               "DISTANCE_METRIC": "COSINE"
                           })
            ]
            
            # Create index
            definition = IndexDefinition(
                prefix=[self.doc_prefix],
                index_type=IndexType.HASH
            )
            
            self.redis_client.ft(self.index_name).create_index(
                schema, 
                definition=definition
            )
            
            logger.info(f"✅ Created vector index '{self.index_name}'")
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector index: {e}")
            raise
    
    def store_document_chunks(self, 
                            document_id: str,
                            filename: str,
                            chunks: List[Dict[str, Any]]) -> bool:
        """
        Store document chunks with embeddings in Redis.
        
        Args:
            document_id: Unique document identifier
            filename: Original filename
            chunks: List of chunks with content and embeddings
        
        Returns:
            bool: Success status
        """
        try:
            stored_count = 0
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                key = f"{self.doc_prefix}{chunk_id}"
                
                # Prepare data
                data = {
                    "content": chunk["content"],
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "timestamp": int(datetime.now().timestamp()),
                    "embedding": np.array(chunk["embedding"]).astype(np.float32).tobytes()
                }
                
                # Store in Redis
                self.redis_client.hset(key, mapping=data)
                stored_count += 1
            
            logger.info(f"✅ Stored {stored_count} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store document chunks: {e}")
            return False
    
    def search_similar_chunks(self, 
                            query_embedding: List[float], 
                            top_k: int = 5,
                            document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            document_id: Optional filter by document ID
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Convert embedding to bytes
            query_vector = np.array(query_embedding).astype(np.float32).tobytes()
            
            # Build query
            base_query = f"*=>[KNN {top_k} @embedding $query_vector AS score]"
            
            if document_id:
                base_query = f"(@document_id:{document_id}) => {base_query}"
            
            query = Query(base_query).return_fields(
                "content", "document_id", "filename", "chunk_id", 
                "chunk_index", "timestamp", "score"
            ).sort_by("score").dialect(2)
            
            # Execute search
            results = self.redis_client.ft(self.index_name).search(
                query, 
                {"query_vector": query_vector}
            )
            
            # Process results
            chunks = []
            for doc in results.docs:
                chunks.append({
                    "content": doc.content,
                    "document_id": doc.document_id,
                    "filename": doc.filename,
                    "chunk_id": doc.chunk_id,
                    "chunk_index": int(doc.chunk_index),
                    "timestamp": int(doc.timestamp),
                    "similarity_score": float(doc.score)
                })
            
            logger.info(f"✅ Found {len(chunks)} similar chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            query = Query(f"@document_id:{document_id}").return_fields(
                "content", "chunk_id", "chunk_index", "timestamp"
            ).sort_by("chunk_index")
            
            results = self.redis_client.ft(self.index_name).search(query)
            
            chunks = []
            for doc in results.docs:
                chunks.append({
                    "content": doc.content,
                    "chunk_id": doc.chunk_id,
                    "chunk_index": int(doc.chunk_index),
                    "timestamp": int(doc.timestamp)
                })
            
            return sorted(chunks, key=lambda x: x["chunk_index"])
            
        except Exception as e:
            logger.error(f"❌ Failed to get document chunks: {e}")
            return []
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # Find all chunk keys for the document
            query = Query(f"@document_id:{document_id}").return_fields("chunk_id")
            results = self.redis_client.ft(self.index_name).search(query)
            
            # Delete each chunk
            deleted_count = 0
            for doc in results.docs:
                key = f"{self.doc_prefix}{doc.chunk_id}"
                self.redis_client.delete(key)
                deleted_count += 1
            
            logger.info(f"✅ Deleted {deleted_count} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            info = self.redis_client.ft(self.index_name).info()
            
            # Count documents and chunks
            all_docs = self.redis_client.ft(self.index_name).search(Query("*"))
            
            # Get unique documents
            unique_docs = set()
            for doc in all_docs.docs:
                unique_docs.add(doc.document_id)
            
            return {
                "total_chunks": len(all_docs.docs),
                "unique_documents": len(unique_docs),
                "index_size": info.get("index_size", 0),
                "vector_dimension": 768,
                "distance_metric": "COSINE"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {}
    
    def health_check(self) -> Dict[str, Any]:
        """Check Redis connection and index health."""
        try:
            # Test basic connection
            self.redis_client.ping()
            
            # Test index
            info = self.redis_client.ft(self.index_name).info()
            
            return {
                "redis_connected": True,
                "index_exists": True,
                "index_name": self.index_name,
                "status": "healthy"
            }
            
        except Exception as e:
            return {
                "redis_connected": False,
                "index_exists": False,
                "error": str(e),
                "status": "unhealthy"
            }

def create_vector_store() -> RedisVectorStore:
    """Create and return a Redis vector store instance."""
    return RedisVectorStore(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        password=os.getenv("REDIS_PASSWORD")
    )
