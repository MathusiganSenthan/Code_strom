"""
Mock Vector Store for testing RAG functionality without Redis dependency
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MockVectorStore:
    """Mock vector store that simulates Redis functionality without requiring Redis."""
    
    def __init__(self, **kwargs):
        """Initialize mock vector store."""
        self.documents = {}
        self.chunks = {}
        self.embeddings = {}
        logger.info("✅ Mock Vector Store initialized (no Redis required)")
    
    def store_document_chunks(self, 
                            document_id: str,
                            filename: str,
                            chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks in memory."""
        try:
            self.documents[document_id] = {
                'filename': filename,
                'chunk_count': len(chunks),
                'stored_at': datetime.now().isoformat()
            }
            
            for i, chunk in enumerate(chunks):
                chunk_key = f"{document_id}:chunk:{i}"
                self.chunks[chunk_key] = chunk
                
                # Store embedding if present
                if 'embedding' in chunk:
                    self.embeddings[chunk_key] = chunk['embedding']
            
            logger.info(f"✅ Stored {len(chunks)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store chunks: {e}")
            return False
    
    def search_similar_chunks(self, 
                            query_embedding: List[float], 
                            top_k: int = 5,
                            document_id: Optional[str] = None,
                            similarity_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity."""
        
        try:
            if not self.embeddings:
                logger.warning("⚠️ No embeddings available for search")
                return self._get_sample_chunks()
            
            query_vector = np.array(query_embedding)
            similarities = []
            
            for chunk_key, embedding in self.embeddings.items():
                if document_id and not chunk_key.startswith(f"{document_id}:"):
                    continue
                
                chunk_vector = np.array(embedding)
                
                # Calculate cosine similarity
                dot_product = np.dot(query_vector, chunk_vector)
                norm_query = np.linalg.norm(query_vector)
                norm_chunk = np.linalg.norm(chunk_vector)
                
                if norm_query > 0 and norm_chunk > 0:
                    similarity = dot_product / (norm_query * norm_chunk)
                else:
                    similarity = 0.0
                
                similarities.append({
                    'chunk_key': chunk_key,
                    'similarity_score': similarity,
                    'chunk_data': self.chunks.get(chunk_key, {})
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            results = []
            for item in similarities[:top_k]:
                if item['similarity_score'] >= similarity_threshold:
                    chunk_data = item['chunk_data']
                    results.append({
                        'content': chunk_data.get('content', ''),
                        'similarity_score': item['similarity_score'],
                        'chunk_index': chunk_data.get('chunk_index', 0),
                        'document_id': chunk_data.get('document_id', 'unknown')
                    })
            
            logger.info(f"🔍 Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return self._get_sample_chunks()
    
    def _get_sample_chunks(self) -> List[Dict[str, Any]]:
        """Return sample chunks for testing when no real data is available."""
        return [
            {
                'content': "This contract establishes obligations for both parties including payment terms, delivery schedules, and performance standards.",
                'similarity_score': 0.85,
                'chunk_index': 1,
                'document_id': 'sample_doc'
            },
            {
                'content': "The termination clause specifies that either party may terminate with 30 days written notice.",
                'similarity_score': 0.72,
                'chunk_index': 2,
                'document_id': 'sample_doc'
            },
            {
                'content': "Dispute resolution shall be handled through arbitration in accordance with local laws.",
                'similarity_score': 0.68,
                'chunk_index': 3,
                'document_id': 'sample_doc'
            }
        ]
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents."""
        total_chunks = len(self.chunks)
        total_embeddings = len(self.embeddings)
        
        return {
            'total_documents': len(self.documents),
            'total_chunks': total_chunks,
            'total_embeddings': total_embeddings,
            'storage_type': 'memory',
            'status': 'healthy'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store."""
        return {
            'status': 'healthy',
            'type': 'mock_vector_store',
            'documents_count': len(self.documents),
            'chunks_count': len(self.chunks),
            'embeddings_count': len(self.embeddings)
        }
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            if document_id in self.documents:
                del self.documents[document_id]
                
                # Delete associated chunks and embeddings
                keys_to_delete = [key for key in self.chunks.keys() if key.startswith(f"{document_id}:")]
                for key in keys_to_delete:
                    del self.chunks[key]
                    if key in self.embeddings:
                        del self.embeddings[key]
                
                logger.info(f"✅ Deleted document {document_id}")
                return True
            else:
                logger.warning(f"⚠️ Document {document_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return False

def create_mock_vector_store(**kwargs):
    """Create a mock vector store instance."""
    return MockVectorStore(**kwargs)
