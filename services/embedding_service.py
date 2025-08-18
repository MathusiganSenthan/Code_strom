# =============================================================================
# services/embedding_service.py - Text Chunking and Embedding Service
# =============================================================================

import os
import hashlib
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for text chunking and generating embeddings."""
    
    def __init__(self):
        """Initialize the embedding service."""
        try:
            # Initialize Google embeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                task_type="retrieval_document"
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            logger.info("✅ Embedding service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding service: {e}")
            raise
    
    def create_document_chunks(self, 
                             text: str, 
                             filename: str,
                             metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Split document text into chunks with metadata.
        
        Args:
            text: Document text content
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            List of chunks with metadata
        """
        try:
            logger.info(f"📝 Creating chunks for document: {filename}")
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk_text in enumerate(chunks):
                chunk_obj = {
                    "content": chunk_text.strip(),
                    "chunk_index": i,
                    "filename": filename,
                    "chunk_size": len(chunk_text),
                    "metadata": metadata or {},
                    "chunk_hash": hashlib.md5(chunk_text.encode()).hexdigest()
                }
                chunk_objects.append(chunk_obj)
            
            logger.info(f"✅ Created {len(chunk_objects)} chunks from {len(text)} characters")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"❌ Failed to create chunks: {e}")
            return []
    
    async def generate_embeddings_async(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts asynchronously.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"🔄 Generating embeddings for {len(texts)} texts...")
            start_time = time.time()
            
            # Use asyncio to run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Split texts into batches for parallel processing
                batch_size = 10
                batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
                
                # Process batches in parallel
                futures = []
                for batch in batches:
                    future = loop.run_in_executor(
                        executor, 
                        self._generate_batch_embeddings, 
                        batch
                    )
                    futures.append(future)
                
                # Wait for all batches to complete
                batch_results = await asyncio.gather(*futures)
                
                # Flatten results
                all_embeddings = []
                for batch_embeddings in batch_results:
                    all_embeddings.extend(batch_embeddings)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Generated {len(all_embeddings)} embeddings in {elapsed_time:.2f}s")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embeddings: {e}")
            return []
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"❌ Batch embedding failed: {e}")
            return [[0.0] * 768] * len(texts)  # Return zero vectors as fallback
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(query)
            logger.info(f"✅ Generated query embedding for: '{query[:50]}...'")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Failed to generate query embedding: {e}")
            return [0.0] * 768  # Return zero vector as fallback
    
    async def process_document_for_vector_store(self, 
                                              text: str,
                                              filename: str,
                                              document_id: str,
                                              metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Complete pipeline: chunk text and generate embeddings.
        
        Args:
            text: Document text
            filename: Original filename
            document_id: Unique document identifier
            metadata: Additional metadata
            
        Returns:
            List of chunks with embeddings ready for vector store
        """
        try:
            logger.info(f"🚀 Processing document for vector store: {filename}")
            start_time = time.time()
            
            # Step 1: Create chunks
            chunks = self.create_document_chunks(text, filename, metadata)
            
            if not chunks:
                logger.error("❌ No chunks created from document")
                return []
            
            # Step 2: Extract text content for embedding
            chunk_texts = [chunk["content"] for chunk in chunks]
            
            # Step 3: Generate embeddings
            embeddings = await self.generate_embeddings_async(chunk_texts)
            
            if len(embeddings) != len(chunks):
                logger.error(f"❌ Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")
                return []
            
            # Step 4: Combine chunks with embeddings
            processed_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                processed_chunk = {
                    **chunk,
                    "document_id": document_id,
                    "embedding": embedding,
                    "embedding_model": "text-embedding-004",
                    "embedding_dimension": len(embedding)
                }
                processed_chunks.append(processed_chunk)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Processed {len(processed_chunks)} chunks in {elapsed_time:.2f}s")
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            return []
    
    def calculate_similarity(self, 
                           embedding1: List[float], 
                           embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {e}")
            return 0.0
    
    def get_optimal_chunk_size(self, text_length: int) -> int:
        """Determine optimal chunk size based on document length."""
        if text_length < 5000:
            return 500
        elif text_length < 20000:
            return 1000
        elif text_length < 50000:
            return 1500
        else:
            return 2000
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        return {
            "embedding_model": "text-embedding-004",
            "embedding_dimension": 768,
            "chunk_size": self.text_splitter._chunk_size,
            "chunk_overlap": self.text_splitter._chunk_overlap,
            "task_type": "retrieval_document",
            "max_batch_size": 10,
            "status": "ready"
        }

def create_embedding_service() -> EmbeddingService:
    """Create and return an embedding service instance."""
    return EmbeddingService()
