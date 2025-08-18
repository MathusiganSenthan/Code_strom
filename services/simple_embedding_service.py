"""
Simplified embedding service without LangChain dependencies
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class SimpleTextSplitter:
    """Simple text splitter without LangChain dependency"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find end position
            end = start + self.chunk_size
            
            # If we're not at the end of the text, try to break at a good point
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start + self.chunk_size - 100, start)
                search_text = text[search_start:end]
                
                # Find the last sentence ending
                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_idx = search_text.rfind(punct)
                    if last_idx != -1:
                        end = search_start + last_idx + len(punct)
                        break
            
            # Extract chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks"""
        all_chunks = []
        
        for doc in documents:
            text = doc.get('content', '')
            chunks = self.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    **doc,  # Copy original metadata
                    'content': chunk,
                    'chunk_id': f"{doc.get('id', 'unknown')}_{i}",
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk)  # Add chunk size
                }
                all_chunks.append(chunk_doc)
        
        return all_chunks

class SimpleEmbeddingService:
    """Simplified embedding service"""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            task_type="retrieval_document"
        )
        self.text_splitter = SimpleTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents by splitting and generating embeddings"""
        
        try:
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                return []
            
            # Extract text for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            print(f"🔄 Generating embeddings for {len(texts)} chunks...")
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk['embedding'] = embedding
                chunk['embedding_dim'] = len(embedding)
            
            print(f"✅ Successfully processed {len(chunks)} chunks with embeddings")
            return chunks
            
        except Exception as e:
            print(f"❌ Error in process_documents: {e}")
            import traceback
            traceback.print_exc()
            # Return chunks without embeddings if possible
            try:
                chunks = self.text_splitter.split_documents(documents)
                for chunk in chunks:
                    chunk['embedding'] = None
                    chunk['embedding_dim'] = 0
                return chunks
            except:
                return []
    
    def generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for a search query"""
        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return None
    
    def calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def search_similar_chunks(self, query_embedding: List[float], 
                            chunks: List[Dict[str, Any]], 
                            top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity"""
        
        results = []
        
        for chunk in chunks:
            if chunk.get('embedding') is None:
                continue
            
            similarity = self.calculate_similarity(query_embedding, chunk['embedding'])
            
            result = {
                **chunk,
                'similarity_score': similarity
            }
            results.append(result)
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return results[:top_k]
