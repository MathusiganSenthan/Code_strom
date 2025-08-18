"""
Simplified RAG Service without complex LangChain dependencies
This version provides Q&A functionality using direct API calls
"""

import os
import asyncio
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class RAGAnswer:
    """Structured answer from RAG system"""
    answer: str
    confidence_score: float
    citations: List[Dict]
    related_topics: List[str]
    follow_up_questions: List[str]
    processing_time: float
    response_type: str = "direct_answer"

class SimpleRAGService:
    """Simplified RAG service for Q&A functionality"""
    
    def __init__(self, vector_store=None, embedding_service=None):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Initialize Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel("gemini-2.0-flash-exp")
            logger.info("✅ Gemini LLM initialized")
        else:
            self.llm = None
            logger.warning("⚠️ No Google API key found - LLM responses will be limited")
    
    async def ask_question(self, query: str, document_id: str = None) -> RAGAnswer:
        """Process a question and return structured answer"""
        start_time = time.time()
        
        try:
            # Step 1: Semantic search for relevant context
            if self.vector_store and self.embedding_service:
                query_embedding = self.embedding_service.generate_query_embedding(query)
                similar_chunks = self.vector_store.search_similar_chunks(
                    query_embedding=query_embedding,
                    top_k=5,
                    document_id=document_id
                )
                logger.info(f"🔍 Found {len(similar_chunks)} relevant chunks")
            else:
                similar_chunks = []
                logger.warning("⚠️ Vector store not available - using fallback mode")
            
            # Step 2: Generate answer using LLM
            if self.llm and similar_chunks:
                context = self._build_context(similar_chunks)
                answer = await self._generate_llm_answer(query, context)
                confidence = self._calculate_confidence(similar_chunks)
            else:
                answer = self._generate_fallback_answer(query, similar_chunks)
                confidence = 60.0 if similar_chunks else 30.0
            
            # Step 3: Create citations
            citations = self._create_citations(similar_chunks)
            
            # Step 4: Generate related topics and follow-ups
            related_topics = self._generate_related_topics(query)
            follow_up_questions = self._generate_follow_up_questions(query)
            
            processing_time = time.time() - start_time
            
            return RAGAnswer(
                answer=answer,
                confidence_score=confidence,
                citations=citations,
                related_topics=related_topics,
                follow_up_questions=follow_up_questions,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ RAG processing failed: {e}")
            processing_time = time.time() - start_time
            
            return RAGAnswer(
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                confidence_score=10.0,
                citations=[],
                related_topics=[],
                follow_up_questions=["Could you please rephrase your question?"],
                processing_time=processing_time,
                response_type="error_response"
            )
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('content', '')
            if content:
                context_parts.append(f"[Context {i}]: {content}")
        return "\n\n".join(context_parts)
    
    async def _generate_llm_answer(self, query: str, context: str) -> str:
        """Generate answer using Gemini LLM"""
        try:
            prompt = f"""You are a legal document AI assistant. Based on the provided context from legal documents, answer the user's question accurately and concisely.

Context from documents:
{context}

User Question: {query}

Instructions:
- Provide a clear, accurate answer based only on the provided context
- If the context doesn't contain sufficient information, state this clearly
- Use professional legal language but remain accessible
- Cite specific sections when relevant
- Be concise but comprehensive

Answer:"""

            response = await asyncio.to_thread(
                self.llm.generate_content, 
                prompt
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return self._generate_fallback_answer(query, [])
    
    def _generate_fallback_answer(self, query: str, chunks: List[Dict]) -> str:
        """Generate fallback answer when LLM is not available"""
        if chunks:
            # Extract key information from chunks
            content_preview = chunks[0].get('content', '')[:300]
            return f"""Based on the document analysis for your question about "{query}":

{content_preview}...

For more detailed information, please refer to the specific document sections cited below."""
        else:
            return f"""I don't have specific information in the analyzed documents to answer your question about "{query}". 

This could be because:
- The document hasn't been processed yet
- The information isn't present in the current document
- Your question might need to be more specific

Please try rephrasing your question or ask about a different topic covered in the document."""
    
    def _calculate_confidence(self, chunks: List[Dict]) -> float:
        """Calculate confidence score based on retrieved chunks"""
        if not chunks:
            return 30.0
        
        # Calculate average similarity score
        total_similarity = sum(chunk.get('similarity_score', 0) for chunk in chunks)
        avg_similarity = total_similarity / len(chunks)
        
        # Convert to percentage and cap at 95%
        confidence = min(avg_similarity * 100, 95.0)
        
        # Boost confidence if we have multiple relevant chunks
        if len(chunks) >= 3 and confidence > 70:
            confidence = min(confidence + 10, 95.0)
        
        return round(confidence, 1)
    
    def _create_citations(self, chunks: List[Dict]) -> List[Dict]:
        """Create citation objects from chunks"""
        citations = []
        for i, chunk in enumerate(chunks[:5], 1):
            content = chunk.get('content', '')
            citations.append({
                "id": i,
                "content_preview": content[:200] + "..." if len(content) > 200 else content,
                "similarity_score": round(chunk.get('similarity_score', 0) * 100, 1),
                "chunk_index": chunk.get('chunk_index', i),
                "source": f"Document Section {chunk.get('chunk_index', i)}"
            })
        return citations
    
    def _generate_related_topics(self, query: str) -> List[str]:
        """Generate related topics based on the query"""
        # Legal document common topics
        legal_topics = [
            "Contract Obligations", "Termination Conditions", "Liability Limitations",
            "Payment Terms", "Dispute Resolution", "Intellectual Property",
            "Confidentiality", "Governing Law", "Force Majeure", "Indemnification"
        ]
        
        # Return a subset based on query keywords
        query_lower = query.lower()
        if "obligation" in query_lower:
            return ["Contract Obligations", "Performance Requirements", "Compliance Terms"]
        elif "termination" in query_lower or "end" in query_lower:
            return ["Termination Conditions", "Contract Expiry", "Notice Requirements"]
        elif "payment" in query_lower or "money" in query_lower:
            return ["Payment Terms", "Billing Procedures", "Late Fees"]
        elif "dispute" in query_lower or "conflict" in query_lower:
            return ["Dispute Resolution", "Arbitration", "Governing Law"]
        else:
            return legal_topics[:3]  # Default topics
    
    def _generate_follow_up_questions(self, query: str) -> List[str]:
        """Generate relevant follow-up questions"""
        follow_ups = [
            "Can you provide more specific details about this clause?",
            "What are the practical implications of this provision?",
            "Are there any exceptions or limitations to this rule?"
        ]
        
        query_lower = query.lower()
        if "obligation" in query_lower:
            follow_ups.extend([
                "What happens if these obligations are not met?",
                "How are these obligations enforced?"
            ])
        elif "termination" in query_lower:
            follow_ups.extend([
                "What notice period is required for termination?",
                "What are the consequences of early termination?"
            ])
        elif "payment" in query_lower:
            follow_ups.extend([
                "What are the consequences of late payment?",
                "Are there any payment milestones or schedules?"
            ])
        
        return follow_ups[:3]  # Return top 3
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions for legal documents"""
        return [
            "What are the key obligations for each party?",
            "What are the termination conditions?",
            "How are disputes resolved?",
            "What are the liability limitations?",
            "What intellectual property rights are involved?",
            "What are the payment terms and conditions?",
            "Are there any confidentiality requirements?",
            "What happens in case of breach of contract?",
            "What are the governing law and jurisdiction?",
            "Are there any automatic renewal clauses?"
        ]
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return {
            "status": "healthy",
            "vector_store_available": self.vector_store is not None,
            "embedding_service_available": self.embedding_service is not None,
            "llm_available": self.llm is not None,
            "capabilities": {
                "question_answering": True,
                "semantic_search": self.vector_store is not None,
                "llm_generation": self.llm is not None
            }
        }

def create_simple_rag_service(vector_store=None, embedding_service=None):
    """Create and return a simple RAG service instance"""
    try:
        # If no vector store provided, create a mock one for testing
        if vector_store is None:
            from .mock_vector_store import create_mock_vector_store
            vector_store = create_mock_vector_store()
            logger.info("✅ Using mock vector store for testing")
        
        # If no embedding service provided, create a simple one
        if embedding_service is None:
            from .simple_embedding_service import SimpleEmbeddingService
            embedding_service = SimpleEmbeddingService()
            logger.info("✅ Created simple embedding service")
        
        service = SimpleRAGService(vector_store, embedding_service)
        logger.info("✅ Simple RAG service created successfully")
        return service
    except Exception as e:
        logger.error(f"❌ Failed to create simple RAG service: {e}")
        raise
