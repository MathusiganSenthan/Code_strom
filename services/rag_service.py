# =============================================================================
# services/rag_service.py - RAG Pipeline for Q&A Tasks
# =============================================================================

import os
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from services.simple_vector_store import SimpleVectorStore
from services.simple_embedding_service import SimpleEmbeddingService
import logging
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

class RAGResponseType(str, Enum):
    """Types of RAG responses for different Q&A scenarios."""
    DIRECT_ANSWER = "direct_answer"
    CLARIFICATION = "clarification"
    INSUFFICIENT_INFO = "insufficient_info"
    MULTI_PART = "multi_part"

class RAGAnswer(BaseModel):
    """Structured RAG response model aligned with frontend expectations."""
    answer: str = Field(description="Main answer content")
    response_type: RAGResponseType = Field(description="Type of response")
    confidence_score: float = Field(description="Confidence in answer (0-100)", ge=0, le=100)
    source_sections: List[str] = Field(description="Referenced document sections")
    related_topics: List[str] = Field(description="Related topics for follow-up")
    citations: List[Dict[str, Any]] = Field(description="Document chunks used as sources")
    processing_time: float = Field(description="Time taken to generate response")
    follow_up_questions: List[str] = Field(description="Suggested follow-up questions")

class RAGState(BaseModel):
    """State management for RAG workflow."""
    query: str
    document_id: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = []
    context: str = ""
    answer: str = ""
    confidence_score: float = 0.0
    source_sections: List[str] = []
    related_topics: List[str] = []
    citations: List[Dict[str, Any]] = []
    follow_up_questions: List[str] = []
    processing_time: float = 0.0
    response_type: RAGResponseType = RAGResponseType.DIRECT_ANSWER
    error: Optional[str] = None

class RAGService:
    """
    Advanced RAG (Retrieval-Augmented Generation) service for legal document Q&A.
    
    Features:
    - Semantic search with Redis vector store
    - Context-aware answer generation with Gemini
    - Confidence scoring and source attribution
    - Frontend-aligned response formatting
    - LangGraph workflow orchestration
    """
    
    def __init__(self, 
                 vector_store: SimpleVectorStore = None,
                 embedding_service: SimpleEmbeddingService = None):
        """Initialize RAG service with vector store and embedding service."""
        
        # Initialize services
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,  # Low temperature for consistent, factual responses
            max_tokens=2048,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize workflow
        self.workflow = self._create_rag_workflow()
        
        logger.info("✅ RAG Service initialized successfully")
    
    def _create_rag_workflow(self) -> CompiledStateGraph:
        """Create LangGraph workflow for RAG pipeline."""
        
        # Define the workflow
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_answer)
        workflow.add_node("enhance", self._enhance_response)
        
        # Define the flow
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "enhance")
        workflow.add_edge("enhance", END)
        
        # Compile workflow
        return workflow.compile()
    
    async def _retrieve_context(self, state: RAGState) -> RAGState:
        """Retrieve relevant context from vector store."""
        try:
            logger.info(f"🔍 Retrieving context for query: '{state.query[:50]}...'")
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_query_embedding(state.query)
            
            # Search for similar chunks
            similar_chunks = self.vector_store.search_similar_chunks(
                query_embedding=query_embedding,
                top_k=5,  # Retrieve top 5 most relevant chunks
                document_id=state.document_id
            )
            
            # Update state
            state.retrieved_chunks = similar_chunks
            
            # Create context from retrieved chunks
            context_parts = []
            for i, chunk in enumerate(similar_chunks, 1):
                context_parts.append(f"[Source {i}] {chunk['content']}")
            
            state.context = "\n\n".join(context_parts)
            
            logger.info(f"✅ Retrieved {len(similar_chunks)} relevant chunks")
            return state
            
        except Exception as e:
            logger.error(f"❌ Context retrieval failed: {e}")
            state.error = f"Failed to retrieve context: {str(e)}"
            return state
    
    async def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using Gemini with retrieved context."""
        try:
            logger.info("🤖 Generating answer with Gemini")
            
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", self._get_system_prompt()),
                ("human", self._get_human_prompt())
            ])
            
            # Create chain
            chain = prompt_template | self.llm | StrOutputParser()
            
            # Generate response
            response = await asyncio.to_thread(
                chain.invoke,
                {
                    "context": state.context,
                    "query": state.query,
                    "document_type": "legal contract"
                }
            )
            
            state.answer = response
            logger.info("✅ Answer generated successfully")
            return state
            
        except Exception as e:
            logger.error(f"❌ Answer generation failed: {e}")
            state.error = f"Failed to generate answer: {str(e)}"
            return state
    
    async def _enhance_response(self, state: RAGState) -> RAGState:
        """Enhance response with metadata and additional information."""
        try:
            logger.info("🔧 Enhancing response with metadata")
            
            # Calculate confidence score based on semantic similarity
            if state.retrieved_chunks:
                similarities = [chunk.get('similarity_score', 0) for chunk in state.retrieved_chunks]
                avg_similarity = sum(similarities) / len(similarities)
                state.confidence_score = min(avg_similarity * 100, 100)  # Convert to percentage
            else:
                state.confidence_score = 30  # Low confidence if no chunks found
            
            # Extract source sections
            state.source_sections = self._extract_source_sections(state.retrieved_chunks)
            
            # Generate related topics
            state.related_topics = self._generate_related_topics(state.query, state.answer)
            
            # Create citations
            state.citations = self._create_citations(state.retrieved_chunks)
            
            # Generate follow-up questions
            state.follow_up_questions = self._generate_follow_up_questions(state.query, state.answer)
            
            # Determine response type
            state.response_type = self._determine_response_type(state.answer, state.confidence_score)
            
            logger.info("✅ Response enhanced successfully")
            return state
            
        except Exception as e:
            logger.error(f"❌ Response enhancement failed: {e}")
            state.error = f"Failed to enhance response: {str(e)}"
            return state
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for legal document Q&A."""
        return """You are a specialized AI legal assistant focused on analyzing legal documents and contracts. 

Your responsibilities:
1. Provide accurate, factual answers based ONLY on the provided document context
2. Clearly cite specific sections or clauses when making statements
3. Acknowledge when information is not available in the provided context
4. Use plain language while maintaining legal accuracy
5. Highlight potential risks or important considerations
6. Structure responses clearly with bullet points or numbered lists when appropriate

Response guidelines:
- Be concise but comprehensive
- Always indicate confidence level in your answer
- Suggest follow-up questions when relevant
- Never provide legal advice, only document analysis
- If context is insufficient, clearly state this limitation"""
    
    def _get_human_prompt(self) -> str:
        """Get human prompt template for Q&A."""
        return """Based on the following legal document context, please answer the user's question:

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. References specific sections or clauses from the document
3. Explains any legal implications in plain language
4. Highlights important considerations or potential risks
5. Indicates if any information is missing or unclear

Answer:"""
    
    def _extract_source_sections(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract source sections from retrieved chunks."""
        sections = set()
        for chunk in chunks:
            # Try to extract section numbers from content
            content = chunk.get('content', '')
            # Simple regex patterns for common section formats
            import re
            section_patterns = [
                r'Section\s+(\d+\.?\d*)',
                r'Article\s+(\d+\.?\d*)',
                r'Clause\s+(\d+\.?\d*)',
                r'(\d+\.\d+)',
                r'§\s*(\d+\.?\d*)'
            ]
            
            for pattern in section_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    sections.add(f"Section {match}")
        
        return list(sections)[:5]  # Return top 5 sections
    
    def _generate_related_topics(self, query: str, answer: str) -> List[str]:
        """Generate related topics based on query and answer."""
        # Common legal topics that might be related
        legal_topics = [
            "Termination Conditions",
            "Liability Limitations", 
            "Intellectual Property Rights",
            "Dispute Resolution",
            "Confidentiality Clauses",
            "Payment Terms",
            "Force Majeure",
            "Indemnification",
            "Non-Compete Clauses",
            "Governing Law"
        ]
        
        # Simple keyword matching for related topics
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        related = []
        for topic in legal_topics:
            topic_keywords = topic.lower().split()
            if any(keyword in query_lower or keyword in answer_lower for keyword in topic_keywords):
                if topic not in related:
                    related.append(topic)
        
        return related[:4]  # Return top 4 related topics
    
    def _create_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create citation objects for frontend display."""
        citations = []
        for i, chunk in enumerate(chunks[:3], 1):  # Top 3 citations
            citations.append({
                "id": i,
                "content_preview": chunk.get('content', '')[:200] + "..." if len(chunk.get('content', '')) > 200 else chunk.get('content', ''),
                "similarity_score": round(chunk.get('similarity_score', 0) * 100, 1),
                "chunk_index": chunk.get('chunk_index', i),
                "source": f"Document Section {chunk.get('chunk_index', i)}"
            })
        
        return citations
    
    def _generate_follow_up_questions(self, query: str, answer: str) -> List[str]:
        """Generate relevant follow-up questions."""
        # Common follow-up patterns based on legal Q&A
        follow_ups = [
            "What are the exceptions to this rule?",
            "How does this affect my obligations?",
            "What happens if this condition is not met?",
            "Are there any time limits associated with this?",
            "What are the potential risks here?",
            "How can I protect myself in this situation?"
        ]
        
        # Select relevant follow-ups based on context
        return follow_ups[:3]  # Return top 3 follow-up questions
    
    def _determine_response_type(self, answer: str, confidence_score: float) -> RAGResponseType:
        """Determine the type of response based on content and confidence."""
        if confidence_score < 50:
            return RAGResponseType.INSUFFICIENT_INFO
        elif "clarify" in answer.lower() or "unclear" in answer.lower():
            return RAGResponseType.CLARIFICATION
        elif len(answer.split('\n')) > 3:  # Multi-paragraph response
            return RAGResponseType.MULTI_PART
        else:
            return RAGResponseType.DIRECT_ANSWER
    
    async def ask_question(self, 
                          query: str, 
                          document_id: Optional[str] = None) -> RAGAnswer:
        """
        Main method to ask a question about a document.
        
        Args:
            query: User's question
            document_id: Optional specific document ID to search
            
        Returns:
            RAGAnswer: Structured response with answer and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎯 Processing Q&A query: '{query[:50]}...'")
            
            # Check if services are available
            if not self.vector_store or not self.embedding_service:
                raise Exception("Vector store or embedding service not available")
            
            # Create initial state
            initial_state = RAGState(
                query=query,
                document_id=document_id
            )
            
            # Run workflow
            final_state = await asyncio.to_thread(
                self.workflow.invoke,
                initial_state
            )
            
            # Check for errors
            if final_state.error:
                raise Exception(final_state.error)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            final_state.processing_time = processing_time
            
            # Create structured response
            response = RAGAnswer(
                answer=final_state.answer,
                response_type=final_state.response_type,
                confidence_score=final_state.confidence_score,
                source_sections=final_state.source_sections,
                related_topics=final_state.related_topics,
                citations=final_state.citations,
                processing_time=processing_time,
                follow_up_questions=final_state.follow_up_questions
            )
            
            logger.info(f"✅ Q&A completed in {processing_time:.2f}s with {final_state.confidence_score:.1f}% confidence")
            return response
            
        except Exception as e:
            logger.error(f"❌ Q&A processing failed: {e}")
            
            # Return error response
            return RAGAnswer(
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again or rephrase your question.",
                response_type=RAGResponseType.INSUFFICIENT_INFO,
                confidence_score=0.0,
                source_sections=[],
                related_topics=[],
                citations=[],
                processing_time=time.time() - start_time,
                follow_up_questions=["Can you rephrase your question?", "Would you like to try a different question?"]
            )
    
    async def get_suggested_questions(self, document_id: Optional[str] = None) -> List[str]:
        """Generate suggested questions for a document."""
        # Common legal document questions
        suggestions = [
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
        
        return suggestions
    
    def health_check(self) -> Dict[str, Any]:
        """Check RAG service health."""
        try:
            vector_health = self.vector_store.health_check() if self.vector_store else {"status": "unavailable"}
            
            return {
                "status": "healthy",
                "vector_store": vector_health.get("status", "unknown"),
                "embedding_service": "healthy" if self.embedding_service else "unavailable",
                "llm_model": "gemini-2.0-flash-exp",
                "workflow_ready": self.workflow is not None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

def create_rag_service() -> RAGService:
    """Create and return a RAG service instance."""
    try:
        # Initialize vector store and embedding service
        vector_store = SimpleVectorStore(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD")  # Will be None if not set
        )
        
        embedding_service = SimpleEmbeddingService()
        
        return RAGService(
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to create RAG service: {e}")
        return RAGService(vector_store=None, embedding_service=None)
