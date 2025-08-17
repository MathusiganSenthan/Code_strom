"""
Ultra-simplified RAG Service for immediate testing
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SimpleRAGAnswer:
    """Simple answer structure"""
    answer: str
    confidence_score: float
    citations: List[Dict]
    related_topics: List[str]
    follow_up_questions: List[str]
    processing_time: float
    response_type: str = "mock_answer"

class UltraSimpleRAGService:
    """Ultra-simple RAG service that works without any external dependencies"""
    
    def __init__(self, vector_store=None, embedding_service=None):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        logger.info("✅ Ultra-simple RAG service initialized")
    
    async def ask_question(self, query: str, document_id: str = None) -> SimpleRAGAnswer:
        """Process a question and return a mock answer"""
        start_time = time.time()
        
        try:
            # Generate a contextual answer based on the question
            answer = self._generate_mock_answer(query)
            citations = self._generate_mock_citations()
            related_topics = self._generate_related_topics(query)
            follow_up_questions = self._generate_follow_up_questions(query)
            
            processing_time = time.time() - start_time
            
            return SimpleRAGAnswer(
                answer=answer,
                confidence_score=75.0,
                citations=citations,
                related_topics=related_topics,
                follow_up_questions=follow_up_questions,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error in ultra-simple RAG: {e}")
            processing_time = time.time() - start_time
            
            return SimpleRAGAnswer(
                answer="I apologize, but I encountered an error processing your question.",
                confidence_score=10.0,
                citations=[],
                related_topics=[],
                follow_up_questions=["Could you please rephrase your question?"],
                processing_time=processing_time,
                response_type="error_response"
            )
    
    def _generate_mock_answer(self, query: str) -> str:
        """Generate a mock answer based on the query"""
        query_lower = query.lower()
        
        if "obligation" in query_lower:
            return """Based on the legal document analysis, the key obligations for each party typically include:

**For Party A (Service Provider):**
- Deliver services according to specified standards and timelines
- Maintain confidentiality of client information
- Provide regular progress reports and updates
- Ensure compliance with applicable laws and regulations

**For Party B (Client):**
- Provide timely payment according to agreed terms
- Furnish necessary information and access for service delivery
- Cooperate in good faith during the service period
- Give reasonable notice for any changes or modifications

These obligations are fundamental to the contractual relationship and ensure mutual accountability between the parties."""

        elif "termination" in query_lower:
            return """The termination conditions in this contract include:

**Standard Termination:**
- Either party may terminate with 30 days written notice
- Termination for convenience requires completion of ongoing deliverables

**Immediate Termination:**
- Material breach of contract terms
- Insolvency or bankruptcy of either party
- Failure to cure breaches within specified cure period

**Post-Termination Obligations:**
- Return of confidential information
- Payment for services rendered up to termination date
- Transition assistance for ongoing projects"""

        elif "dispute" in query_lower:
            return """Dispute resolution mechanisms include:

**Primary Resolution:**
- Good faith negotiations between parties
- 30-day informal resolution period

**Alternative Dispute Resolution:**
- Mediation through qualified neutral mediator
- Binding arbitration if mediation fails

**Governing Law:**
- Disputes governed by local jurisdiction laws
- Exclusive venue in courts of contract jurisdiction
- Attorney fees may be awarded to prevailing party"""

        elif "payment" in query_lower:
            return """Payment terms and conditions:

**Payment Schedule:**
- Monthly invoicing on the 1st of each month
- Net 30 payment terms from invoice date
- Late fees of 1.5% per month on overdue amounts

**Payment Methods:**
- Wire transfer or ACH preferred
- Checks accepted with 5-day processing time

**Invoicing Requirements:**
- Detailed breakdown of services provided
- Supporting documentation as required
- Proper tax identification numbers"""

        else:
            return f"""Based on your question about "{query}", here are the key points from the legal document analysis:

This appears to be a comprehensive legal agreement with standard commercial terms. The document outlines the rights, responsibilities, and obligations of all parties involved.

Key areas typically covered include:
- Scope of work and deliverables
- Payment terms and conditions
- Intellectual property rights
- Confidentiality requirements
- Limitation of liability clauses
- Termination and dispute resolution procedures

For more specific information about this topic, please ask a more targeted question about particular clauses or provisions."""
    
    def _generate_mock_citations(self) -> List[Dict]:
        """Generate mock citations"""
        return [
            {
                "id": 1,
                "content_preview": "This contract establishes the fundamental obligations and responsibilities of each party...",
                "similarity_score": 85.2,
                "chunk_index": 1,
                "source": "Document Section 1"
            },
            {
                "id": 2,
                "content_preview": "The terms and conditions set forth herein shall govern the relationship between...",
                "similarity_score": 78.9,
                "chunk_index": 2,
                "source": "Document Section 2"
            },
            {
                "id": 3,
                "content_preview": "All provisions of this agreement are binding and enforceable according to...",
                "similarity_score": 72.4,
                "chunk_index": 3,
                "source": "Document Section 3"
            }
        ]
    
    def _generate_related_topics(self, query: str) -> List[str]:
        """Generate related topics based on query"""
        query_lower = query.lower()
        
        if "obligation" in query_lower:
            return ["Performance Standards", "Compliance Requirements", "Deliverables"]
        elif "termination" in query_lower:
            return ["Notice Requirements", "Breach Conditions", "Post-Termination Rights"]
        elif "payment" in query_lower:
            return ["Invoicing Procedures", "Late Fees", "Payment Methods"]
        elif "dispute" in query_lower:
            return ["Mediation Process", "Arbitration Rules", "Governing Law"]
        else:
            return ["Contract Terms", "Legal Obligations", "Compliance"]
    
    def _generate_follow_up_questions(self, query: str) -> List[str]:
        """Generate relevant follow-up questions"""
        return [
            "Can you provide more details about the specific requirements?",
            "What are the consequences if these terms are not met?",
            "Are there any exceptions or special circumstances to consider?"
        ]
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions for legal documents"""
        return [
            "What are the key obligations for each party?",
            "What are the termination conditions?",
            "How are disputes resolved?",
            "What are the liability limitations?",
            "What are the payment terms and conditions?",
            "Are there any confidentiality requirements?",
            "What happens in case of breach of contract?",
            "What are the governing law and jurisdiction?",
            "Are there any automatic renewal clauses?",
            "What intellectual property rights are involved?"
        ]
    
    def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        return {
            "status": "healthy",
            "type": "ultra_simple_rag",
            "vector_store_available": self.vector_store is not None,
            "embedding_service_available": self.embedding_service is not None,
            "llm_available": False,  # No external LLM
            "capabilities": {
                "question_answering": True,
                "semantic_search": False,  # Mock responses
                "llm_generation": False   # Mock responses
            }
        }

def create_ultra_simple_rag_service(vector_store=None, embedding_service=None):
    """Create an ultra-simple RAG service for immediate testing"""
    try:
        service = UltraSimpleRAGService(vector_store, embedding_service)
        logger.info("✅ Ultra-simple RAG service created successfully")
        return service
    except Exception as e:
        logger.error(f"❌ Failed to create ultra-simple RAG service: {e}")
        raise
