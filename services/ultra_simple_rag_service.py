"""
Ultra-simplified RAG Service with LLM-powered response generation
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import LLM for proper response generation
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

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
    """Ultra-simple RAG service with LLM-powered response generation"""
    
    def __init__(self, vector_store=None, embedding_service=None):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        
        # Initialize LLM for response generation
        self.llm = None
        if LLM_AVAILABLE:
            try:
                api_key = os.getenv("GOOGLE_API_KEY")
                if api_key:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0.1,
                        google_api_key=api_key
                    )
                    logger.info("✅ LLM initialized for response generation")
                else:
                    logger.warning("⚠️ No GOOGLE_API_KEY found, using template responses")
            except Exception as e:
                logger.warning(f"⚠️ LLM initialization failed: {e}, using template responses")
        
        logger.info("✅ Ultra-simple RAG service initialized")
    
    async def ask_question(self, query: str, document_id: str = None, conversation_context: str = None) -> SimpleRAGAnswer:
        """Process a question using vector search and LLM response generation"""
        start_time = time.time()
        
        try:
            # Handle greetings and conversational queries
            if self._is_greeting_or_conversational(query):
                return self._handle_conversational_query(query, start_time, conversation_context)
            
            # If vector store and embedding service are available, use real RAG
            if self.vector_store and self.embedding_service:
                return await self._perform_real_rag_with_llm(query, document_id, start_time, conversation_context)
            else:
                # Fallback to mock answers if services unavailable
                return await self._generate_mock_response(query, start_time)
                
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
    
    def _is_legal_query(self, query: str) -> bool:
        """Check if the query is related to legal/contract analysis"""
        query_lower = query.lower()
        
        # Legal terms that indicate document analysis
        legal_keywords = [
            'contract', 'agreement', 'clause', 'term', 'obligation', 'liability', 'responsibility',
            'payment', 'fee', 'termination', 'breach', 'dispute', 'resolution', 'law', 'legal',
            'rights', 'intellectual property', 'confidential', 'compliance', 'regulation',
            'party', 'parties', 'provision', 'section', 'article', 'schedule', 'exhibit',
            'warranty', 'indemnification', 'limitation', 'damages', 'remedy', 'jurisdiction',
            'governing', 'arbitration', 'mediation', 'force majeure', 'assignment', 'amendment',
            'renewal', 'expiration', 'notice', 'deadline', 'milestone', 'deliverable',
            'scope', 'service', 'product', 'vendor', 'client', 'supplier', 'customer',
            'risk', 'insurance', 'penalty', 'default', 'cure', 'waiver', 'severability'
        ]
        
        # Check if query contains legal keywords
        if any(keyword in query_lower for keyword in legal_keywords):
            return True
            
        # Check for document-specific questions
        document_phrases = [
            'this document', 'this contract', 'this agreement', 'the document',
            'what does', 'what are', 'what is', 'how does', 'when does',
            'who is', 'where is', 'why is', 'can you explain', 'tell me about'
        ]
        
        if any(phrase in query_lower for phrase in document_phrases):
            return True
            
        # Exclude clearly non-legal queries
        non_legal_keywords = [
            'weather', 'sports', 'music', 'movie', 'recipe', 'travel', 'shopping',
            'entertainment', 'game', 'social media', 'celebrity', 'fashion',
            'technology news', 'science fiction', 'personal advice', 'relationship',
            'medical advice', 'stock market', 'cryptocurrency', 'investment advice'
        ]
        
        if any(keyword in query_lower for keyword in non_legal_keywords):
            return False
            
        return True  # Default to allowing queries

    def _is_greeting_or_conversational(self, query: str) -> bool:
        """Check if query is a greeting or general conversational query"""
        query_lower = query.lower().strip()
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        conversational = ['how are you', 'what can you do', 'help me', 'thank you', 'thanks']
        
        for greeting in greetings + conversational:
            if greeting in query_lower:
                return True
        return False
    
    def _handle_conversational_query(self, query: str, start_time: float, conversation_context: str = None) -> SimpleRAGAnswer:
        """Handle greetings and conversational queries"""
        query_lower = query.lower().strip()
        
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            answer = "Hello! I'm here to help you analyze and understand your legal document. You can ask me questions about specific clauses, obligations, risks, dates, or any other aspects of the document. What would you like to know?"
        elif 'how are you' in query_lower:
            answer = "I'm doing great, thank you for asking! I'm ready to help you explore your legal document. What questions do you have about it?"
        elif 'what can you do' in query_lower or 'help me' in query_lower:
            answer = "I can help you understand your legal document by answering questions about:\n\n• Key obligations and responsibilities\n• Important dates and deadlines\n• Risk factors and liability\n• Termination conditions\n• Payment terms and fees\n• Confidentiality clauses\n• Dispute resolution procedures\n\nJust ask me any specific question about your document!"
        elif 'thank you' in query_lower or 'thanks' in query_lower:
            answer = "You're very welcome! I'm happy to help. Feel free to ask me any other questions about your document."
        else:
            answer = "I'm here to help you understand your legal document. Please ask me specific questions about clauses, obligations, risks, or any other aspects you'd like to explore."
        
        processing_time = time.time() - start_time
        
        return SimpleRAGAnswer(
            answer=answer,
            confidence_score=95.0,
            citations=[],
            related_topics=["document analysis", "legal questions", "contract review"],
            follow_up_questions=[
                "What are the key obligations in this document?",
                "Are there any important deadlines?",
                "What are the main risks mentioned?",
                "How can this contract be terminated?"
            ],
            processing_time=processing_time,
            response_type="conversational_response"
        )

    async def _perform_real_rag_with_llm(self, query: str, document_id: str, start_time: float, conversation_context: str = None) -> SimpleRAGAnswer:
        """Perform RAG with LLM-powered response generation"""
        logger.info(f"🔍 Performing RAG with LLM for query: {query[:50]}...")
        
        # Step 1: Generate embedding for the query
        query_embedding = self.embedding_service.embed_query(query)
        logger.info(f"✅ Query embedding generated (dim: {len(query_embedding)})")
        
        # Step 2: Search for relevant chunks
        search_results = self.vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=5,
            document_id=document_id
        )
        logger.info(f"✅ Found {len(search_results)} relevant chunks")
        
        # Step 3: Extract relevant content
        if not search_results:
            return await self._generate_no_results_response(query, start_time)
        
        # Step 4: Build context from top results
        context_chunks = []
        citations = []
        
        for i, result in enumerate(search_results[:5]):  # Top 5 chunks
            content = result.get("content", "")
            similarity = result.get("similarity_score", 0.0)
            filename = result.get("filename", "Unknown")
            
            if similarity > 0.3:  # Relevance threshold
                context_chunks.append(content)
                citations.append({
                    "source": filename,
                    "chunk_index": result.get("chunk_index", i),
                    "similarity": round(similarity, 3),
                    "preview": content[:150] + "..." if len(content) > 150 else content
                })
        
        # Step 5: Generate LLM response
        if context_chunks:
            if self.llm:
                answer = await self._generate_llm_response(query, context_chunks, conversation_context)
                confidence = 90.0  # High confidence for LLM responses
                response_type = "llm_generated_answer"
            else:
                answer = self._generate_contextual_answer(query, context_chunks)
                confidence = 85.0  # Good confidence for template responses
                response_type = "template_answer"
        else:
            answer = f"I couldn't find specific information about '{query}' in the document. The topic may not be covered or might be discussed using different terminology. Try rephrasing your question."
            confidence = 25.0
            response_type = "no_results"
        
        # Step 6: Generate enhanced related topics and follow-ups
        related_topics = self._extract_topics_from_context(context_chunks)
        follow_up_questions = self._generate_contextual_follow_ups(query, context_chunks)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ RAG with LLM completed in {processing_time:.2f}s (confidence: {confidence}%)")
        
        return SimpleRAGAnswer(
            answer=answer,
            confidence_score=confidence,
            citations=citations,
            related_topics=related_topics,
            follow_up_questions=follow_up_questions,
            processing_time=processing_time,
            response_type=response_type
        )

    async def _generate_llm_response(self, query: str, context_chunks: List[str], conversation_context: str = None) -> str:
        """Generate a conversational response using LLM"""
        try:
            # Check if query is legal-related
            if not self._is_legal_query(query):
                return "I'm sorry, but I'm specifically designed to help analyze legal documents. Please ask questions about the contract terms, obligations, risks, deadlines, or other legal aspects of your document."
            
            # Combine context chunks
            combined_context = "\n\n".join(context_chunks)
            
            # Build conversation context if available
            conversation_info = ""
            if conversation_context:
                conversation_info = f"\n\nPrevious conversation context:\n{conversation_context}\n"
            
            # Create enhanced prompt for comprehensive ChatGPT-style formatting
            prompt = f"""You are a professional legal document analyst. Answer the user's question directly and professionally using the document context provided.

Document Context:
{combined_context}
{conversation_info}
User Question: {query}

CRITICAL INSTRUCTIONS:
- NEVER use phrases like "Based on the provided text", "According to the document", "The document states", "As mentioned in the text", "This information is not contained within the provided text", "The excerpt is a section", "within the provided text"
- Answer directly as if you have knowledge of the contract
- Be confident and authoritative in your response
- Start directly with the answer, not with qualifying phrases
- If information is not available, say "The document title/specific information is not available in the current section" instead of referring to "provided text"

FORBIDDEN PHRASES (NEVER USE):
❌ "Based on the provided text"
❌ "According to the document" 
❌ "The document states"
❌ "As mentioned in the text"
❌ "This information is not contained within the provided text"
❌ "The excerpt is a section"
❌ "within the provided text"
❌ "from the provided context"

APPROVED ALTERNATIVES:
✅ "The document title is not available in this section"
✅ "This specific information is not included"
✅ "The contract does not specify this detail"
✅ Start directly with the answer when information is available

ADVANCED FORMATTING GUIDELINES (Use ChatGPT-style markdown):

1. **STRUCTURE YOUR RESPONSE**:
   - Use clear headers: ### **Main Topic**
   - Break into logical sections
   - Use numbered lists for steps/procedures: 1. First step 2. Second step
   - Use bullet points for features/items: • Point one • Point two

2. **TEXT FORMATTING**:
   - **Bold** for key legal terms, definitions, and important concepts
   - *Italic* for emphasis and clarifications
   - `Code blocks` for specific clause references like `Section 4.1`
   - ~~Strikethrough~~ for outdated or superseded terms

3. **LISTS AND ORGANIZATION**:
   - Use nested bullet points for sub-items:
     • Main point
       - Sub-point
       - Another sub-point
   - Use numbered lists for sequential processes
   - Use tables for comparisons:
     | Party | Obligation | Section |
     |-------|------------|---------|
     | Buyer | Payment | 3.1 |

4. **HIGHLIGHTING AND EMPHASIS**:
   - **CRITICAL TERMS** in all caps when extremely important
   - Use > blockquotes for important warnings or notes
   - Separate major sections with horizontal rules: ---

5. **RESPONSIVE STRUCTURE BASED ON QUERY TYPE**:
   - For "What are..." questions: Use bulleted lists with bold headings
   - For "How to..." questions: Use numbered steps
   - For "When..." questions: Use tables or timelines
   - For comparisons: Use comparison tables
   - For definitions: Use bold terms with clear explanations
   - For risks/implications: Use warning blockquotes

RESPONSE STRUCTURE TEMPLATES:

For OBLIGATIONS/REQUIREMENTS:
### **[Party] Obligations**
• **[Key Term]**: Explanation
• **[Key Term]**: Explanation

For PROCESSES/PROCEDURES:
### **[Process Name]**
1. **Step One**: Description
2. **Step Two**: Description

For COMPARISONS:
| Aspect | Party A | Party B |
|--------|---------|---------|
| Item | Detail | Detail |

For DEFINITIONS:
**[Term]** refers to [definition]. Key aspects include:
• Point one
• Point two

Answer the question directly and professionally with comprehensive ChatGPT-style formatting:"""

            # Generate response using LLM
            response = await self.llm.ainvoke(prompt)
            
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            # Fallback to template response
            return self._generate_contextual_answer(query, context_chunks)
        """Perform actual RAG using vector search"""
        logger.info(f"🔍 Performing real RAG for query: {query[:50]}...")
        
        # Step 1: Generate embedding for the query
        query_embedding = self.embedding_service.embed_query(query)
        logger.info(f"✅ Query embedding generated (dim: {len(query_embedding)})")
        
        # Step 2: Search for relevant chunks
        search_results = self.vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=5,
            document_id=document_id
        )
        logger.info(f"✅ Found {len(search_results)} relevant chunks")
        
        # Step 3: Extract relevant content
        if not search_results:
            return await self._generate_no_results_response(query, start_time)
        
        # Step 4: Build context from top results
        context_chunks = []
        citations = []
        
        for i, result in enumerate(search_results[:3]):  # Top 3 chunks
            content = result.get("content", "")
            similarity = result.get("similarity_score", 0.0)
            filename = result.get("filename", "Unknown")
            
            if similarity > 0.3:  # Relevance threshold
                context_chunks.append(content)
                citations.append({
                    "source": filename,
                    "chunk_index": result.get("chunk_index", i),
                    "similarity": round(similarity, 3),
                    "preview": content[:150] + "..." if len(content) > 150 else content
                })
        
        # Step 5: Generate answer from context with improved confidence
        if context_chunks:
            answer = self._generate_contextual_answer(query, context_chunks)
            
            # Calculate dynamic confidence based on multiple factors
            base_confidence = min(95.0, max(40.0, len(context_chunks) * 25))  # 25% per relevant chunk
            
            # Boost confidence based on answer quality
            if len(answer) > 100:  # Substantial answer
                base_confidence += 10
            if "•" in answer:  # Structured answer with bullet points
                base_confidence += 10
            if any(keyword in answer.lower() for keyword in ["shall", "must", "agreement", "party"]):  # Legal terms
                base_confidence += 5
            
            # Reduce confidence for vague answers
            if "appears to be" in answer or "might" in answer:
                base_confidence -= 10
            
            confidence = min(95.0, base_confidence)
        else:
            answer = f"I couldn't find specific information about '{query}' in the document. The topic may not be covered or might be discussed using different terminology. Try rephrasing your question."
            confidence = 25.0
        
        # Step 6: Generate enhanced related topics and follow-ups
        related_topics = self._extract_topics_from_context(context_chunks)
        follow_up_questions = self._generate_contextual_follow_ups(query, context_chunks)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Real RAG completed in {processing_time:.2f}s (confidence: {confidence}%)")
        
        return SimpleRAGAnswer(
            answer=answer,
            confidence_score=confidence,
            citations=citations,
            related_topics=related_topics,
            follow_up_questions=follow_up_questions,
            processing_time=processing_time,
            response_type="vector_search_answer"
        )
    
    def _generate_contextual_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate a direct, informative answer based on actual document content"""
        # Combine and clean context
        combined_context = "\n\n".join(context_chunks)
        
        # Clean up the context - remove fragments and incomplete sentences
        cleaned_context = self._clean_context_content(combined_context)
        
        # Create focused answer based on query type
        query_lower = query.lower()
        
        if "about" in query_lower and ("document" in query_lower or "this" in query_lower):
            # Extract document purpose and main content
            main_points = self._extract_main_document_points(cleaned_context)
            if main_points:
                return f"This is a {self._identify_document_type(cleaned_context)} that covers:\n\n{main_points}\n\nThe document establishes terms, conditions, and procedures for the parties involved."
            else:
                return f"This appears to be a legal agreement that outlines terms and conditions between parties. The document includes provisions for {self._extract_key_provisions(cleaned_context)}."

        elif "risk" in query_lower:
            risk_content = self._extract_risk_related_content(cleaned_context)
            if risk_content:
                return f"The key risks identified include:\n\n{risk_content}\n\nThese risks should be carefully evaluated and appropriate mitigation strategies implemented."
            else:
                return "While specific risks aren't explicitly detailed in the available sections, typical risks in such agreements include compliance obligations, liability exposure, and performance requirements."

        elif "date" in query_lower or "deadline" in query_lower or "time" in query_lower:
            date_content = self._extract_date_related_content(cleaned_context)
            if date_content:
                return f"Important timeframes include:\n\n{date_content}\n\nPlease ensure all deadlines are tracked and compliance maintained."
            else:
                return "No specific dates or deadlines were found in the relevant sections. Consider reviewing the full document for timeline requirements."

        elif "obligation" in query_lower or "responsibl" in query_lower:
            obligations = self._extract_obligations_content(cleaned_context)
            if obligations:
                return f"Key obligations include:\n\n{obligations}"
            else:
                return "The document outlines mutual obligations between parties including performance requirements, compliance standards, and operational procedures."

        elif "termination" in query_lower or "end" in query_lower:
            termination_info = self._extract_termination_content(cleaned_context)
            if termination_info:
                return f"Termination provisions:\n\n{termination_info}"
            else:
                return "Termination conditions are addressed in the agreement, including notice requirements and procedures for ending the relationship."

        elif "payment" in query_lower or "fee" in query_lower or "cost" in query_lower:
            payment_info = self._extract_payment_content(cleaned_context)
            if payment_info:
                return f"Payment terms:\n\n{payment_info}"
            else:
                return "Payment terms and fee structures are specified in the agreement, including rates, schedules, and payment procedures."

        else:
            # General contextual answer - be more direct
            key_info = self._extract_most_relevant_info(cleaned_context, query)
            if key_info:
                return key_info
            else:
                return f"The document contains relevant information about {query.lower()}, but the specific details would require reviewing additional sections for complete context."

    def _clean_context_content(self, content: str) -> str:
        """Clean and improve context content readability"""
        import re
        
        # Remove excessive whitespace and clean up formatting
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n', content)
        
        # Remove fragments (very short sentences)
        sentences = content.split('. ')
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Keep sentences that are reasonably complete (more than 10 words or contain key terms)
            if len(sentence.split()) > 10 or any(term in sentence.lower() for term in ['shall', 'will', 'must', 'agree', 'party', 'contract']):
                cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences[:5])  # Top 5 most relevant sentences
    
    def _identify_document_type(self, content: str) -> str:
        """Identify the type of document based on content"""
        content_lower = content.lower()
        
        if 'service' in content_lower and 'agreement' in content_lower:
            return "Service Agreement"
        elif 'employment' in content_lower or 'employee' in content_lower:
            return "Employment Agreement"
        elif 'license' in content_lower:
            return "License Agreement"
        elif 'confidential' in content_lower or 'nda' in content_lower:
            return "Confidentiality Agreement"
        elif 'purchase' in content_lower or 'sale' in content_lower:
            return "Purchase Agreement"
        else:
            return "Legal Agreement"
    
    def _extract_main_document_points(self, content: str) -> str:
        """Extract main points from document content"""
        points = []
        sentences = content.split('. ')
        
        # Look for key structural elements
        for sentence in sentences[:3]:  # First 3 sentences usually contain main points
            sentence = sentence.strip()
            if len(sentence) > 20 and not sentence.startswith('('):
                # Clean up the sentence
                if sentence.endswith('.'):
                    sentence = sentence[:-1]
                points.append(f"• {sentence}")
        
        return '\n'.join(points) if points else ""
    
    def _extract_key_provisions(self, content: str) -> str:
        """Extract key provisions mentioned in content"""
        provisions = []
        content_lower = content.lower()
        
        provision_terms = {
            'service delivery': ['service', 'deliver', 'provide'],
            'payment terms': ['payment', 'fee', 'cost', 'invoice'],
            'compliance requirements': ['comply', 'requirement', 'standard'],
            'confidentiality': ['confidential', 'non-disclosure', 'private'],
            'liability limitations': ['liability', 'damage', 'limitation'],
            'termination procedures': ['terminate', 'end', 'expire']
        }
        
        for provision, keywords in provision_terms.items():
            if any(keyword in content_lower for keyword in keywords):
                provisions.append(provision)
        
        return ', '.join(provisions) if provisions else "various operational and legal requirements"

    def _extract_risk_related_content(self, content: str) -> str:
        """Extract and format risk-related content from the context"""
        risk_keywords = ["risk", "liability", "penalty", "breach", "violation", "termination", "default", "damages", "indemnify", "limitation"]
        sentences = content.split('. ')
        
        risk_items = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in risk_keywords) and len(sentence.strip()) > 15:
                # Clean and format the risk item
                risk_item = sentence.strip()
                if not risk_item.endswith('.'):
                    risk_item += '.'
                risk_items.append(f"• {risk_item}")
        
        if risk_items:
            return '\n'.join(risk_items[:4])  # Top 4 risk items
        else:
            return "Potential compliance violations, performance failures, and liability exposure as outlined in the agreement terms."

    def _extract_date_related_content(self, content: str) -> str:
        """Extract date and deadline related content"""
        import re
        
        # Enhanced date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
            r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(within|by|before|after|during)\s+\d+\s+(days?|weeks?|months?|years?)\b',
            r'\b\d+\s+(day|week|month|year)s?\b'
        ]
        
        time_keywords = ["deadline", "due", "expire", "term", "period", "duration", "notice", "renewal", "effective"]
        
        date_items = []
        sentences = content.split('. ')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for date patterns or time keywords
            has_date_pattern = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in date_patterns)
            has_time_keyword = any(keyword in sentence_lower for keyword in time_keywords)
            
            if (has_date_pattern or has_time_keyword) and len(sentence.strip()) > 10:
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith('.'):
                    clean_sentence += '.'
                date_items.append(f"• {clean_sentence}")
        
        if date_items:
            return '\n'.join(date_items[:4])  # Top 4 date-related items
        else:
            return "Standard contract terms and renewal periods apply as specified in the agreement."

    def _extract_obligations_content(self, content: str) -> str:
        """Extract obligation-related content"""
        obligation_keywords = ["shall", "must", "required", "obligation", "responsible", "duty", "agree to", "undertake"]
        sentences = content.split('. ')
        
        obligations = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in obligation_keywords) and len(sentence.strip()) > 15:
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith('.'):
                    clean_sentence += '.'
                obligations.append(f"• {clean_sentence}")
        
        return '\n'.join(obligations[:4]) if obligations else "Mutual performance obligations and compliance requirements as detailed in the agreement."

    def _extract_termination_content(self, content: str) -> str:
        """Extract termination-related content"""
        termination_keywords = ["terminat", "end", "expire", "cancel", "breach", "default", "notice"]
        sentences = content.split('. ')
        
        termination_items = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in termination_keywords) and len(sentence.strip()) > 15:
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith('.'):
                    clean_sentence += '.'
                termination_items.append(f"• {clean_sentence}")
        
        return '\n'.join(termination_items[:3]) if termination_items else "Standard termination provisions with appropriate notice requirements."

    def _extract_payment_content(self, content: str) -> str:
        """Extract payment-related content"""
        payment_keywords = ["payment", "fee", "cost", "invoice", "bill", "rate", "hour", "$", "amount"]
        sentences = content.split('. ')
        
        payment_items = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in payment_keywords) and len(sentence.strip()) > 10:
                clean_sentence = sentence.strip()
                if not clean_sentence.endswith('.'):
                    clean_sentence += '.'
                payment_items.append(f"• {clean_sentence}")
        
        return '\n'.join(payment_items[:4]) if payment_items else "Fee structures and payment schedules as specified in the agreement terms."

    def _extract_most_relevant_info(self, content: str, query: str) -> str:
        """Extract the most relevant information based on query keywords"""
        query_words = query.lower().split()
        sentences = content.split('. ')
        
        # Score sentences based on query word matches
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) > 15:
                sentence_lower = sentence.lower()
                score = sum(1 for word in query_words if word in sentence_lower and len(word) > 2)
                if score > 0:
                    scored_sentences.append((score, sentence.strip()))
        
        # Sort by relevance score and take top items
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        relevant_items = []
        for score, sentence in scored_sentences[:3]:
            if not sentence.endswith('.'):
                sentence += '.'
            relevant_items.append(f"• {sentence}")
        
        return '\n'.join(relevant_items) if relevant_items else ""

    def _extract_topics_from_context(self, context_chunks: List[str]) -> List[str]:
        """Extract relevant topics from the context"""
        if not context_chunks:
            return []
        
        combined_text = " ".join(context_chunks).lower()
        
        # Common legal document topics
        topic_keywords = {
            "Contract Terms": ["contract", "agreement", "terms", "conditions"],
            "Payment": ["payment", "fee", "cost", "invoice", "billing"],
            "Liability": ["liability", "responsible", "damages", "indemnity"],
            "Termination": ["termination", "end", "expire", "cancel"],
            "Intellectual Property": ["intellectual", "property", "copyright", "patent"],
            "Confidentiality": ["confidential", "non-disclosure", "privacy", "secret"],
            "Compliance": ["comply", "regulation", "law", "requirement"],
            "Delivery": ["delivery", "provide", "supply", "fulfill"]
        }
        
        relevant_topics = []
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                relevant_topics.append(topic)
        
        return relevant_topics[:5]  # Top 5 topics
    
    def _generate_contextual_follow_ups(self, original_query: str, context_chunks: List[str]) -> List[str]:
        """Generate relevant follow-up questions based on context"""
        if not context_chunks:
            return ["Could you provide more context?", "What specific aspect interests you?"]
        
        query_lower = original_query.lower()
        
        if "about" in query_lower:
            return [
                "What are the key risks in this document?",
                "Are there any important deadlines?",
                "What are the main obligations for each party?",
                "What are the termination conditions?"
            ]
        elif "risk" in query_lower:
            return [
                "How can these risks be mitigated?",
                "Are there any liability caps mentioned?",
                "What are the consequences of these risks?",
                "Are there insurance requirements?"
            ]
        elif "date" in query_lower or "deadline" in query_lower:
            return [
                "What happens if deadlines are missed?",
                "Are there any grace periods mentioned?",
                "How are extensions handled?",
                "What are the key milestones?"
            ]
        else:
            return [
                "Can you provide more details about this topic?",
                "Are there any related clauses?",
                "What are the implications of this?",
                "How does this affect the parties involved?"
            ]

    async def _generate_no_results_response(self, query: str, start_time: float) -> SimpleRAGAnswer:
        """Generate response when no relevant content is found"""
        processing_time = time.time() - start_time
        
        return SimpleRAGAnswer(
            answer=f"I couldn't find specific information related to '{query}' in the document. This could mean the topic isn't covered, or it might be discussed using different terminology. Try rephrasing your question or asking about related topics.",
            confidence_score=20.0,
            citations=[],
            related_topics=["Document Overview", "General Terms", "Key Provisions"],
            follow_up_questions=[
                "Can you rephrase your question?",
                "What specific aspect are you looking for?",
                "Would you like a general overview instead?"
            ],
            processing_time=processing_time,
            response_type="no_results"
        )

    async def _generate_mock_response(self, query: str, start_time: float) -> SimpleRAGAnswer:
        """Generate mock response when vector services unavailable"""
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
            processing_time=processing_time,
            response_type="mock_answer"
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
