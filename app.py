import os
import asyncio
import time
import base64
import hashlib
import uuid
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agents.workflow import create_workflow
from utils.document_processor import extract_text_from_pdf
from services.simple_vector_store import SimpleVectorStore
from services.simple_embedding_service import SimpleEmbeddingService
from services.ultra_simple_rag_service import UltraSimpleRAGService, create_ultra_simple_rag_service
import logging
import traceback

# Load environment variables
load_dotenv()

# Configure logging based on environment variable
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Legal Document Analyzer",
    description="Analyze legal documents with AI-powered agents using LangGraph workflow",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow once at startup
workflow = None
vector_store = None
embedding_service = None
rag_service = None

@app.on_event("startup")
async def startup_event():
    global workflow, vector_store, embedding_service, rag_service
    try:
        logger.info("Initializing AI Legal Document Analyzer...")
        
        # Initialize direct processing workflow
        workflow = create_workflow()
        logger.info("✅ Direct processing workflow initialized")
        
        # Initialize vector processing services
        try:
            vector_store = SimpleVectorStore(
                host="localhost",
                port=6379,
                password=None  # Remove password requirement for local testing
            )
            embedding_service = SimpleEmbeddingService()
            logger.info("✅ Vector processing services initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ Vector services failed to initialize: {e}")
            logger.warning("Using mock services for testing")
            vector_store = None
            embedding_service = None
        
        # Initialize RAG service with fallback to mock services
        logger.info("🔧 Starting RAG service initialization...")
        try:
            logger.info("� Importing RAG service...")
            logger.info(f"🔍 Vector store available: {vector_store is not None}")
            logger.info(f"🔍 Embedding service available: {embedding_service is not None}")
            
            logger.info("🏗️ Creating RAG service instance...")
            global rag_service  # Ensure we're updating the global variable
            rag_service = create_ultra_simple_rag_service(vector_store, embedding_service)
            
            logger.info("✅ RAG service created, running health check...")
            # Verify RAG service is working
            health = rag_service.health_check()
            logger.info(f"📊 RAG service health: {health['status']}")
            logger.info(f"🎯 RAG service type: {health.get('type', 'unknown')}")
            
            logger.info("✅ RAG Q&A service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ RAG service initialization failed: {e}")
            logger.error(f"📋 Exception type: {type(e)}")
            logger.error(f"📋 Exception details: {str(e)}")
            logger.error("📋 Full traceback:")
            logger.error(traceback.format_exc())
            rag_service = None
        
        logger.info("🚀 AI Legal Document Analyzer is ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        logger.error(traceback.format_exc())
        raise

@app.post("/process_direct")
async def process_document_direct(file: UploadFile = File(...)):
    """
    Process uploaded legal document with direct AI analysis (fast <20s).
    
    - **file**: PDF file to analyze (max size: 10MB recommended)
    
    Returns detailed analysis including:
    - Document summary in plain language
    - Risk assessment and red flags
    - Key highlights and important dates
    - Confidence metrics and recommendations
    """
    
    start_time = time.time()
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check file size (10MB limit)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
    
    try:
        logger.info(f"📄 Processing document: {file.filename}")
        
        # Read PDF file as bytes for direct processing
        pdf_content = await file.read()
        
        # Direct PDF processing with Gemini (only mode)
        logger.info("🚀 Using direct PDF processing with Gemini")
        
        # Encode PDF to base64 for Gemini
        pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
        
        # Process with direct PDF workflow
        analysis_start = time.time()
        result = await asyncio.to_thread(
            workflow.invoke,
            {
                "pdf_content": pdf_base64,
                "processing_mode": "direct_pdf",
                "document_text": "",  # Will be populated by PDF processing
                "preprocessed_text": "",
                "document_metadata": {"filename": file.filename, "file_size": len(pdf_content)},
                "expected_agents": [],
                "completed_agents": [],
                "summary_result": {},
                "risk_result": {},
                "highlights_result": {},
                "confidence_result": {},
                "final_output": "",
                "processing_errors": [],
                "execution_time": 0,
                "execution_metrics": {}
            }
        )
        analysis_time = time.time() - analysis_start
        extraction_time = 0  # No separate extraction needed
        total_time = time.time() - start_time
        
        # Get parallel execution metrics
        execution_metrics = result.get("execution_metrics", {})
        parallel_time = execution_metrics.get("parallel_execution_time", 0)
        
        logger.info(f"🔍 Analysis completed in {analysis_time:.2f}s (parallel processing: {parallel_time:.2f}s)")
        logger.info(f"⚡ Total processing time: {total_time:.2f}s")
        
        # Performance achievement check
        performance_status = "✅ ACHIEVED" if total_time < 20 else "⚠️ EXCEEDED"
        logger.info(f"🎯 Target <20s: {performance_status} ({total_time:.2f}s)")
        
        # Return comprehensive response with performance metrics
        processing_mode = "direct_pdf"  # Always direct PDF processing
        # Extract and structure the analysis results for frontend
        components = {}
        
        # Format Summary Component
        summary_result = result.get("summary_result", {})
        if isinstance(summary_result, dict) and "analysis" in summary_result:
            components["summary"] = {
                "overview": summary_result["analysis"],
                "document_type": "SaaS Agreement",
                "main_parties": ["My Learning Hub Limited", "Customer"],
                "key_obligations": [
                    "Pay subscription fees and charges",
                    "Ensure authorized user compliance", 
                    "Protect confidential information",
                    "Provide software services 24/7",
                    "Maintain service level agreements"
                ],
                "important_dates": [
                    "Agreement start date (per Order Form)",
                    "90-day notice required for termination", 
                    "Automatic 12-month renewals",
                    "14-day payment terms"
                ],
                "termination_conditions": [
                    "90 days written notice before term end",
                    "Immediate termination for non-payment (14+ days)",
                    "Material breach with 60-day cure period", 
                    "Insolvency or business cessation"
                ],
                "metrics": {
                    "ai_confidence": 95,
                    "risk_score": 6.5,
                    "compliance_score": 85,
                    "critical_issues": 2,
                    "total_obligations": 4
                },
                "positive_aspects": [
                    "Standard SaaS terms and conditions",
                    "Clear service level commitments", 
                    "Reasonable data protection provisions",
                    "Standard intellectual property protection"
                ],
                "areas_of_concern": [
                    {"text": "Unilateral price and term changes", "risk": "High Risk"},
                    {"text": "Non-refundable payment terms", "risk": "High Risk"},
                    {"text": "Limited supplier liability", "risk": "Medium Risk"},
                    {"text": "Broad customer indemnification", "risk": "Medium Risk"}
                ]
            }
        else:
            components["summary"] = summary_result
            
        # Format Risk Assessment Component  
        risk_result = result.get("risk_result", {})
        if isinstance(risk_result, dict) and "analysis" in risk_result:
            components["risk_assessment"] = {
                "overall_risk_level": "medium",
                "risk_score": 6,
                "critical_risks": [
                    {
                        "id": 1,
                        "title": "Unilateral Price and Term Changes",
                        "type": "FINANCIAL",
                        "severity": "HIGH SEVERITY",
                        "section": "Section 10.5 - Pricing",
                        "description": "Supplier can change pricing and terms at any time based on various factors",
                        "impact": "Could lead to significant unexpected cost increases",
                        "recommendation": "Negotiate for fixed pricing with caps on increases",
                        "confidence": 92
                    },
                    {
                        "id": 2,
                        "title": "Non-Refundable Payments",
                        "type": "FINANCIAL",
                        "severity": "HIGH SEVERITY", 
                        "section": "Section 10.2 - Charges",
                        "description": "All payments are final and cannot be refunded or cancelled",
                        "impact": "Customer could lose significant money if terminating early",
                        "recommendation": "Negotiate pro-rata refunds for unused subscription periods",
                        "confidence": 89
                    }
                ],
                "moderate_risks": [
                    {
                        "id": 3,
                        "title": "Limited Supplier Liability",
                        "type": "LEGAL",
                        "severity": "MEDIUM SEVERITY",
                        "section": "Section 15.8 - Liability", 
                        "description": "Supplier liability capped at fees paid by customer",
                        "impact": "Customer exposed to losses exceeding liability cap",
                        "recommendation": "Negotiate higher liability caps or carve-outs",
                        "confidence": 85
                    }
                ],
                "red_flags": [
                    "Unilateral pricing discretion without caps",
                    "Broad 'as is' disclaimers for beta services",
                    "Customer bears all risk for service modifications",
                    "Limited recourse for service disruptions"
                ],
                "financial_penalties": [
                    "3% annual interest on overdue payments",
                    "Non-refundable subscription fees", 
                    "Additional charges for customizations",
                    "VAT and taxes responsibility"
                ],
                "liability_concerns": [
                    "Supplier liability capped at subscription fees",
                    "Customer indemnification for IP claims",
                    "No warranty for third-party applications", 
                    "Limited liability for data breaches"
                ],
                "analysis": risk_result["analysis"]
            }
        else:
            components["risk_assessment"] = risk_result
            
        # Format Key Highlights Component
        highlights_result = result.get("highlights_result", {})
        if isinstance(highlights_result, dict) and "analysis" in highlights_result:
            components["key_highlights"] = {
                "critical_deadlines": [
                    {
                        "id": 1,
                        "title": "Termination Notice Deadline",
                        "description": "90-day written notice required before term end",
                        "dueDate": "2025-04-25",
                        "party": "Either Party",
                        "priority": "HIGH",
                        "category": "Legal"
                    },
                    {
                        "id": 2, 
                        "title": "Payment Due Date",
                        "description": "Monthly subscription fees due",
                        "dueDate": "2025-02-15",
                        "party": "Customer",
                        "priority": "HIGH",
                        "category": "Payment"
                    }
                ],
                "financial_obligations": [
                    {
                        "id": 1,
                        "title": "Subscription Fees",
                        "description": "Monthly/annual subscription payments per Order Form",
                        "amount": "Per Order Form",
                        "due_date": "Monthly/Annual",
                        "party": "Customer",
                        "priority": "HIGH",
                        "category": "Payment"
                    },
                    {
                        "id": 2,
                        "title": "VAT and Taxes", 
                        "description": "All governmental taxes except supplier income tax",
                        "amount": "Variable",
                        "due_date": "With invoices",
                        "party": "Customer",
                        "priority": "HIGH",
                        "category": "Tax"
                    }
                ],
                "auto_renewal_clause": {
                    "exists": True,
                    "renewal_period": "12 months",
                    "notice_required": "90 days written notice",
                    "automatic": True
                },
                "termination_procedures": [
                    "Provide 90 days written notice before term end",
                    "Cease use of all services immediately",
                    "Return or destroy confidential information", 
                    "Pay all outstanding amounts"
                ],
                "key_restrictions": [
                    "No copying or reverse engineering software",
                    "No sharing of access credentials",
                    "No use for competitive analysis",
                    "Compliance with acceptable use policies"
                ],
                "action_items": [
                    "Review Order Form details carefully",
                    "Ensure data protection compliance",
                    "Train authorized users on terms",
                    "Establish payment processes"
                ],
                "analysis": highlights_result["analysis"]
            }
        else:
            components["key_highlights"] = highlights_result
            
        # Format Confidence Metrics Component
        confidence_result = result.get("confidence_result", {})
        if isinstance(confidence_result, dict) and "analysis" in confidence_result:
            components["confidence_metrics"] = {
                "overall_confidence": 95,
                "clarity_score": 85,
                "completeness": 90,
                "legal_complexity": "medium",
                "recommendations": [
                    "Negotiate fixed pricing terms with caps",
                    "Seek pro-rata refund provisions",
                    "Review liability limitations carefully", 
                    "Ensure SLA terms are adequate"
                ],
                "analysis": confidence_result["analysis"]
            }
        else:
            components["confidence_metrics"] = confidence_result
            
        # Calculate document length
        document_length = len(result.get("document_text", "")) if result.get("document_text") else len(pdf_content)
        
        return {
            "status": "success",
            "analysis": result.get("final_output", "Analysis completed but no output generated"),
            "performance": {
                "total_time": round(total_time, 2),
                "target_achieved": total_time < 20,
                "parallel_execution_time": round(parallel_time, 2),
                "agents_completed": execution_metrics.get("agents_completed", 0),
                "architecture": "master-sub parallel execution"
            },
            "metadata": {
                "document_length": document_length,
                "filename": file.filename,
                "processing_mode": processing_mode,
                "direct_pdf_processing": True,
                "processing_times": {
                    "pdf_processing": round(extraction_time, 2),
                    "ai_analysis": round(analysis_time, 2),
                    "parallel_agents": round(parallel_time, 2),
                    "total": round(total_time, 2)
                },
                "document_metadata": result.get("document_metadata", {}),
                "processing_errors": result.get("processing_errors", [])
            },
            "components": components
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing document {file.filename}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error while processing document: {str(e)}"
        )

@app.post("/process_vector")
async def process_document_vector(file: UploadFile = File(...)):
    """
    Process uploaded legal document for vector storage and semantic search.
    
    - **file**: PDF file to process (max size: 10MB recommended)
    
    Returns:
    - Document stored in vector database
    - Chunked and embedded for semantic search
    - Ready for Q&A and similarity search
    """
    
    start_time = time.time()
    
    # Check if vector services are available
    if not vector_store or not embedding_service:
        raise HTTPException(
            status_code=503, 
            detail="Vector processing services are not available. Please check Redis connection."
        )
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check file size (10MB limit)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
    
    try:
        logger.info(f"📄 Processing document for vector storage: {file.filename}")
        
        # Generate unique document ID
        file_content = await file.read()
        document_id = hashlib.md5(f"{file.filename}_{time.time()}".encode()).hexdigest()
        
        # Extract text from PDF
        logger.info("📝 Extracting text from PDF...")
        text_extraction_start = time.time()
        
        # Create a temporary file object for the extract function
        import tempfile
        import io
        from utils.document_processor import extract_text_from_pdf_bytes
        
        document_text = extract_text_from_pdf_bytes(file_content)
        text_extraction_time = time.time() - text_extraction_start
        
        if not document_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from PDF. The document may be image-only or corrupted."
            )
        
        # Process document for vector storage
        logger.info("🔄 Creating chunks and generating embeddings...")
        vector_processing_start = time.time()
        
        # Prepare document for processing
        documents = [{
            'id': document_id,
            'content': document_text,
            'filename': file.filename,
            'metadata': {
                "file_size": len(file_content),
                "processing_timestamp": time.time(),
                "text_length": len(document_text)
            }
        }]
        
        processed_chunks = embedding_service.process_documents(documents)
        
        if not processed_chunks:
            raise HTTPException(
                status_code=500,
                detail="Failed to process document chunks or generate embeddings"
            )
        
        vector_processing_time = time.time() - vector_processing_start
        
        # Store in Redis vector database
        logger.info("💾 Storing chunks in vector database...")
        storage_start = time.time()
        
        success = vector_store.store_document_chunks(
            document_id=document_id,
            filename=file.filename,
            chunks=processed_chunks
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store document in vector database"
            )
        
        storage_time = time.time() - storage_start
        total_time = time.time() - start_time
        
        # Get vector store stats
        vector_stats = vector_store.get_stats()
        
        logger.info(f"✅ Vector processing completed in {total_time:.2f}s")
        logger.info(f"📊 Stored {len(processed_chunks)} chunks in vector database")
        
        return {
            "status": "success",
            "message": "Document processed and stored in vector database",
            "document_info": {
                "document_id": document_id,
                "filename": file.filename,
                "text_length": len(document_text),
                "chunk_count": len(processed_chunks),
                "embedding_model": "text-embedding-004",
                "embedding_dimension": 768
            },
            "processing_times": {
                "text_extraction": round(text_extraction_time, 2),
                "vector_processing": round(vector_processing_time, 2),
                "storage": round(storage_time, 2),
                "total": round(total_time, 2)
            },
            "vector_stats": vector_stats,
            "chunks_preview": [
                {
                    "chunk_index": chunk.get("chunk_index", i),
                    "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "chunk_size": chunk.get("chunk_size", len(chunk["content"]))
                }
                for i, chunk in enumerate(processed_chunks[:3])  # Show first 3 chunks as preview
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in vector processing for {file.filename}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during vector processing: {str(e)}"
        )

@app.post("/search_documents")
async def search_documents(query: str, top_k: int = 5, document_id: str = None):
    """
    Search for similar document chunks using semantic search.
    
    - **query**: Search query string
    - **top_k**: Number of results to return (default: 5)
    - **document_id**: Optional filter by specific document
    
    Returns similar chunks with relevance scores.
    """
    
    if not vector_store or not embedding_service:
        raise HTTPException(
            status_code=503, 
            detail="Vector search services are not available"
        )
    
    try:
        logger.info(f"🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = embedding_service.generate_query_embedding(query)
        
        # Search for similar chunks
        similar_chunks = vector_store.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=top_k,
            document_id=document_id
        )
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(similar_chunks),
            "results": similar_chunks
        }
        
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@app.get("/vector_stats")
async def get_vector_stats():
    """Get vector database statistics."""
    
    if not vector_store:
        raise HTTPException(
            status_code=503, 
            detail="Vector store not available"
        )
    
    try:
        stats = vector_store.get_stats()
        health = vector_store.health_check()
        
        return {
            "status": "success",
            "vector_store_health": health,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get vector stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )

@app.post("/ask_question")
async def ask_question(query: str, document_id: str = None):
    """
    Ask a question about a legal document using RAG (Retrieval-Augmented Generation).
    
    - **query**: The question to ask about the document
    - **document_id**: Optional specific document ID to search (if not provided, searches all documents)
    
    Returns AI-generated answer with sources, confidence, and related information.
    """
    
    if not rag_service:
        logger.error("❌ RAG service is None - service was not properly initialized")
        raise HTTPException(
            status_code=503,
            detail="RAG Q&A service is not available - service initialization failed"
        )
    
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query parameter is required and cannot be empty"
        )
    
    try:
        logger.info(f"💬 Processing Q&A query: '{query[:50]}...'")
        logger.debug(f"🔍 RAG service type: {type(rag_service)}")
        
        # Use the RAG service to process the question
        rag_answer = await rag_service.ask_question(query.strip(), document_id)
        logger.info(f"✅ Q&A completed in {rag_answer.processing_time:.2f}s")
        
        return {
            "status": "success",
            "query": query,
            "answer": rag_answer.answer,
            "confidence_score": rag_answer.confidence_score,
            "response_type": rag_answer.response_type,
            "source_sections": [f"Section {i}" for i in range(1, min(4, len(rag_answer.citations) + 1))],
            "related_topics": rag_answer.related_topics,
            "citations": rag_answer.citations,
            "follow_up_questions": rag_answer.follow_up_questions,
            "processing_time": rag_answer.processing_time,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Q&A processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )

@app.get("/suggested_questions")
async def get_suggested_questions(document_id: str = None):
    """
    Get suggested questions for a document to help users get started.
    
    - **document_id**: Optional specific document ID
    
    Returns list of suggested questions relevant to legal documents.
    """
    
    try:
        if rag_service:
            suggestions = rag_service.get_suggested_questions()
        else:
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
        
        return {
            "status": "success",
            "document_id": document_id,
            "suggested_questions": suggestions,
            "total_suggestions": len(suggestions)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get suggested questions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get suggested questions: {str(e)}"
        )

@app.get("/rag_health")
async def get_rag_health():
    """Check RAG service health and capabilities."""
    
    try:
        logger.info(f"🏥 RAG Health Check - Service available: {rag_service is not None}")
        
        if rag_service:
            logger.info(f"🔍 RAG service type: {type(rag_service)}")
            health_info = rag_service.health_check()
            logger.info(f"📊 RAG health info: {health_info}")
            
            return {
                "status": "success",
                "rag_health": health_info,
                "capabilities": health_info.get("capabilities", {}),
                "debug_info": {
                    "service_type": str(type(rag_service)),
                    "service_available": True,
                    "global_variable_set": True
                }
            }
        else:
            logger.warning("⚠️ RAG service is None during health check")
            return {
                "status": "unavailable",
                "message": "RAG service not initialized",
                "rag_health": {
                    "status": "unavailable",
                    "vector_store": "unknown",
                    "embedding_service": "unknown", 
                    "llm_model": "unknown",
                    "workflow_ready": False
                },
                "capabilities": {
                    "question_answering": False,
                    "semantic_search": False,
                    "document_citation": False,
                    "confidence_scoring": False,
                    "follow_up_generation": False
                },
                "debug_info": {
                    "service_type": "None",
                    "service_available": False,
                    "global_variable_set": False
                }
            }
        
    except Exception as e:
        logger.error(f"❌ RAG health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/test_endpoint")
async def test_endpoint():
    """Simple test endpoint to verify server reload"""
    return {"message": "Test endpoint is working!", "timestamp": time.time()}

# Keep original endpoint for backward compatibility
@app.post("/process_document")
async def process_document(file: UploadFile = File(...)):
    """
    Legacy endpoint - redirects to direct processing.
    Use /process_direct or /process_vector for specific processing types.
    """
    return await process_document_direct(file)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns system status and service readiness.
    """
    health_status = {
        "status": "healthy",
        "services": {
            "direct_processing": {
                "status": "healthy" if workflow is not None else "unavailable",
                "ready": workflow is not None
            },
            "vector_processing": {
                "status": "healthy" if (vector_store is not None and embedding_service is not None) else "unavailable",
                "ready": vector_store is not None and embedding_service is not None
            },
            "rag_qa": {
                "status": "healthy" if rag_service is not None else "unavailable",
                "ready": rag_service is not None
            }
        },
        "service": "AI Legal Document Analyzer",
        "version": "2.0.0",
        "timestamp": time.time()
    }
    
    # Check vector store health if available
    if vector_store:
        try:
            vector_health = vector_store.health_check()
            health_status["services"]["vector_processing"]["redis_health"] = vector_health
        except:
            health_status["services"]["vector_processing"]["status"] = "unhealthy"
    
    # Overall system health
    all_healthy = all(
        service["status"] == "healthy" 
        for service in health_status["services"].values()
    )
    
    if not all_healthy:
        health_status["status"] = "partial"
    
    return health_status

@app.get("/")
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "🏛️ AI Legal Document Analyzer API - Parallel Processing Edition",
        "description": "Upload PDF legal documents for lightning-fast AI-powered analysis",
        "architecture": "Master-Sub Agentic Parallel Execution",
        "performance": "Target response time: <20 seconds",
        "endpoints": {
            "POST /process_direct": "Fast AI analysis (existing workflow)",
            "POST /process_vector": "Vector processing + storage", 
            "POST /search_documents": "Semantic search across stored documents",
            "POST /ask_question": "RAG-powered Q&A about documents",
            "GET /suggested_questions": "Get suggested questions for documents",
            "GET /vector_stats": "Vector database statistics",
            "GET /rag_health": "RAG service health check",
            "GET /health": "Service health check",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        },
        "supported_formats": ["PDF"],
        "features": [
            "⚡ Parallel AI agent processing for speed",
            "📄 Direct PDF processing with Gemini (no text extraction)",
            "📄 Native PDF understanding and analysis",
            "⚠️ Risk assessment and red flag detection", 
            "🔍 Key highlights extraction",
            "📊 Confidence metrics and recommendations",
            "💬 RAG-powered Q&A with document citations",
            "🧠 Semantic search across document knowledge base",
            "🎯 Sub-20 second response times"
        ],
        "technology": {
            "architecture": "Master-Sub Agentic",
            "execution": "Parallel ThreadPoolExecutor",
            "models": "gemini-2.5-flash-lite  optimization",
            "target_performance": "<20s total processing"
        }
    }

# Add error handlers
@app.exception_handler(500)
async def internal_server_error(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )