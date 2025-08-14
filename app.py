import os
import asyncio
import time
from typing import Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from agents.workflow import create_workflow
from utils.document_processor import extract_text_from_pdf
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

@app.on_event("startup")
async def startup_event():
    global workflow
    try:
        logger.info("Initializing AI Legal Document Analyzer...")
        workflow = create_workflow()
        logger.info("✅ Workflow initialized successfully")
        logger.info("🚀 AI Legal Document Analyzer is ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize workflow: {e}")
        logger.error(traceback.format_exc())
        raise

@app.post("/process_document")
async def process_document(file: UploadFile = File(...)):
    """
    Process uploaded legal document and return comprehensive analysis.
    
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
        
        # Extract text from PDF
        extraction_start = time.time()
        document_text = await extract_text_from_pdf(file)
        extraction_time = time.time() - extraction_start
        
        if not document_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text could be extracted from the PDF. Please ensure the PDF contains readable text."
            )
        
        logger.info(f"📝 Text extracted in {extraction_time:.2f}s ({len(document_text)} characters)")
        
        # Process with parallel workflow
        analysis_start = time.time()
        result = await asyncio.to_thread(
            workflow.invoke,
            {
                "document_text": document_text,
                "preprocessed_text": "",
                "document_metadata": {},
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
                "document_length": len(document_text),
                "filename": file.filename,
                "processing_times": {
                    "text_extraction": round(extraction_time, 2),
                    "ai_analysis": round(analysis_time, 2),
                    "parallel_agents": round(parallel_time, 2),
                    "total": round(total_time, 2)
                },
                "document_metadata": result.get("document_metadata", {}),
                "processing_errors": result.get("processing_errors", [])
            },
            "components": {
                "summary": result.get("summary_result", {}),
                "risk_assessment": result.get("risk_result", {}),
                "key_highlights": result.get("highlights_result", {}),
                "confidence_metrics": result.get("confidence_result", {})
            }
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

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    
    Returns system status and workflow readiness.
    """
    return {
        "status": "healthy" if workflow is not None else "initializing",
        "workflow_ready": workflow is not None,
        "service": "AI Legal Document Analyzer",
        "version": "1.0.0",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "🏛️ AI Legal Document Analyzer API - Parallel Processing Edition",
        "description": "Upload PDF legal documents for lightning-fast AI-powered analysis",
        "architecture": "Master-Sub Agentic Parallel Execution",
        "performance": "Target response time: <20 seconds",
        "endpoints": {
            "POST /process_document": "Upload and analyze a PDF document",
            "GET /health": "Check service health",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation"
        },
        "supported_formats": ["PDF"],
        "features": [
            "⚡ Parallel AI agent processing for speed",
            "📄 Document summarization in plain language",
            "⚠️ Risk assessment and red flag detection", 
            "🔍 Key highlights extraction",
            "📊 Confidence metrics and recommendations",
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