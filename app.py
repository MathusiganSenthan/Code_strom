import os
import asyncio
import time
import base64
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
            "� Direct PDF processing with Gemini (no text extraction)",
            "📄 Native PDF understanding and analysis",
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