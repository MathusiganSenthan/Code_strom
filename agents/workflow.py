import asyncio
from typing import Dict, Any, List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from agents.schemas import Summary, RiskAssessment, KeyHighlights, ConfidenceAssessment
from agents.prompts import *
import os
import logging
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Define the state structure for LangGraph
class GraphState(TypedDict):
    document_text: str
    preprocessed_text: str
    summary_result: Dict[str, Any]
    risk_result: Dict[str, Any]
    highlights_result: Dict[str, Any]
    confidence_result: Dict[str, Any]
    final_output: str
    completed_agents: List[str]
    expected_agents: List[str]
    processing_errors: List[str]
    document_metadata: Dict[str, Any]
    execution_time: float
    execution_metrics: Dict[str, Any]

class ImprovedLegalAnalyzer:
    def __init__(self):
        # Use Pro model for complex analysis and Flash for preprocessing
        self.pro_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_output_tokens=4096
        )
        
        self.flash_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Create chains with better error handling
        self.summarizer_chain = self._create_chain(SUMMARIZER_PROMPT, Summary, self.pro_llm)
        self.risk_chain = self._create_chain(RISK_ANALYZER_PROMPT, RiskAssessment, self.pro_llm)
        self.highlighter_chain = self._create_chain(HIGHLIGHTER_PROMPT, KeyHighlights, self.flash_llm)
        self.confidence_chain = self._create_chain(CONFIDENCE_PROMPT, ConfidenceAssessment, self.flash_llm)
        
    def _create_chain(self, prompt, schema, llm):
        """Create a chain with better error handling."""
        return prompt | llm.with_structured_output(schema, include_raw=True)

    def roughter_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Improved preprocessing with better text handling."""
        try:
            document_text = state.get("document_text", "")
            
            if not document_text or len(document_text.strip()) < 100:
                return {
                    "preprocessed_text": "",
                    "processing_errors": ["Document text is too short or empty for meaningful analysis"],
                    "expected_agents": [],
                    "completed_agents": []
                }
            
            # Smart text truncation for preprocessing
            max_length = 15000  # Increased limit for better context
            if len(document_text) > max_length:
                # Keep beginning and end of document for context
                half_length = max_length // 2
                preprocessed = (
                    document_text[:half_length] + 
                    "\n\n[... MIDDLE SECTION TRUNCATED FOR PROCESSING ...]\n\n" + 
                    document_text[-half_length:]
                )
            else:
                preprocessed = document_text
            
            # Use faster model for preprocessing
            chain = ROUGHTER_PROMPT | self.flash_llm
            result = chain.invoke({"text": preprocessed})
            
            processed_text = result.content if hasattr(result, 'content') else str(result)
            
            if not processed_text or len(processed_text.strip()) < 50:
                # Fallback: use original text if preprocessing fails
                processed_text = document_text
            
            return {
                "preprocessed_text": processed_text,
                "expected_agents": ["summarizer", "risk_analyzer", "highlighter", "confidence"],
                "completed_agents": [],
                "document_metadata": {
                    "original_length": len(document_text),
                    "processed_length": len(processed_text),
                    "truncated": len(document_text) > max_length
                }
            }
            
        except Exception as e:
            logger.error(f"Roughter agent error: {e}")
            # Graceful fallback
            return {
                "preprocessed_text": state.get("document_text", ""),
                "expected_agents": ["summarizer", "risk_analyzer", "highlighter", "confidence"],
                "completed_agents": [],
                "processing_errors": [f"Preprocessing warning: {str(e)}"]
            }

    def master_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Master agent that coordinates parallel execution of sub-agents."""
        try:
            start_time = time.time()
            preprocessed_text = state.get("preprocessed_text", "")
            
            if not preprocessed_text or len(preprocessed_text.strip()) < 50:
                return {
                    "summary_result": {},
                    "risk_result": {},
                    "highlights_result": {},
                    "confidence_result": {},
                    "completed_agents": [],
                    "processing_errors": ["Insufficient text for analysis"]
                }
            
            # Prepare text chunks for different agents (optimized for their needs)
            summary_text = self._prepare_text_for_agent(preprocessed_text, "summary", 12000)
            risk_text = self._prepare_text_for_agent(preprocessed_text, "risk", 12000)
            highlights_text = self._prepare_text_for_agent(preprocessed_text, "highlights", 10000)
            confidence_text = self._prepare_text_for_agent(preprocessed_text, "confidence", 8000)
            
            # Execute all agents in parallel using ThreadPoolExecutor
            results = {}
            errors = []
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                future_to_agent = {
                    executor.submit(self._execute_summarizer, summary_text): "summarizer",
                    executor.submit(self._execute_risk_analyzer, risk_text): "risk_analyzer", 
                    executor.submit(self._execute_highlighter, highlights_text): "highlighter",
                    executor.submit(self._execute_confidence, confidence_text): "confidence"
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_agent):
                    agent_name = future_to_agent[future]
                    try:
                        result = future.result(timeout=15)  # 15 second timeout per agent
                        results[agent_name] = result
                        logger.info(f"✅ {agent_name} completed successfully")
                    except Exception as e:
                        logger.error(f"❌ {agent_name} failed: {e}")
                        errors.append(f"{agent_name}: {str(e)}")
                        results[agent_name] = self._get_fallback_result(agent_name)
            
            parallel_time = time.time() - start_time
            logger.info(f"🚀 Parallel execution completed in {parallel_time:.2f}s")
            
            return {
                "summary_result": results.get("summarizer", {}),
                "risk_result": results.get("risk_analyzer", {}),
                "highlights_result": results.get("highlighter", {}),
                "confidence_result": results.get("confidence", {}),
                "completed_agents": list(results.keys()),
                "processing_errors": errors,
                "execution_time": parallel_time
            }
            
        except Exception as e:
            logger.error(f"Master agent error: {e}")
            return {
                "summary_result": self._get_fallback_result("summarizer"),
                "risk_result": self._get_fallback_result("risk_analyzer"),
                "highlights_result": self._get_fallback_result("highlighter"),
                "confidence_result": self._get_fallback_result("confidence"),
                "completed_agents": [],
                "processing_errors": [f"Master agent error: {str(e)}"]
            }

    def _prepare_text_for_agent(self, text: str, agent_type: str, max_length: int) -> str:
        """Prepare optimized text chunks for different agent types."""
        if len(text) <= max_length:
            return text
        
        if agent_type == "summary":
            # For summary, prioritize beginning and structure
            return text[:max_length * 3//4] + "\n\n[...truncated...]\n\n" + text[-max_length//4:]
        elif agent_type == "risk":
            # For risk, focus on terms, conditions, and penalties
            # Try to find risk-related sections
            return self._extract_risk_sections(text, max_length)
        elif agent_type == "highlights":
            # For highlights, focus on dates, numbers, and key terms
            return self._extract_highlight_sections(text, max_length)
        else:
            # Default: balanced beginning and end
            half = max_length // 2
            return text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]
    
    def _extract_risk_sections(self, text: str, max_length: int) -> str:
        """Extract sections likely to contain risk-related information."""
        risk_keywords = ['penalty', 'terminate', 'breach', 'default', 'liability', 'damages', 'forfeit', 'void']
        lines = text.split('\n')
        risk_lines = []
        other_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in risk_keywords):
                risk_lines.append(line)
            else:
                other_lines.append(line)
        
        # Prioritize risk lines
        risk_text = '\n'.join(risk_lines)
        if len(risk_text) < max_length:
            remaining = max_length - len(risk_text)
            other_text = '\n'.join(other_lines)[:remaining]
            return risk_text + '\n' + other_text
        
        return risk_text[:max_length]
    
    def _extract_highlight_sections(self, text: str, max_length: int) -> str:
        """Extract sections likely to contain important highlights."""
        import re
        
        # Find lines with dates, amounts, percentages
        highlight_patterns = [
            r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',  # dates
            r'\$[\d,]+\.?\d*',  # money amounts
            r'\d+\%',  # percentages
            r'\d+ days?',  # day periods
            r'\d+ months?',  # month periods
        ]
        
        lines = text.split('\n')
        highlight_lines = []
        other_lines = []
        
        for line in lines:
            if any(re.search(pattern, line) for pattern in highlight_patterns):
                highlight_lines.append(line)
            else:
                other_lines.append(line)
        
        # Prioritize highlight lines
        highlight_text = '\n'.join(highlight_lines)
        if len(highlight_text) < max_length:
            remaining = max_length - len(highlight_text)
            other_text = '\n'.join(other_lines)[:remaining]
            return highlight_text + '\n' + other_text
        
        return highlight_text[:max_length]

    def _execute_summarizer(self, text: str) -> Dict[str, Any]:
        """Execute summarizer agent."""
        response = self._safe_invoke_agent(self.summarizer_chain, text, "Summarizer")
        if "error" in response:
            return self._get_fallback_result("summarizer")
        return response["result"].dict() if hasattr(response["result"], "dict") else response["result"]

    def _execute_risk_analyzer(self, text: str) -> Dict[str, Any]:
        """Execute risk analyzer agent."""
        response = self._safe_invoke_agent(self.risk_chain, text, "Risk Analyzer")
        if "error" in response:
            return self._get_fallback_result("risk_analyzer")
        return response["result"].dict() if hasattr(response["result"], "dict") else response["result"]

    def _execute_highlighter(self, text: str) -> Dict[str, Any]:
        """Execute highlighter agent."""
        response = self._safe_invoke_agent(self.highlighter_chain, text, "Highlighter")
        if "error" in response:
            return self._get_fallback_result("highlighter")
        return response["result"].dict() if hasattr(response["result"], "dict") else response["result"]

    def _execute_confidence(self, text: str) -> Dict[str, Any]:
        """Execute confidence agent."""
        response = self._safe_invoke_agent(self.confidence_chain, text, "Confidence")
        if "error" in response:
            return self._get_fallback_result("confidence")
        return response["result"].dict() if hasattr(response["result"], "dict") else response["result"]

    def _get_fallback_result(self, agent_name: str) -> Dict[str, Any]:
        """Get fallback results for failed agents."""
        fallbacks = {
            "summarizer": {
                "document_type": "other",
                "overview": "Document analysis incomplete due to processing limitations.",
                "key_parties": ["Party 1", "Party 2"],
                "main_purpose": "Unable to determine from available text",
                "obligations_summary": {"general": ["Analysis incomplete"]},
                "duration_and_dates": {"note": "Unable to extract dates"},
                "termination_conditions": ["Unable to determine termination conditions"]
            },
            "risk_analyzer": {
                "overall_risk_level": "medium",
                "risk_score": 5,
                "critical_risks": ["Unable to complete full risk analysis"],
                "moderate_risks": ["Document processing incomplete"],
                "red_flags": [],
                "penalty_clauses": [],
                "liability_concerns": [],
                "recommendation": "Legal review recommended due to incomplete analysis"
            },
            "highlighter": {
                "critical_deadlines": [{"note": "Unable to extract deadlines"}],
                "financial_obligations": [{"note": "Unable to extract financial details"}],
                "auto_renewal_clause": None,
                "termination_rights": ["Unable to determine termination rights"],
                "key_restrictions": ["Analysis incomplete"],
                "action_items": ["Complete document re-analysis recommended"],
                "negotiable_terms": ["Unable to identify negotiable terms"]
            },
            "confidence": {
                "overall_confidence": 30.0,
                "document_clarity": 50.0,
                "well_understood_sections": ["Limited sections analyzed"],
                "unclear_sections": ["Most sections require manual review"],
                "missing_information": ["Complete document analysis"],
                "legal_consultation_recommended": True,
                "consultation_reasons": ["Analysis incomplete due to processing limitations"]
            }
        }
        return fallbacks.get(agent_name, {})

    def _safe_invoke_agent(self, chain, text: str, agent_name: str) -> Dict[str, Any]:
        """Safely invoke an agent with proper error handling."""
        try:
            if not text or len(text.strip()) < 50:
                return {
                    "error": f"{agent_name} cannot analyze: insufficient text",
                    "fallback_used": True
                }
            
            # Limit text length per agent to prevent token overflow
            max_text = 12000
            if len(text) > max_text:
                # Use beginning and end for context
                half = max_text // 2
                analysis_text = text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]
            else:
                analysis_text = text
            
            result = chain.invoke({"text": analysis_text})
            
            # Handle both raw and parsed responses
            if hasattr(result, 'parsed') and result.parsed:
                return {"result": result.parsed, "raw_response": result.raw}
            elif hasattr(result, 'dict'):
                return {"result": result}
            else:
                return {"result": result}
                
        except Exception as e:
            logger.error(f"{agent_name} error: {e}")
            return {
                "error": str(e),
                "fallback_used": True
            }

    def coordinator_agent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final coordinated output with improved performance metrics."""
        try:
            # Collect all results with safe defaults
            summary = state.get("summary_result", {})
            risk = state.get("risk_result", {})
            highlights = state.get("highlights_result", {})
            confidence = state.get("confidence_result", {})
            errors = state.get("processing_errors", [])
            execution_time = state.get("execution_time", 0)
            
            # Check if we have meaningful results
            has_meaningful_data = (
                bool(summary.get("overview")) and 
                len(summary.get("overview", "")) > 50 and
                "analysis incomplete" not in summary.get("overview", "").lower()
            )
            
            if not has_meaningful_data:
                # Generate error report
                final_output = self._generate_error_report(errors, state)
            else:
                # Generate normal report with performance info
                chain = COORDINATOR_PROMPT | self.pro_llm
                result = chain.invoke({
                    "summary": json.dumps(summary, indent=2),
                    "risk_analysis": json.dumps(risk, indent=2),
                    "highlights": json.dumps(highlights, indent=2),
                    "confidence": json.dumps(confidence, indent=2),
                    "execution_time": execution_time
                })
                final_output = result.content
            
            return {
                "final_output": final_output,
                "execution_metrics": {
                    "parallel_execution_time": execution_time,
                    "agents_completed": len(state.get("completed_agents", [])),
                    "errors_count": len(errors)
                }
            }
            
        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            return {
                "final_output": self._generate_error_report([str(e)], state),
                "processing_errors": [f"Coordination error: {str(e)}"]
            }

    def _generate_error_report(self, errors: List[str], state: Dict[str, Any]) -> str:
        """Generate a user-friendly error report."""
        doc_length = state.get("document_metadata", {}).get("original_length", 0)
        
        return f"""# Document Analysis Report - Processing Issues Encountered

## Summary
We encountered technical difficulties while analyzing your legal document. While we detected a document of {doc_length:,} characters, our analysis agents were unable to complete a full review.

## What We Know
- **Document Detected**: Yes, we received a document with {doc_length:,} characters
- **Document Type**: Appears to be a legal agreement or contract
- **Processing Status**: Partially completed with technical issues

## Issues Encountered
{chr(10).join(f"• {error}" for error in errors[:5])}

## Recommended Next Steps

### Immediate Actions:
1. **Try Re-uploading**: Sometimes re-uploading the document resolves processing issues
2. **Check File Quality**: Ensure the PDF is not corrupted or password-protected
3. **Manual Review**: Consider having a legal professional review the document directly

### What You Should Do:
- **Don't ignore this document** - it appears to be a substantial legal agreement
- **Seek legal counsel** if this is an important contract you're considering signing
- **Contact our support** if processing issues persist

## Important Note
This processing failure doesn't mean your document is unimportant or risk-free. Legal documents of this size typically contain significant terms that require careful review.

**When in doubt, consult with a qualified attorney before signing any legal document.**
"""

# =============================================================================
# IMPROVED USER INTERFACE RESPONSE FORMAT
# =============================================================================

def format_analysis_response(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Format the analysis result for better user experience."""
    
    # Extract components
    summary = raw_result.get("summary_result", {})
    risk = raw_result.get("risk_result", {})
    highlights = raw_result.get("highlights_result", {})
    confidence = raw_result.get("confidence_result", {})
    final_output = raw_result.get("final_output", "")
    errors = raw_result.get("processing_errors", [])
    
    # Determine status
    if errors or not summary.get("overview"):
        status = "partial_analysis"
    elif confidence.get("overall_confidence", 0) < 50:
        status = "low_confidence"
    else:
        status = "success"
    
    # Create user-friendly response
    formatted_response = {
        "status": status,
        "analysis": final_output,
        "executive_summary": {
            "document_type": summary.get("document_type", "unknown"),
            "risk_level": risk.get("overall_risk_level", "unknown"),
            "confidence_score": f"{confidence.get('overall_confidence', 0):.0f}%",
            "legal_review_recommended": confidence.get("legal_consultation_recommended", True),
            "key_concerns": risk.get("critical_risks", [])[:3]  # Top 3 concerns
        },
        "key_insights": {
            "main_parties": summary.get("key_parties", []),
            "primary_purpose": summary.get("main_purpose", "Not determined"),
            "critical_deadlines": [
                deadline for deadline in highlights.get("critical_deadlines", [])
                if isinstance(deadline, dict)
            ][:3],
            "financial_obligations": [
                obligation for obligation in highlights.get("financial_obligations", [])
                if isinstance(obligation, dict)
            ][:3]
        },
        "warnings": {
            "high_risk_items": risk.get("critical_risks", []),
            "red_flags": risk.get("red_flags", []),
            "unclear_sections": confidence.get("unclear_sections", [])
        },
        "next_steps": _generate_next_steps(risk, confidence, errors),
        "metadata": {
            "document_length": raw_result.get("document_metadata", {}).get("original_length", 0),
            "processing_errors": errors,
            "analysis_timestamp": "2024-01-01T00:00:00Z",  # Add actual timestamp
            "confidence_breakdown": {
                "document_clarity": f"{confidence.get('document_clarity', 0):.0f}%",
                "analysis_completeness": "Partial" if errors else "Complete"
            }
        }
    }
    
    return formatted_response

def _generate_next_steps(risk: Dict, confidence: Dict, errors: List[str]) -> List[str]:
    """Generate contextual next steps for the user."""
    steps = []
    
    if errors:
        steps.append("Re-upload the document if analysis was incomplete")
    
    if confidence.get("legal_consultation_recommended", True):
        steps.append("Consult with a qualified attorney before signing")
    
    if risk.get("overall_risk_level") == "high":
        steps.append("Carefully review all highlighted risk factors")
        steps.append("Consider negotiating problematic clauses")
    
    critical_risks = risk.get("critical_risks", [])
    if critical_risks:
        steps.append(f"Pay special attention to: {critical_risks[0]}")
    
    unclear_sections = confidence.get("unclear_sections", [])
    if unclear_sections:
        steps.append("Request clarification on unclear sections")
    
    if not steps:
        steps = [
            "Review the full analysis carefully",
            "Consider legal consultation if signing important documents",
            "Keep a copy of this analysis for your records"
        ]
    
    return steps[:5]

# =============================================================================
# LANGGRAPH WORKFLOW SETUP
# =============================================================================

def create_workflow():
    """Create and configure the LangGraph workflow with parallel execution."""
    analyzer = ImprovedLegalAnalyzer()
    
    # Create the workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes - now using master-sub architecture
    workflow.add_node("roughter", analyzer.roughter_agent)
    workflow.add_node("master_agent", analyzer.master_agent)  # Master coordinates parallel sub-agents
    workflow.add_node("coordinator", analyzer.coordinator_agent)
    
    # Define the streamlined workflow edges (parallel execution via master)
    workflow.add_edge(START, "roughter")
    workflow.add_edge("roughter", "master_agent")  # Master handles all sub-agents in parallel
    workflow.add_edge("master_agent", "coordinator")
    workflow.add_edge("coordinator", END)
    
    # Compile the workflow
    app = workflow.compile()
    
    logger.info("🚀 Created optimized workflow with parallel master-sub architecture")
    return app

async def process_document_workflow(document_text: str) -> Dict[str, Any]:
    """Process a document through the complete workflow."""
    try:
        # Create the workflow
        workflow_app = create_workflow()
        
        # Initialize state
        initial_state = {
            "document_text": document_text,
            "preprocessed_text": "",
            "summary_result": {},
            "risk_result": {},
            "highlights_result": {},
            "confidence_result": {},
            "final_output": "",
            "completed_agents": [],
            "expected_agents": [],
            "processing_errors": [],
            "document_metadata": {}
        }
        
        # Run the workflow
        result = await workflow_app.ainvoke(initial_state)
        
        # Format the response for the user
        formatted_result = format_analysis_response(result)
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"Workflow processing error: {e}")
        return {
            "status": "error",
            "analysis": f"Processing failed: {str(e)}",
            "executive_summary": {
                "document_type": "unknown",
                "risk_level": "unknown",
                "confidence_score": "0%",
                "legal_review_recommended": True,
                "key_concerns": ["Processing error occurred"]
            },
            "error": str(e)
        } 