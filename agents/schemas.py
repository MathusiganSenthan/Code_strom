# =============================================================================
# agents/schemas.py - Optimized Pydantic Models
# =============================================================================

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class DocumentType(str, Enum):
    CONTRACT = "contract"
    LEASE = "lease"
    TERMS = "terms_conditions"
    AGREEMENT = "agreement"
    OTHER = "other"

class Summary(BaseModel):
    """Enhanced document summary with document type detection."""
    document_type: DocumentType = Field(description="Type of legal document")
    overview: str = Field(description="Clear, concise overview in plain language")
    key_parties: List[str] = Field(description="Names/roles of key parties involved")
    main_purpose: str = Field(description="Primary purpose of the document")
    obligations_summary: str = Field(description="Key obligations summary in plain language")
    duration_and_dates: str = Field(description="Important dates and durations in plain language")
    termination_conditions: List[str] = Field(description="How the agreement can be terminated")

class RiskAssessment(BaseModel):
    """Comprehensive risk analysis."""
    overall_risk_level: RiskLevel = Field(description="Overall risk classification")
    risk_score: int = Field(description="Risk score from 1-10", ge=1, le=10)
    critical_risks: List[str] = Field(description="High-priority risks requiring attention")
    moderate_risks: List[str] = Field(description="Medium-priority risks to monitor")
    red_flags: List[str] = Field(description="Unusual or concerning clauses")
    penalty_clauses: List[str] = Field(description="Financial penalties and consequences")
    liability_concerns: List[str] = Field(description="Liability and indemnification issues")
    recommendation: str = Field(description="Overall risk recommendation")

class KeyHighlights(BaseModel):
    """Important document highlights."""
    critical_deadlines: List[str] = Field(description="Important dates and deadlines")
    financial_obligations: List[str] = Field(description="Financial commitments and amounts")
    auto_renewal_clause: str = Field(description="Auto-renewal details if present, or 'None' if not applicable")
    termination_rights: List[str] = Field(description="Rights to terminate the agreement")
    key_restrictions: List[str] = Field(description="Important restrictions or limitations")
    action_items: List[str] = Field(description="Required actions for the user")
    negotiable_terms: List[str] = Field(description="Terms that might be negotiable")

class ConfidenceAssessment(BaseModel):
    """Analysis confidence and clarity metrics."""
    overall_confidence: float = Field(description="Confidence in analysis (0-100)", ge=0, le=100)
    document_clarity: float = Field(description="How clear the document language is (0-100)", ge=0, le=100)
    well_understood_sections: List[str] = Field(description="Clearly analyzed sections")
    unclear_sections: List[str] = Field(description="Sections needing expert review")
    missing_information: List[str] = Field(description="Important info that appears missing")
    legal_consultation_recommended: bool = Field(description="Whether lawyer consultation is advised")
    consultation_reasons: List[str] = Field(description="Reasons for legal consultation if recommended")

class WorkflowState(BaseModel):
    """Enhanced state management for the workflow."""
    document_text: str
    preprocessed_text: Optional[str] = None
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    expected_agents: List[str] = Field(default_factory=list)
    completed_agents: List[str] = Field(default_factory=list)
    summary_result: Optional[Dict[str, Any]] = None
    risk_result: Optional[Dict[str, Any]] = None
    highlights_result: Optional[Dict[str, Any]] = None
    confidence_result: Optional[Dict[str, Any]] = None
    final_output: Optional[str] = None
    processing_errors: List[str] = Field(default_factory=list)