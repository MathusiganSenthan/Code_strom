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
    """Enhanced document summary with document type detection and markdown support."""
    document_type: str = Field(description="Specific type of legal document")
    overview: str = Field(description="Professional executive summary with markdown formatting")
    main_parties: List[str] = Field(description="Names/roles of key parties involved")
    key_obligations: List[str] = Field(description="Key obligations for all parties")
    important_dates: List[str] = Field(description="Critical dates and deadlines")
    termination_conditions: List[str] = Field(description="How the agreement can be terminated")
    metrics: Dict[str, int] = Field(description="Quantitative metrics for the document", default_factory=dict)
    positive_aspects: List[str] = Field(description="Favorable terms and benefits")
    areas_of_concern: List[Dict[str, str]] = Field(description="Concerning terms with risk levels")
    complexity_assessment: Dict[str, Any] = Field(description="Document complexity analysis", default_factory=dict)

class RiskItem(BaseModel):
    """Individual risk item with structured data."""
    id: int = Field(description="Unique identifier for the risk")
    title: str = Field(description="Risk title in plain English")
    type: str = Field(description="Risk category: FINANCIAL, LEGAL, OPERATIONAL, COMPLIANCE")
    severity: str = Field(description="Risk severity: HIGH SEVERITY, MEDIUM SEVERITY, LOW SEVERITY")
    section: str = Field(description="Document section where risk is found")
    description: str = Field(description="Clear explanation of the risk")
    impact: str = Field(description="Real-world consequences and potential costs")
    recommendation: str = Field(description="Specific action the user should take")
    confidence: int = Field(description="Confidence level in this assessment (0-100)", ge=0, le=100)

class RiskAssessment(BaseModel):
    """Comprehensive risk analysis with structured data for frontend display."""
    overall_risk_level: RiskLevel = Field(description="Overall risk classification")
    risk_score: int = Field(description="Risk score from 1-10", ge=1, le=10)
    critical_risks: List[RiskItem] = Field(description="High-priority risks requiring immediate attention")
    moderate_risks: List[RiskItem] = Field(description="Medium-priority risks to monitor")
    red_flags: List[str] = Field(description="Warning signs and concerning clauses")
    financial_penalties: List[str] = Field(description="Potential financial costs and penalties")
    liability_concerns: List[str] = Field(description="Liability and indemnification issues")
    analysis: str = Field(description="Detailed risk analysis in markdown format - clean analysis only, no system prompts")

class DeadlineItem(BaseModel):
    """Individual deadline with structured data."""
    id: int = Field(description="Unique identifier for the deadline")
    title: str = Field(description="Deadline title in plain English")
    description: str = Field(description="Clear explanation of what needs to be done")
    dueDate: str = Field(description="When this deadline occurs")
    party: str = Field(description="Who is responsible for this deadline")
    priority: str = Field(description="Priority level: HIGH, MEDIUM, LOW")
    category: str = Field(description="Deadline category: Legal, Financial, Operational, Compliance")

class FinancialObligationItem(BaseModel):
    """Individual financial obligation with structured data."""
    id: int = Field(description="Unique identifier for the obligation")
    title: str = Field(description="Obligation title in plain English")
    description: str = Field(description="Clear explanation of the financial requirement")
    amount: str = Field(description="Amount or cost involved")
    due_date: str = Field(description="When payment is due")
    party: str = Field(description="Who is responsible for payment")
    priority: str = Field(description="Priority level: HIGH, MEDIUM, LOW")
    category: str = Field(description="Obligation category: Payment, Fee, Penalty, Insurance")

class AutoRenewalClause(BaseModel):
    """Auto-renewal clause details."""
    exists: bool = Field(description="Whether auto-renewal clause exists")
    renewal_period: str = Field(description="How long each renewal period lasts")
    notice_required: str = Field(description="Notice period required to prevent renewal")
    automatic: bool = Field(description="Whether renewal happens automatically")

class KeyHighlights(BaseModel):
    """Important document highlights with structured data for frontend display."""
    critical_deadlines: List[DeadlineItem] = Field(description="Important dates and deadlines")
    financial_obligations: List[FinancialObligationItem] = Field(description="Financial commitments and amounts")
    auto_renewal_clause: AutoRenewalClause = Field(description="Auto-renewal details if present")
    termination_procedures: List[str] = Field(description="Steps to terminate the agreement")
    key_restrictions: List[str] = Field(description="Important restrictions or limitations")
    action_items: List[str] = Field(description="Required actions for the user")
    analysis: str = Field(description="Detailed obligations analysis in markdown format - clean analysis only, no system prompts")

class ConfidenceAssessment(BaseModel):
    """Enhanced analysis confidence and clarity metrics."""
    overall_confidence: float = Field(description="Confidence in analysis (0-100)", ge=0, le=100)
    clarity_score: float = Field(description="How clear the document language is (0-100)", ge=0, le=100)
    completeness: float = Field(description="Completeness of the analysis (0-100)", ge=0, le=100)
    legal_complexity: str = Field(description="Complexity level: simple, moderate, complex, very_complex")
    well_understood_sections: List[str] = Field(description="Clearly analyzed sections")
    complex_sections: List[str] = Field(description="Sections requiring expert knowledge")
    unclear_sections: List[str] = Field(description="Sections needing expert review")
    missing_information: List[str] = Field(description="Important info that appears missing")
    recommendations: List[str] = Field(description="Professional recommendations and next steps")
    legal_consultation_recommended: bool = Field(description="Whether lawyer consultation is advised")
    consultation_urgency: str = Field(description="Urgency level: low, medium, high, critical")
    consultation_reasons: List[str] = Field(description="Reasons for legal consultation if recommended")
    quality_metrics: Dict[str, Any] = Field(description="Analysis quality scores", default_factory=dict)
    analysis: str = Field(description="Detailed confidence analysis explanation")

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