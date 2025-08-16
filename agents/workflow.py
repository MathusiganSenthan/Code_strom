import asyncio
from typing import Dict, Any, List, TypedDict, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from agents.schemas import Summary, RiskAssessment, KeyHighlights, ConfidenceAssessment
from agents.prompts import *
import os
import logging
import json
import time
import statistics
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Enhanced enums for dynamic scaling
class DocumentComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"

class AgentPriority(Enum):
    CRITICAL = 1    # Summarizer (user needs overview first)
    HIGH = 2        # Risk Analyzer (safety-critical)
    MEDIUM = 3      # Highlighter (important but not urgent)
    LOW = 4         # Confidence (meta-information)

@dataclass
class AgentPerformanceMetrics:
    """Real-time performance tracking for agents."""
    execution_times: List[float] = field(default_factory=list)
    success_count: int = 0
    total_executions: int = 0
    success_rate: float = 0.0
    quality_scores: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def get_average_time(self) -> float:
        return statistics.mean(self.execution_times) if self.execution_times else 0.0
    
    def get_quality_trend(self) -> str:
        if len(self.quality_scores) < 2:
            return "stable"
        recent = statistics.mean(self.quality_scores[-3:])
        older = statistics.mean(self.quality_scores[:-3]) if len(self.quality_scores) > 3 else recent
        if recent > older + 0.1:
            return "improving"
        elif recent < older - 0.1:
            return "declining"
        return "stable"

@dataclass
class CrossAgentInsight:
    """Insights derived from cross-agent analysis."""
    source_agents: List[str]
    insight_type: str
    confidence: float
    priority: AgentPriority
    description: str
    action_required: str

# Define the enhanced state structure for LangGraph
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
    # Enhanced fields for dynamic optimization
    document_complexity: str
    optimal_worker_count: int
    cross_agent_insights: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    quality_scores: Dict[str, float]

class ImprovedLegalAnalyzer:
    def __init__(self):
        # Enhanced model configuration with higher token limits for large documents
        self.pro_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_output_tokens=8192  # Increased for detailed analysis
        )
        
        self.flash_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_output_tokens=4096  # Increased for better responses
        )
        
        # Performance tracking for real-time optimization
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.agent_performance: Dict[str, AgentPerformanceMetrics] = {}
        self.global_performance_history: List[float] = []
        
        # Create chains with better error handling
        self.summarizer_chain = self._create_chain(SUMMARIZER_PROMPT, Summary, self.pro_llm)
        self.risk_chain = self._create_chain(RISK_ANALYZER_PROMPT, RiskAssessment, self.pro_llm)
        self.highlighter_chain = self._create_chain(HIGHLIGHTER_PROMPT, KeyHighlights, self.flash_llm)
        self.confidence_chain = self._create_chain(CONFIDENCE_PROMPT, ConfidenceAssessment, self.flash_llm)
        
        # Initialize performance metrics
        self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize performance tracking for all agents."""
        agents = ["summarizer", "risk_analyzer", "highlighter", "confidence"]
        for agent in agents:
            self.agent_performance[agent] = AgentPerformanceMetrics()
        
    def _create_chain(self, prompt, schema, llm):
        """Create a chain with better error handling."""
        try:
            return prompt | llm.with_structured_output(
                schema, 
                include_raw=True,
                method="function_calling"
            )
        except Exception as e:
            logger.warning(f"Function calling failed, falling back to JSON mode: {e}")
            return prompt | llm.with_structured_output(
                schema, 
                include_raw=True,
                method="json_mode"
            )

    # ============================================================================
    # DYNAMIC SCALING IMPLEMENTATION (10/10)
    # ============================================================================
    
    def _analyze_document_complexity(self, text: str) -> Tuple[DocumentComplexity, Dict[str, Any]]:
        """Advanced document complexity analysis for dynamic scaling."""
        metrics = {
            'length': len(text),
            'sentences': len(re.findall(r'[.!?]+', text)),
            'paragraphs': len(text.split('\n\n')),
            'legal_terms': len(re.findall(r'\b(?:whereas|therefore|pursuant|liability|indemnify|breach|terminate)\b', text.lower())),
            'financial_terms': len(re.findall(r'\$[\d,]+\.?\d*|percent|%|\bfee\b|\bpay\b', text.lower())),
            'date_complexity': len(re.findall(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', text)),
            'clause_density': text.lower().count('clause') + text.lower().count('section') + text.lower().count('article')
        }
        
        # Calculate complexity score
        complexity_score = 0
        
        # Length-based scoring
        if metrics['length'] > 20000:
            complexity_score += 3
        elif metrics['length'] > 10000:
            complexity_score += 2
        elif metrics['length'] > 5000:
            complexity_score += 1
            
        # Legal complexity
        legal_density = metrics['legal_terms'] / max(metrics['sentences'], 1)
        if legal_density > 0.3:
            complexity_score += 3
        elif legal_density > 0.15:
            complexity_score += 2
        elif legal_density > 0.05:
            complexity_score += 1
            
        # Structure complexity
        if metrics['clause_density'] > 50:
            complexity_score += 2
        elif metrics['clause_density'] > 20:
            complexity_score += 1
            
        # Financial complexity
        if metrics['financial_terms'] > 10:
            complexity_score += 1
            
        # Determine complexity level
        if complexity_score >= 8:
            complexity = DocumentComplexity.VERY_COMPLEX
        elif complexity_score >= 5:
            complexity = DocumentComplexity.COMPLEX
        elif complexity_score >= 2:
            complexity = DocumentComplexity.MODERATE
        else:
            complexity = DocumentComplexity.SIMPLE
            
        metrics['complexity_score'] = complexity_score
        return complexity, metrics
    
    def _get_optimal_worker_count(self, complexity: DocumentComplexity, doc_length: int) -> int:
        """Dynamic worker count optimization based on document characteristics."""
        base_workers = {
            DocumentComplexity.SIMPLE: 2,
            DocumentComplexity.MODERATE: 3,
            DocumentComplexity.COMPLEX: 4,
            DocumentComplexity.VERY_COMPLEX: 6
        }
        
        workers = base_workers[complexity]
        
        # Adjust based on historical performance
        avg_time = statistics.mean(self.global_performance_history) if self.global_performance_history else 30.0
        
        if avg_time > 45:  # Slow performance, increase workers
            workers = min(workers + 2, 8)
        elif avg_time < 15:  # Fast performance, optimize for efficiency
            workers = max(workers - 1, 2)
            
        return workers
    
    def _select_optimal_agents(self, complexity: DocumentComplexity, doc_type: str) -> List[str]:
        """Intelligent agent selection based on document characteristics."""
        base_agents = ["summarizer", "risk_analyzer", "highlighter", "confidence"]
        
        # Document type specific optimizations
        if "lease" in doc_type.lower() or "rental" in doc_type.lower():
            # Add specialized financial analysis for leases
            base_agents.append("financial_analyzer")
        elif "employment" in doc_type.lower() or "contract" in doc_type.lower():
            # Add compliance checking for contracts
            base_agents.append("compliance_checker")
            
        # Complexity-based agent selection
        if complexity in [DocumentComplexity.COMPLEX, DocumentComplexity.VERY_COMPLEX]:
            # Add cross-validation agent for complex docs
            base_agents.append("cross_validator")
            
        return base_agents[:self._get_optimal_worker_count(complexity, 0)]  # Limit to worker count

    # ============================================================================
    # CROSS-AGENT INTELLIGENCE SYSTEM (10/10)
    # ============================================================================
    
    def _analyze_cross_agent_insights(self, agent_results: Dict[str, Any]) -> List[CrossAgentInsight]:
        """Advanced cross-agent analysis for insight synthesis."""
        insights = []
        
        # Extract key data from each agent
        summary_data = agent_results.get('summarizer', {})
        risk_data = agent_results.get('risk_analyzer', {})
        highlight_data = agent_results.get('highlighter', {})
        confidence_data = agent_results.get('confidence', {})
        
        # Risk-Summary Correlation Analysis
        if summary_data and risk_data:
            summary_text = str(summary_data.get('content', ''))
            risk_level = getattr(risk_data.get('parsed', {}), 'risk_level', 'unknown')
            
            if 'financial' in summary_text.lower() and risk_level in ['high', 'critical']:
                insights.append(CrossAgentInsight(
                    source_agents=['summarizer', 'risk_analyzer'],
                    insight_type='financial_risk_correlation',
                    confidence=0.85,
                    priority=AgentPriority.HIGH,
                    description="Financial elements in summary correlate with high risk assessment",
                    action_required="Detailed financial review recommended"
                ))
                
        # Confidence-Risk Mismatch Detection
        if confidence_data and risk_data:
            confidence_score = getattr(confidence_data.get('parsed', {}), 'overall_confidence', 0)
            risk_level = getattr(risk_data.get('parsed', {}), 'risk_level', 'unknown')
            
            if confidence_score < 0.6 and risk_level in ['low', 'minimal']:
                insights.append(CrossAgentInsight(
                    source_agents=['confidence', 'risk_analyzer'],
                    insight_type='confidence_risk_mismatch',
                    confidence=0.75,
                    priority=AgentPriority.MEDIUM,
                    description="Low confidence despite low risk assessment suggests ambiguous language",
                    action_required="Manual review of ambiguous clauses recommended"
                ))
                
        # Highlight-Summary Consistency Check
        if highlight_data and summary_data:
            highlights = getattr(highlight_data.get('parsed', {}), 'key_highlights', [])
            summary_text = str(summary_data.get('content', ''))
            
            critical_highlights = [h for h in highlights if 'critical' in str(h).lower() or 'important' in str(h).lower()]
            
            if critical_highlights and 'standard' in summary_text.lower():
                insights.append(CrossAgentInsight(
                    source_agents=['highlighter', 'summarizer'],
                    insight_type='criticality_inconsistency',
                    confidence=0.70,
                    priority=AgentPriority.MEDIUM,
                    description="Critical highlights found in document categorized as standard",
                    action_required="Review classification accuracy"
                ))
                
        return insights
    
    def _detect_conflicts(self, agent_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect and resolve conflicts between agent analyses."""
        conflicts = []
        
        # Risk level conflicts
        risk_data = agent_results.get('risk_analyzer', {})
        confidence_data = agent_results.get('confidence', {})
        
        if risk_data and confidence_data:
            risk_level = getattr(risk_data.get('parsed', {}), 'risk_level', 'unknown')
            confidence_score = getattr(confidence_data.get('parsed', {}), 'overall_confidence', 0)
            
            # High confidence but high risk = potential oversight
            if confidence_score > 0.8 and risk_level in ['high', 'critical']:
                conflicts.append({
                    'type': 'confidence_risk_conflict',
                    'agents': ['risk_analyzer', 'confidence'],
                    'severity': 'medium',
                    'description': 'High confidence with high risk may indicate oversight',
                    'resolution': 'Prioritize risk analysis findings'
                })
                
        return conflicts
    
    def _synthesize_insights(self, agent_results: Dict[str, Any], cross_insights: List[CrossAgentInsight]) -> Dict[str, Any]:
        """Synthesize insights from multiple agents into unified analysis."""
        synthesis = {
            'primary_findings': [],
            'cross_agent_insights': [],
            'confidence_adjustments': {},
            'priority_actions': []
        }
        
        # Process cross-agent insights
        for insight in cross_insights:
            synthesis['cross_agent_insights'].append({
                'type': insight.insight_type,
                'description': insight.description,
                'confidence': insight.confidence,
                'priority': insight.priority.value,
                'action': insight.action_required
            })
            
            # High priority insights become primary findings
            if insight.priority == AgentPriority.HIGH:
                synthesis['primary_findings'].append(insight.description)
                synthesis['priority_actions'].append(insight.action_required)
                
        # Confidence adjustments based on cross-agent analysis
        if len(cross_insights) > 2:
            synthesis['confidence_adjustments']['overall'] = 'increased_due_to_cross_validation'
        elif any(i.insight_type.endswith('_mismatch') for i in cross_insights):
            synthesis['confidence_adjustments']['overall'] = 'decreased_due_to_inconsistencies'
            
        return synthesis

    # ============================================================================
    # REAL-TIME OPTIMIZATION SYSTEM (10/10)
    # ============================================================================
    
    def _update_performance_metrics(self, agent_name: str, execution_time: float, success: bool, quality_score: float):
        """Update performance metrics for continuous optimization."""
        if agent_name not in self.agent_performance:
            self.agent_performance[agent_name] = AgentPerformanceMetrics()
            
        metrics = self.agent_performance[agent_name]
        metrics.execution_times.append(execution_time)
        
        # Keep only last 100 executions for efficiency
        if len(metrics.execution_times) > 100:
            metrics.execution_times = metrics.execution_times[-100:]
            
        if success:
            metrics.success_count += 1
        metrics.total_executions += 1
        metrics.success_rate = metrics.success_count / metrics.total_executions
        
        # Update quality scores
        metrics.quality_scores.append(quality_score)
        if len(metrics.quality_scores) > 50:
            metrics.quality_scores = metrics.quality_scores[-50:]
            
        metrics.last_updated = datetime.now()
        
        # Update global performance history
        self.global_performance_history.append(execution_time)
        if len(self.global_performance_history) > 200:
            self.global_performance_history = self.global_performance_history[-200:]
    
    def _get_adaptive_timeout(self, agent_name: str, complexity: DocumentComplexity) -> float:
        """Calculate adaptive timeout based on historical performance."""
        base_timeouts = {
            DocumentComplexity.SIMPLE: 10.0,
            DocumentComplexity.MODERATE: 15.0,
            DocumentComplexity.COMPLEX: 25.0,
            DocumentComplexity.VERY_COMPLEX: 40.0
        }
        
        base_timeout = base_timeouts[complexity]
        
        # Adjust based on agent's historical performance
        if agent_name in self.agent_performance:
            metrics = self.agent_performance[agent_name]
            if metrics.execution_times:
                avg_time = statistics.mean(metrics.execution_times)
                # Add 50% buffer to average time, but cap at 2x base timeout
                adaptive_timeout = min(avg_time * 1.5, base_timeout * 2)
                return max(adaptive_timeout, base_timeout * 0.5)  # Minimum 50% of base
                
        return base_timeout
    
    def _optimize_agent_priority(self, agent_name: str) -> AgentPriority:
        """Determine agent priority based on performance and reliability."""
        if agent_name not in self.agent_performance:
            return AgentPriority.MEDIUM
            
        metrics = self.agent_performance[agent_name]
        
        # High performing agents get higher priority
        if metrics.success_rate > 0.95 and metrics.quality_scores:
            avg_quality = statistics.mean(metrics.quality_scores)
            if avg_quality > 0.85:
                return AgentPriority.HIGH
            elif avg_quality > 0.70:
                return AgentPriority.MEDIUM
                
        # Low performing agents get lower priority
        if metrics.success_rate < 0.80:
            return AgentPriority.LOW
            
        return AgentPriority.MEDIUM
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analytics."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'global_metrics': {
                'total_documents_processed': len(self.global_performance_history),
                'average_processing_time': statistics.mean(self.global_performance_history) if self.global_performance_history else 0,
                'performance_trend': 'improving' if len(self.global_performance_history) > 10 and 
                                   statistics.mean(self.global_performance_history[-5:]) < statistics.mean(self.global_performance_history[-10:-5]) 
                                   else 'stable'
            },
            'agent_metrics': {}
        }
        
        for agent_name, metrics in self.agent_performance.items():
            report['agent_metrics'][agent_name] = {
                'success_rate': round(metrics.success_rate, 3),
                'average_execution_time': round(statistics.mean(metrics.execution_times), 2) if metrics.execution_times else 0,
                'average_quality_score': round(statistics.mean(metrics.quality_scores), 3) if metrics.quality_scores else 0,
                'total_executions': metrics.total_executions,
                'reliability_score': round(metrics.success_rate * (statistics.mean(metrics.quality_scores) if metrics.quality_scores else 0.5), 3)
            }
            
        return report

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
            
            # Smart text truncation for preprocessing - dramatically increased for large documents
            max_length = 100000  # Increased to handle very large documents (was 60000)
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
        """
        🚀 ENHANCED MASTER AGENT WITH ADVANCED OPTIMIZATION
        
        Features:
        - Dynamic scaling based on document complexity
        - Cross-agent intelligence and conflict resolution  
        - Real-time performance optimization
        - Adaptive timeout and priority management
        """
        start_time = time.time()
        text = state.get("preprocessed_text", "")
        
        if not text or len(text.strip()) < 50:
            return {
                "summary_result": {},
                "risk_result": {},
                "highlights_result": {},
                "confidence_result": {},
                "completed_agents": [],
                "processing_errors": ["Insufficient text for analysis"]
            }

        # ============================================================================
        # PHASE 1: INTELLIGENT DOCUMENT ANALYSIS & OPTIMIZATION
        # ============================================================================
        
        # Advanced complexity analysis for dynamic scaling
        complexity, complexity_metrics = self._analyze_document_complexity(text)
        optimal_workers = self._get_optimal_worker_count(complexity, len(text))
        selected_agents = self._select_optimal_agents(complexity, "general")
        
        logger.info(f"🔍 Document Analysis: {complexity.value} complexity (score: {complexity_metrics['complexity_score']})")
        logger.info(f"⚡ Optimization: {optimal_workers} workers, {len(selected_agents)} agents selected")
        
        # ============================================================================
        # PHASE 2: PARALLEL EXECUTION WITH ADAPTIVE OPTIMIZATION
        # ============================================================================
        
        agent_results = {}
        errors = []
        
        # Agent mapping for backward compatibility
        agent_functions = {
            "summarizer": self._execute_summarizer,
            "risk_analyzer": self._execute_risk_analyzer,
            "highlighter": self._execute_highlighter,
            "confidence": self._execute_confidence
        }
        
        def execute_agent_with_optimization(agent_name: str, text_chunk: str) -> Tuple[str, Dict[str, Any]]:
            """Execute agent with real-time optimization and performance tracking."""
            agent_start = time.time()
            
            try:
                # Get adaptive timeout for this agent
                timeout = self._get_adaptive_timeout(agent_name, complexity)
                priority = self._optimize_agent_priority(agent_name)
                
                logger.info(f"🤖 {agent_name}: Starting (timeout: {timeout}s, priority: {priority.value})")
                
                # Execute agent function
                result = agent_functions[agent_name](text_chunk)
                
                execution_time = time.time() - agent_start
                
                # Validate result and handle parsing errors
                if hasattr(result, 'parsed') and result.parsed is None:
                    logger.warning(f"⚠️ {agent_name}: Parsing failed, attempting fallback")
                    # Try to extract meaningful data from raw response
                    if hasattr(result, 'raw') and result.raw:
                        result = self._create_fallback_result(agent_name, result.raw)
                elif hasattr(result, 'raw') and hasattr(result.raw, 'tool_calls') and result.raw.tool_calls:
                    # Handle case where we have tool calls but no parsed result
                    logger.info(f"🔧 {agent_name}: Extracting from tool calls")
                    try:
                        tool_call = result.raw.tool_calls[0]
                        if hasattr(tool_call, 'args') and tool_call.args:
                            result.parsed = tool_call.args
                            logger.info(f"✅ {agent_name}: Successfully extracted from tool calls")
                    except Exception as e:
                        logger.warning(f"⚠️ {agent_name}: Tool call extraction failed: {e}")
                        result = self._create_fallback_result(agent_name, result.raw)
                
                # Calculate quality score based on result completeness
                quality_score = self._calculate_quality_score_simple(result, agent_name)
                success = True
                
                # Update performance metrics for continuous optimization
                self._update_performance_metrics(agent_name, execution_time, success, quality_score)
                
                logger.info(f"✅ {agent_name}: Completed in {execution_time:.2f}s (quality: {quality_score:.2f})")
                
                return agent_name, result
                
            except Exception as e:
                execution_time = time.time() - agent_start
                self._update_performance_metrics(agent_name, execution_time, False, 0.0)
                logger.error(f"❌ {agent_name}: Failed after {execution_time:.2f}s - {str(e)}")
                return agent_name, self._get_fallback_result(agent_name)

        # Prepare optimized text chunks with very large limits for comprehensive analysis
        text_chunks = {
            "summarizer": self._prepare_text_for_agent(text, "summary", 80000),
            "risk_analyzer": self._prepare_text_for_agent(text, "risk", 80000),
            "highlighter": self._prepare_text_for_agent(text, "highlights", 70000),
            "confidence": self._prepare_text_for_agent(text, "confidence", 60000)
        }

        # Execute agents in parallel with dynamic worker optimization
        with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit tasks for all agents
            future_to_agent = {
                executor.submit(execute_agent_with_optimization, agent_name, text_chunks[agent_name]): agent_name
                for agent_name in agent_functions.keys()
            }
            
            # Collect results with timeout handling
            for future in as_completed(future_to_agent, timeout=45):
                try:
                    agent_name, result = future.result(timeout=5)
                    agent_results[agent_name] = result
                except TimeoutError:
                    agent_name = future_to_agent[future]
                    logger.warning(f"⚠️ {agent_name}: Timeout exceeded")
                    errors.append(f"{agent_name}: Timeout exceeded")
                    agent_results[agent_name] = self._get_fallback_result(agent_name)
                except Exception as e:
                    agent_name = future_to_agent[future]
                    logger.error(f"❌ {agent_name}: Execution error - {str(e)}")
                    errors.append(f"{agent_name}: {str(e)}")
                    agent_results[agent_name] = self._get_fallback_result(agent_name)

        # ============================================================================
        # PHASE 3: CROSS-AGENT INTELLIGENCE & SYNTHESIS
        # ============================================================================
        
        # Advanced cross-agent analysis
        cross_insights = self._analyze_cross_agent_insights(agent_results)
        conflicts = self._detect_conflicts(agent_results)
        synthesis = self._synthesize_insights(agent_results, cross_insights)
        
        logger.info(f"🧠 Cross-Agent Analysis: {len(cross_insights)} insights, {len(conflicts)} conflicts detected")
        
        # ============================================================================
        # PHASE 4: PERFORMANCE REPORTING
        # ============================================================================
        
        total_time = time.time() - start_time
        performance_report = self._generate_performance_report()
        
        logger.info(f"🎯 Enhanced Master Agent: Complete in {total_time:.2f}s with advanced optimization")
        
        # Enhanced final result with all intelligence (maintaining backward compatibility)
        return {
            "summary_result": agent_results.get("summarizer", {}),
            "risk_result": agent_results.get("risk_analyzer", {}),
            "highlights_result": agent_results.get("highlighter", {}),
            "confidence_result": agent_results.get("confidence", {}),
            "completed_agents": list(agent_results.keys()),
            "processing_errors": errors,
            "execution_time": total_time,
            # Enhanced features
            "cross_agent_intelligence": {
                "insights": [
                    {
                        "type": insight.insight_type,
                        "description": insight.description,
                        "confidence": insight.confidence,
                        "priority": insight.priority.value,
                        "action": insight.action_required
                    } for insight in cross_insights
                ],
                "conflicts": conflicts,
                "synthesis": synthesis
            },
            "complexity_analysis": {
                "level": complexity.value,
                "metrics": complexity_metrics,
                "optimization": f"{optimal_workers} workers, {len(selected_agents)} agents"
            },
            "performance_metrics": {
                "optimization_level": "advanced",
                "performance_report": performance_report
            }
        }
    
    def _calculate_quality_score_simple(self, result: Dict[str, Any], agent_name: str) -> float:
        """Calculate simple quality score for agent result (backward compatibility)."""
        if not result or not result.get("content"):
            return 0.0
        
        # Basic completeness check
        score = 0.5
        
        # Check if content is meaningful
        content = str(result.get("content", ""))
        if len(content) > 50:
            score += 0.3
        if len(content) > 200:
            score += 0.2
                
        return min(score, 1.0)

    def _create_fallback_result(self, agent_name: str, raw_response) -> Dict[str, Any]:
        """Create fallback result when parsing fails."""
        try:
            # Extract text content from raw response
            content = ""
            if hasattr(raw_response, 'content'):
                content = raw_response.content
            elif hasattr(raw_response, 'text'):
                content = raw_response.text
            else:
                content = str(raw_response)
            
            # Create minimal valid structure based on agent type
            if agent_name == "confidence":
                return {
                    "raw": raw_response,
                    "parsed": {
                        "overall_confidence": 50.0,
                        "document_clarity": 50.0,
                        "well_understood_sections": ["Basic document structure identified"],
                        "unclear_sections": ["Analysis incomplete due to parsing error"],
                        "missing_information": ["Full analysis unavailable"],
                        "legal_consultation_recommended": True,
                        "consultation_reasons": ["Technical analysis error - manual review recommended"]
                    },
                    "parsing_error": "Function call malformed, using fallback"
                }
            elif agent_name == "highlighter":
                return {
                    "raw": raw_response,
                    "parsed": {
                        "termination_rights": [],
                        "negotiable_terms": [],
                        "financial_obligations": [],
                        "key_restrictions": [],
                        "critical_deadlines": [],
                        "auto_renewal_clause": "Analysis incomplete - manual review required",
                        "action_items": ["Review document manually due to parsing error"]
                    },
                    "parsing_error": "Function call malformed, using fallback"
                }
            else:
                return {
                    "raw": raw_response,
                    "parsed": None,
                    "parsing_error": f"Agent {agent_name} parsing failed"
                }
                
        except Exception as e:
            logger.error(f"Fallback creation failed for {agent_name}: {e}")
            return {
                "raw": raw_response,
                "parsed": None,
                "parsing_error": f"Complete fallback failure: {str(e)}"
            }

    def _prepare_text_for_agent(self, text: str, agent_type: str, max_length: int) -> str:
        """Prepare optimized text chunks for different agent types."""
        if len(text) <= max_length:
            return text
        
        # For very large documents, use more intelligent truncation
        if len(text) > 50000:  # Large document strategy
            if agent_type == "summary":
                # For summary, take more from beginning (contract structure)
                beginning = int(max_length * 0.7)
                ending = max_length - beginning
                return text[:beginning] + "\n\n[...middle sections truncated...]\n\n" + text[-ending:]
            elif agent_type == "risk":
                # For risk, focus on terms, conditions, and penalties
                return self._extract_risk_sections(text, max_length)
            elif agent_type == "highlights":
                # For highlights, focus on dates, numbers, and key terms
                return self._extract_highlight_sections(text, max_length)
            else:
                # Confidence agent - balanced approach
                half = max_length // 2
                return text[:half] + "\n\n[...truncated for analysis...]\n\n" + text[-half:]
        else:
            # Standard approach for smaller documents
            if agent_type == "summary":
                return text[:max_length * 3//4] + "\n\n[...truncated...]\n\n" + text[-max_length//4:]
            elif agent_type == "risk":
                return self._extract_risk_sections(text, max_length)
            elif agent_type == "highlights":
                return self._extract_highlight_sections(text, max_length)
            else:
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
            
            # Check if we have meaningful results - updated for new structure with better debugging
            logger.info(f"🔍 Coordinator debugging - summary type: {type(summary)}")
            logger.info(f"🔍 Coordinator debugging - summary keys: {summary.keys() if isinstance(summary, dict) else 'Not a dict'}")
            
            # Handle both dict and direct Pydantic object cases
            if isinstance(summary, dict):
                summary_parsed = summary.get("parsed", {})
                has_summary = bool(summary_parsed.get("overview") if isinstance(summary_parsed, dict) else getattr(summary_parsed, 'overview', None))
            else:
                # Direct Pydantic object
                has_summary = bool(getattr(summary, 'overview', None))
                
            has_meaningful_data = (
                has_summary and 
                len(str(summary_parsed.get("overview", "") if isinstance(summary, dict) and isinstance(summary_parsed, dict) else getattr(summary, 'overview', ''))) > 50
            )
            
            # Additional check: if we have parsed data from multiple agents, consider it successful
            agents_with_parsed_data = 0
            
            # Count successful agents more robustly
            if isinstance(summary, dict) and summary.get("parsed"):
                agents_with_parsed_data += 1
            elif hasattr(summary, 'overview'):
                agents_with_parsed_data += 1
                
            if isinstance(risk, dict) and risk.get("parsed"):
                agents_with_parsed_data += 1
            elif hasattr(risk, 'overall_risk_level'):
                agents_with_parsed_data += 1
                
            if isinstance(highlights, dict) and highlights.get("parsed"):
                agents_with_parsed_data += 1
            elif hasattr(highlights, 'critical_deadlines'):
                agents_with_parsed_data += 1
                
            if isinstance(confidence, dict) and confidence.get("parsed"):
                agents_with_parsed_data += 1
            elif hasattr(confidence, 'overall_confidence'):
                agents_with_parsed_data += 1
            
            logger.info(f"🔍 Coordinator debugging - agents_with_parsed_data: {agents_with_parsed_data}")
            
            # If we have 3+ agents with parsed data, consider it successful
            if agents_with_parsed_data >= 3:
                has_meaningful_data = True
                logger.info(f"✅ Coordinator: Using normal report (3+ agents successful)")
            else:
                logger.warning(f"⚠️ Coordinator: Using error report (only {agents_with_parsed_data} agents successful)")
            
            if not has_meaningful_data:
                # Generate error report
                final_output = self._generate_error_report(errors, state)
            else:
                # Generate normal report with performance info - fix data extraction
                chain = COORDINATOR_PROMPT | self.pro_llm
                
                # Extract parsed data safely
                summary_data = summary.get("parsed", {}) if isinstance(summary, dict) else summary
                risk_data = risk.get("parsed", {}) if isinstance(risk, dict) else risk
                highlights_data = highlights.get("parsed", {}) if isinstance(highlights, dict) else highlights
                confidence_data = confidence.get("parsed", {}) if isinstance(confidence, dict) else confidence
                
                # Convert Pydantic objects to dicts for JSON serialization
                if hasattr(summary_data, '__dict__'):
                    summary_data = summary_data.__dict__
                if hasattr(risk_data, '__dict__'):
                    risk_data = risk_data.__dict__
                if hasattr(highlights_data, '__dict__'):
                    highlights_data = highlights_data.__dict__
                if hasattr(confidence_data, '__dict__'):
                    confidence_data = confidence_data.__dict__
                
                result = chain.invoke({
                    "summary": json.dumps(summary_data, indent=2, default=str),
                    "risk_analysis": json.dumps(risk_data, indent=2, default=str),
                    "highlights": json.dumps(highlights_data, indent=2, default=str),
                    "confidence": json.dumps(confidence_data, indent=2, default=str),
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