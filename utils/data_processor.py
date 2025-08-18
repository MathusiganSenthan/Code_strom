"""
Data processing utilities to clean analysis responses and remove system prompt artifacts.
"""

import re
from typing import Dict, Any, List

def clean_analysis_text(text: str) -> str:
    """
    Clean analysis text by removing system prompts and artifacts.
    
    Args:
        text: Raw analysis text that may contain system prompts
        
    Returns:
        Cleaned analysis text with just the content
    """
    if not text:
        return ""
    
    # Remove common system prompt patterns
    patterns_to_remove = [
        r"You are an expert.*?(?=\n\n|\n#|$)",
        r"Your analysis should be.*?(?=\n\n|\n#|$)",
        r"Focus on practical implications.*?(?=\n\n|\n#|$)",
        r"Always prioritize the user's interests.*?(?=\n\n|\n#|$)",
        r"Analyze this legal document.*?(?=\n\n|\n#|$)",
        r"Document Text:\s*\{text\}",
        r"Create a clear.*?(?=\n\n|\n#|$)",
        r"IMPORTANT:.*?(?=\n\n|\n#|$)",
        r"Remember:.*?(?=\n\n|\n#|$)",
    ]
    
    cleaned_text = text
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove multiple blank lines
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    
    # Remove leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    # If text starts with a header, ensure it's properly formatted
    if cleaned_text and not cleaned_text.startswith('#'):
        # Find the first proper content section (usually starts with ## or **Risk Level**)
        content_match = re.search(r'(##?\s|^\*\*Risk Level)', cleaned_text, re.MULTILINE)
        if content_match:
            cleaned_text = cleaned_text[content_match.start():]
    
    return cleaned_text

def process_risk_assessment_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and clean risk assessment response data.
    
    Args:
        raw_response: Raw response from risk analyzer agent
        
    Returns:
        Cleaned and structured risk assessment data
    """
    if not raw_response:
        return {}
    
    # Clean the analysis text if it exists
    if 'analysis' in raw_response:
        raw_response['analysis'] = clean_analysis_text(raw_response['analysis'])
    
    # Ensure critical_risks is properly structured
    if 'critical_risks' in raw_response:
        critical_risks = raw_response['critical_risks']
        if isinstance(critical_risks, list) and critical_risks:
            # Convert string risks to structured format if needed
            structured_risks = []
            for i, risk in enumerate(critical_risks):
                if isinstance(risk, str):
                    # Convert string to structured risk item
                    structured_risks.append({
                        'id': i + 1,
                        'title': risk,
                        'type': 'GENERAL',
                        'severity': 'HIGH SEVERITY',
                        'section': 'General',
                        'description': risk,
                        'impact': 'Potential negative consequences',
                        'recommendation': 'Review and consider mitigation',
                        'confidence': 75
                    })
                elif isinstance(risk, dict):
                    # Ensure all required fields are present
                    structured_risk = {
                        'id': risk.get('id', i + 1),
                        'title': risk.get('title', f'Risk {i + 1}'),
                        'type': risk.get('type', 'GENERAL'),
                        'severity': risk.get('severity', 'HIGH SEVERITY'),
                        'section': risk.get('section', 'General'),
                        'description': risk.get('description', ''),
                        'impact': risk.get('impact', ''),
                        'recommendation': risk.get('recommendation', ''),
                        'confidence': risk.get('confidence', 75)
                    }
                    structured_risks.append(structured_risk)
            
            raw_response['critical_risks'] = structured_risks
    
    return raw_response

def process_highlights_response(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and clean key highlights response data.
    
    Args:
        raw_response: Raw response from highlighter agent
        
    Returns:
        Cleaned and structured highlights data
    """
    if not raw_response:
        return {}
    
    # Clean the analysis text if it exists
    if 'analysis' in raw_response:
        raw_response['analysis'] = clean_analysis_text(raw_response['analysis'])
    
    # Ensure critical_deadlines is properly structured
    if 'critical_deadlines' in raw_response:
        deadlines = raw_response['critical_deadlines']
        if isinstance(deadlines, list) and deadlines:
            structured_deadlines = []
            for i, deadline in enumerate(deadlines):
                if isinstance(deadline, str):
                    structured_deadlines.append({
                        'id': i + 1,
                        'title': deadline,
                        'description': deadline,
                        'dueDate': 'TBD',
                        'party': 'TBD',
                        'priority': 'MEDIUM',
                        'category': 'General'
                    })
                elif isinstance(deadline, dict):
                    structured_deadline = {
                        'id': deadline.get('id', i + 1),
                        'title': deadline.get('title', f'Deadline {i + 1}'),
                        'description': deadline.get('description', ''),
                        'dueDate': deadline.get('dueDate', 'TBD'),
                        'party': deadline.get('party', 'TBD'),
                        'priority': deadline.get('priority', 'MEDIUM'),
                        'category': deadline.get('category', 'General')
                    }
                    structured_deadlines.append(structured_deadline)
            
            raw_response['critical_deadlines'] = structured_deadlines
    
    # Ensure financial_obligations is properly structured
    if 'financial_obligations' in raw_response:
        obligations = raw_response['financial_obligations']
        if isinstance(obligations, list) and obligations:
            structured_obligations = []
            for i, obligation in enumerate(obligations):
                if isinstance(obligation, str):
                    structured_obligations.append({
                        'id': i + 1,
                        'title': obligation,
                        'description': obligation,
                        'amount': 'TBD',
                        'due_date': 'TBD',
                        'party': 'TBD',
                        'priority': 'MEDIUM',
                        'category': 'Payment'
                    })
                elif isinstance(obligation, dict):
                    structured_obligation = {
                        'id': obligation.get('id', i + 1),
                        'title': obligation.get('title', f'Obligation {i + 1}'),
                        'description': obligation.get('description', ''),
                        'amount': obligation.get('amount', 'TBD'),
                        'due_date': obligation.get('due_date', 'TBD'),
                        'party': obligation.get('party', 'TBD'),
                        'priority': obligation.get('priority', 'MEDIUM'),
                        'category': obligation.get('category', 'Payment')
                    }
                    structured_obligations.append(structured_obligation)
            
            raw_response['financial_obligations'] = structured_obligations
    
    # Ensure auto_renewal_clause is properly structured
    if 'auto_renewal_clause' in raw_response:
        renewal = raw_response['auto_renewal_clause']
        if isinstance(renewal, str):
            raw_response['auto_renewal_clause'] = {
                'exists': renewal.lower() not in ['none', 'no', 'false'],
                'renewal_period': renewal if renewal.lower() not in ['none', 'no', 'false'] else 'N/A',
                'notice_required': 'TBD',
                'automatic': True
            }
        elif not isinstance(renewal, dict):
            raw_response['auto_renewal_clause'] = {
                'exists': False,
                'renewal_period': 'N/A',
                'notice_required': 'N/A',
                'automatic': False
            }
    
    return raw_response

def clean_final_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and process the final analysis response.
    
    Args:
        response_data: Raw response data from the workflow
        
    Returns:
        Cleaned response data ready for frontend
    """
    if not response_data:
        return {}
    
    # Clean analysis text in the main analysis field
    if 'analysis' in response_data:
        response_data['analysis'] = clean_analysis_text(response_data['analysis'])
    
    # Process components
    if 'components' in response_data:
        components = response_data['components']
        
        # Clean risk assessment
        if 'risk_assessment' in components:
            components['risk_assessment'] = process_risk_assessment_response(
                components['risk_assessment']
            )
        
        # Clean key highlights
        if 'key_highlights' in components:
            components['key_highlights'] = process_highlights_response(
                components['key_highlights']
            )
        
        # Clean summary analysis
        if 'summary' in components and 'analysis' in components['summary']:
            components['summary']['analysis'] = clean_analysis_text(
                components['summary']['analysis']
            )
        
        # Clean confidence analysis
        if 'confidence_metrics' in components and 'analysis' in components['confidence_metrics']:
            components['confidence_metrics']['analysis'] = clean_analysis_text(
                components['confidence_metrics']['analysis']
            )
    
    return response_data
