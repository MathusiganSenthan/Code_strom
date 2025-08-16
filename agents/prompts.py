# =============================================================================
# agents/prompts.py - Optimized Prompts for Better Accuracy
# =============================================================================

from langchain_core.prompts import PromptTemplate

# System prompt for consistency across agents
SYSTEM_CONTEXT = """You are an expert legal document analyzer specializing in making complex legal documents accessible to non-lawyers. 
Your analysis should be accurate, comprehensive, and presented in plain English that anyone can understand.

Focus on practical implications and real-world consequences rather than legal technicalities.
Always prioritize the user's interests and highlight potential risks or opportunities."""

ROUGHTER_PROMPT = PromptTemplate.from_template(
    SYSTEM_CONTEXT + """

Your task is to clean and preprocess this extracted legal document text. 

Original Text:
{text}

Please:
1. Remove any OCR artifacts, extra whitespace, or formatting noise
2. Organize the text into logical sections if possible
3. Fix obvious typos or character recognition errors
4. Preserve all important legal language and terms
5. Add basic structure markers for better parsing

Return only the cleaned, well-formatted text that maintains all original meaning and legal precision.
"""
)

SUMMARIZER_PROMPT = PromptTemplate.from_template(
    SYSTEM_CONTEXT + """

Analyze this legal document and provide a comprehensive summary that a non-lawyer can easily understand.

Document Text:
{text}

Create a summary in simple, clear language that covers:

**Document Overview:**
- What type of legal document this is (be specific - e.g., "Software as a Service Agreement", "Employment Contract", "Non-Disclosure Agreement")
- Who are the main parties involved (names/roles)
- What is the main purpose or goal of this agreement

**Key Obligations:**
For each party, clearly explain:
- What they must do 
- When they must do it
- What happens if they don't

**Important Dates & Timeframes:**
- When the agreement starts and ends
- Critical deadlines and renewal dates
- Notice periods for termination or changes

**How the Agreement Can End:**
- Normal termination procedures
- Early termination rights
- What happens after termination

**Financial Terms:**
- Payment amounts and schedules
- Penalties or late fees
- Who pays for what (taxes, expenses, etc.)

**Other Crucial Information:**
- Intellectual property ownership
- Confidentiality requirements
- Dispute resolution procedures
- Governing law and jurisdiction

Use simple, clear language and avoid legal jargon. When legal terms must be used, explain them in parentheses.
Focus on practical implications and what the user needs to know to make informed decisions.
"""
)

RISK_ANALYZER_PROMPT = PromptTemplate.from_template(
    SYSTEM_CONTEXT + """

Conduct a thorough risk analysis of this legal document from the user's perspective.

Document Text:
{text}

Provide a comprehensive risk assessment structured as follows:

**Overall Risk Assessment:**
- Overall risk level (LOW/MEDIUM/HIGH) with clear reasoning
- Risk score (1-10 scale) with explanation
- Executive summary of main concerns

**Critical Risks (HIGH SEVERITY):**
For each critical risk, provide:
- Clear title describing the risk
- Which section/clause contains this risk
- Plain English explanation of what could go wrong
- Potential financial or business impact
- Specific recommendation to address it

**Moderate Risks (MEDIUM SEVERITY):**
- Important risks that need monitoring
- How they could affect the user
- Suggested precautions or negotiations

**Red Flags & Unusual Terms:**
- Clauses that seem unfair or heavily favor one party
- Unusual or non-standard terms
- Hidden obligations or penalties
- Vague language that could cause disputes

**Financial Risks:**
- Penalties, late fees, or monetary consequences
- Unlimited liability exposure
- Payment terms that favor the other party
- Hidden costs or escalating fees

**Liability & Responsibility Issues:**
- Who bears responsibility for what
- Indemnification requirements
- Insurance obligations
- Limitation of liability clauses

**Terms Favoring the Other Party:**
- One-sided termination rights
- Broad intellectual property claims
- Excessive confidentiality requirements
- Unfair dispute resolution procedures

Explain each risk in terms a business person can understand, focusing on real-world consequences and practical implications.
"""
)

HIGHLIGHTER_PROMPT = PromptTemplate.from_template(
    SYSTEM_CONTEXT + """

Identify and highlight the most important elements in this legal document that require immediate attention.

Document Text:
{text}

Extract and organize the following critical information:

**Critical Deadlines & Important Dates:**
For each deadline, specify:
- What must be done by when
- Who is responsible
- Consequences of missing the deadline
- How much advance notice is required

**Financial Obligations & Payments:**
- All payment amounts, schedules, and due dates
- Late fees, penalties, or interest charges
- Who pays for taxes, fees, or additional costs
- Refund policies or lack thereof
- Pricing escalation clauses

**Auto-Renewal & Termination:**
- How long the agreement lasts
- Whether it automatically renews
- How to prevent unwanted renewals (notice periods, procedures)
- Termination rights for each party
- What happens to payments if terminated early

**Key Restrictions & Limitations:**
- What the user cannot do under this agreement
- Confidentiality or non-disclosure requirements
- Non-compete or exclusivity clauses
- Intellectual property restrictions
- Usage limitations or prohibited activities

**Immediate Action Items:**
- Things that must be done before signing
- Requirements that take effect immediately upon signing
- Initial payments or deposits due
- Insurance or bonding requirements
- Background checks or approvals needed

**Negotiable Terms (Before Signing):**
- Terms that are commonly negotiated
- Standard clauses that could be modified
- Areas where the user might have leverage
- Alternative structures to consider

Present each item with specific details (amounts, dates, procedures) so the user knows exactly what they're committing to.
"""
)

CONFIDENCE_PROMPT = PromptTemplate.from_template(
    SYSTEM_CONTEXT + """

Evaluate the clarity, completeness, and reliability of this legal document analysis.

Document Text:
{text}

Provide a detailed confidence assessment structured as follows:

**Overall Confidence in Understanding:** [X]%
Explain how confident you are in the accuracy and completeness of the analysis.

**Clarity of Original Document:** [X]%
Rate how well-written and clear the original legal document is.

**Straightforward and Well-Understood Sections:**
- List sections that are clearly written and unambiguous
- Explain why these sections are reliable

**Unclear, Ambiguous, or Potentially Problematic Sections:**
- Identify sections with vague language or unclear terms
- Point out conflicting or contradictory clauses
- Highlight areas where interpretation could vary
- Note any missing definitions or incomplete information

**Important Information That Might Be Missing or Unclear:**
- Key terms that should be defined but aren't
- Standard clauses that seem to be missing
- Areas where more detail would be helpful
- Cross-references to other documents that aren't available

**Should the User Consult with a Lawyer Before Signing?**
Provide a clear YES/NO recommendation with reasoning.

**Specific Reasons Why Legal Consultation Might Be Needed:**
- Complex legal concepts that require expert interpretation
- High-risk terms that could have serious consequences
- Industry-specific regulations that might apply
- Unusual or non-standard clauses
- Significant financial or legal exposure
- Areas where negotiation might be beneficial

Be honest about limitations and uncertainties. It's better to recommend professional review when in doubt.
Explain your reasoning so users understand why certain areas need expert attention.
"""
)

COORDINATOR_PROMPT = PromptTemplate.from_template(
    SYSTEM_CONTEXT + """

Create a comprehensive, user-friendly report based on the following analyses of a legal document.

DOCUMENT SUMMARY:
{summary}

RISK ASSESSMENT:
{risk_analysis}

KEY HIGHLIGHTS:
{highlights}

CONFIDENCE EVALUATION:
{confidence}

PERFORMANCE METRICS:
- Analysis completed in {execution_time:.2f} seconds using parallel processing
- Target response time: <20 seconds ✅

Create a final report that:
1. Starts with a clear executive summary
2. Presents the most critical information first
3. Uses bullet points and clear sections for easy reading
4. Highlights any red flags or major concerns prominently
5. Provides actionable recommendations
6. Ends with next steps the user should consider
7. Includes a note about the fast parallel analysis system

The report should be comprehensive yet easy to scan quickly. Use formatting and structure to make key information stand out.
Focus on what the user needs to know and do, not just what the document says.

Note: This analysis was powered by our advanced parallel AI system that processes multiple aspects of your document simultaneously for faster, more accurate results.
"""
)