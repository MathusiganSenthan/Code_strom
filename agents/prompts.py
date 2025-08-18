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

Analyze this legal document and create a comprehensive summary in simple, everyday language. Avoid legal jargon and explain everything in plain English.

Document Text:
{text}

IMPORTANT: Start directly with the document description. Do NOT include ANY headers like "Document Analysis", "Executive Summary", "Document Overview", or similar. The frontend will provide the main heading. Start immediately with the document title and description.

**Document:** [Document Type - e.g., Software as a Service Agreement, Employment Contract, Terms of Service]

This is a <mark class="legal-term">**[document type]**</mark> between [main parties] that [main purpose and key terms in 2-3 clear sentences]. [Overall assessment of fairness/complexity in 1 sentence].

**Key Parties:**

- **[Party 1]:** [Their role and responsibilities]
- **[Party 2]:** [Their role and responsibilities]

---

## Your Main Responsibilities

### Primary Obligations
1. **[Primary obligation]:** [Clear explanation of what this means]
2. **[Secondary obligation]:** [Clear explanation of what this means]  
3. **[Third obligation]:** [Clear explanation of what this means]

### Ongoing Requirements
- Regular compliance with [specific requirement]
- Notification obligations when [specific events occur]
- Maintenance of [specific standards/certifications]

---

## Financial Terms & Money Matters

### Payment Structure
1. **Base Payment:** <mark class="financial">**$[amount]**</mark> [frequency/timing]
2. **Additional Fees:** <mark class="financial">**$[amount]**</mark> for [specific services]
3. **Late Penalties:** <mark class="financial">**$[amount]**</mark> or [percentage] if payment delayed beyond <mark class="deadline">**[X days]**</mark>

### Cost Protection
- **Payment Cap:** Maximum you'll pay is <mark class="financial">**$[amount]**</mark>
- **Refund Policy:** [Conditions under which money can be returned]
- **Hidden Costs:** [Any additional fees to watch for]

---

## Important Dates & Critical Deadlines

### Contract Timeline
1. **Start Date:** <mark class="deadline">**[date]**</mark>
2. **Key Milestones:** <mark class="deadline">**[dates]**</mark> for [specific deliverables]
3. **End Date:** <mark class="deadline">**[date]**</mark> (unless renewed)

### Action Required Dates
- **Payment Due:** <mark class="deadline">**[X days]**</mark> from [trigger event]
- **Notice Required:** <mark class="deadline">**[X days]**</mark> before [termination/changes]
- **Renewal Decision:** <mark class="deadline">**[X days]**</mark> before contract expires

---

## Critical Terms That Need Your Attention

### Auto-Renewal & Termination
1. **Auto-Renewal:** <mark class="legal-term">**Automatic renewal**</mark> for [period] unless you give <mark class="deadline">**[X days]**</mark> notice
2. **Termination Rights:** [Who can terminate and under what conditions]
3. **Exit Requirements:** [What you must do to end the agreement]

### High-Risk Legal Clauses
1. **Liability Cap:** They're only responsible for up to <mark class="financial">**$[amount]**</mark>
2. **Indemnification:** <mark class="legal-term">**You must protect them**</mark> from [specific types of claims]
3. **IP Ownership:** [Who owns what intellectual property]
4. **Data Rights:** [What happens to your data]

### One-Sided Terms to Watch
- **Unilateral Changes:** <mark class="legal-term">**They can modify**</mark> [terms/pricing] with [notice period]
- **Dispute Resolution:** <mark class="legal-term">**Mandatory arbitration**</mark> in [location]
- **Governing Law:** Disputes handled under [state/country] law

---

## Overall Assessment & Recommendations

### Risk Summary
- **Complexity Level:** [Simple/Moderate/Complex] - [One sentence explanation]
- **Fairness Rating:** [Fair/Mostly Fair/Concerning/Heavily One-Sided] - [Brief explanation]
- **Overall Risk:** [Low/Medium/High] based on [key factors]

### Action Items
1. **Immediate:** [Most urgent thing to address before signing]
2. **Important:** [Key terms to negotiate or clarify]
3. **Consider:** [Whether legal review is recommended]

### Bottom Line
[2-3 sentences summarizing whether this is a good deal and main things to watch out for]

Use selective highlighting for maximum impact:
- <mark class="legal-term">**critical legal terms**</mark> for auto-renewal, termination, liability, indemnification, penalties
- <mark class="financial">**$[amount]**</mark> for all monetary values
- <mark class="deadline">**[timeframe]**</mark> for all dates and deadlines  
- **bold** for general structure and emphasis
- *italics* for general legal language

CRITICAL: Always mark these high-risk terms:
- <mark class="legal-term">auto-renewal</mark>
- <mark class="legal-term">automatic renewal</mark>
- <mark class="legal-term">penalty</mark>
- <mark class="legal-term">termination fee</mark>
- <mark class="legal-term">cancellation fee</mark>
- <mark class="legal-term">breach</mark>
- <mark class="legal-term">default</mark>
- <mark class="legal-term">indemnification</mark>
- <mark class="legal-term">liability limitation</mark>
"""
)

RISK_ANALYZER_PROMPT = PromptTemplate.from_template(
    SYSTEM_CONTEXT + """

Analyze this legal document for potential risks and explain them in simple, everyday language. Focus on practical consequences and what could realistically go wrong.

Document Text:
{text}

IMPORTANT: Use selective highlighting for legal terms only:
- Use **bold text** for general emphasis and structure (headings, section titles)
- Use `<mark class="legal-term">important legal term</mark>` for critical legal terms like: penalties, fees, termination, auto-renewal, liability, indemnification
- Use `<mark class="financial">$amounts</mark>` for financial figures and costs
- Use `<mark class="deadline">dates and deadlines</mark>` for time-sensitive information
- Use numbered lists (1. 2. 3.) for sequential steps or priorities
- Use bullet points (- or •) for related items
- Use ### for subsection headers
- Use > blockquotes for important warnings
- Use tables for structured data when appropriate

Provide ONLY the risk analysis content below. Do NOT include any introductory phrases. Start directly with the Overall Risk Assessment.

## Overall Risk Assessment

**Risk Level:** [LOW/MEDIUM/HIGH] 

**Risk Score:** [1-10]/10 - [Explain in one simple sentence why this score]

**Quick Summary:** [2-3 sentences explaining the main things to watch out for]

---

## Critical Risks (Things That Could Really Hurt You)

For each major risk, use this numbered structure:

### 1. **[Risk Title in Plain English]**

| **Aspect** | **Details** |
|------------|-------------|
| **What This Means** | [Explain like you're talking to a friend] |
| **Where It's Found** | *Section [X], Clause [Y]* |
| **What Could Go Wrong** | [Real-world consequences] |
| **Potential Cost/Impact** | <mark class="financial">**$[Amount]**</mark> or [time/business impact] |
| **What You Should Do** | [Specific recommendation] |
| **Warning** | [Any immediate action needed] |

### 2. **[Next Risk Title]**
[Continue with same structure...]

---

## Important Things to Watch For

### A. **Unusual or One-Sided Terms**

1. **Auto-Renewal Clauses**
   - How the contract might renew: <mark class="legal-term">auto-renewal</mark> terms
   - Notice required: <mark class="deadline">**[timeframe] days**</mark>
   - Impact: [explain consequences]

2. **Termination Rights**
   - Who can end it: [details]
   - How easily: [process and requirements]
   - Penalties: <mark class="financial">**$[amount]**</mark> or <mark class="legal-term">termination fees</mark>

3. **Penalty Clauses**
   - Types of penalties: <mark class="legal-term">late fees</mark>, <mark class="legal-term">penalties</mark>
   - Amounts: <mark class="financial">**$[specific amounts]**</mark>
   - Triggers: [what causes them]

### B. **Financial Risks**

| **Risk Type** | **Details** | **Potential Cost** | **Priority** |
|---------------|-------------|-------------------|--------------|
| **Payment Penalties** | [Late fees, interest] | <mark class="financial">**$[amount]**</mark> | High/Medium/Low |
| **Liability Exposure** | [Maximum you could owe] | <mark class="financial">**$[amount]**</mark> | High/Medium/Low |
| **Hidden Costs** | [Surprise fees] | <mark class="financial">**$[amount]**</mark> | High/Medium/Low |

### C. **Problematic Clauses**

- **Unfair Terms:** [Things heavily favoring the other party]
  - Specific issues: [bullet list]
  - Impact on you: [consequences]

- **Vague Language:** [Unclear terms that could cause disputes]
  - Examples: `"[quote specific problematic language]"`
  - Why it matters: [explanation]

---

## What You Should Consider

### Before Signing (Priority Order):

1. **Immediate Action Required:**
   - [ ] [Specific task 1]
   - [ ] [Specific task 2]
   - [ ] [Specific task 3]

2. **Questions to Ask:**
   - *[Question 1]*
   - *[Question 2]*
   - *[Question 3]*

3. **Terms to Negotiate:**
   - **[Term 1]:** [why and how to change]
   - **[Term 2]:** [why and how to change]

### Red Flags That Need Immediate Attention:

> 🚨 **CRITICAL:** [Most concerning issue]
> 
> **Impact:** [what this means for you]
> **Action:** [what to do about it]

> ⚠️ **IMPORTANT:** [Second most concerning issue]
> 
> **Impact:** [consequences]
> **Action:** [recommendation]

Remember: Only highlight critical legal terms using the <mark> tags. Use **bold** for general structure and emphasis, not for highlighting every important word.
"""
)

HIGHLIGHTER_PROMPT = PromptTemplate.from_template(
    SYSTEM_CONTEXT + """

Extract and highlight the most critical information from this legal document in simple, everyday language. Focus on obligations, unusual terms, and important clauses that need immediate attention.

Document Text:
{text}

IMPORTANT: Use proper markdown formatting with selective highlighting:
- Use <mark class="legal-term">**[term]**</mark> for critical legal terms and contract clauses
- Use <mark class="financial">**$[amount]**</mark> for money amounts, costs, and penalties  
- Use <mark class="deadline">**[date/time]**</mark> for important dates, deadlines, and timeframes
- Use **bold text** for general emphasis and structure (not highlighting)
- Use *italic text* for general legal language and contract clauses
- Use numbered lists for sequential steps and priorities
- Use bullet points for related items and lists
- Use tables for structured financial information
- Use > blockquotes for warnings and important notices
- Use `code formatting` for specific contract language quotes

Provide ONLY the key highlights content below. Do NOT include any introductory phrases. Start directly with the content sections.

## Critical Deadlines & Important Dates

### 📅 Key Timeframes You Must Remember

| **What You Need to Do** | **When It's Due** | **Who's Responsible** | **Consequences if Missed** |
|--------------------------|--------------------|-----------------------|----------------------------|
| [Action required] | <mark class="deadline">**[Specific date]**</mark> | [You/Them/Both] | [Real consequences] |
| [Action required] | <mark class="deadline">**[Timeframe]**</mark> | [You/Them/Both] | [Real consequences] |

### Contract Duration & Renewal

1. **Duration:** [Length of agreement]
2. **Renewal Process:** 
   - *Auto-renewal:* [How it continues automatically]
   - *Manual renewal:* [Steps required to renew]
   - **Notice required:** <mark class="deadline">**[X days]**</mark> before [event]

---

## Money Matters & Financial Obligations

### 💰 What You Pay

| **Payment Type** | **Amount** | **Frequency** | **Due Date** |
|------------------|------------|---------------|--------------|
| Main payment | <mark class="financial">**$[amount]**</mark> | [frequency] | <mark class="deadline">**[date]**</mark> |
| Setup fees | <mark class="financial">**$[amount]**</mark> | One-time | <mark class="deadline">**[when]**</mark> |
| Additional charges | <mark class="financial">**$[amount]**</mark> | [when applicable] | <mark class="deadline">**[timing]**</mark> |

### ⚠️ Penalties & Extra Costs

> **Warning:** These charges can add up quickly!

- **Late payment fees:** **$[amount]** or **[%]%** after **[X] days**
- **Termination penalties:** **$[amount]** if ended before **[date]**
- **Other penalties:** [List specific triggers and costs]

### Refunds & Cancellation Policy

- **Refund eligibility:** [If/when you can get money back]
- **Refund amount:** **[%]%** or **$[amount]**
- **Process time:** **[X] business days**
- **Early termination costs:** **$[amount]**

---

## Important Clauses That Need Your Attention

### 🔍 Unusual or Concerning Terms

1. **Auto-Renewal Clauses**
   - How it works: *[specific mechanism]*
   - Notice required: <mark class="deadline">**[X] days**</mark> before *[date]*
   - How to cancel: [specific steps]

2. **One-Sided Terms** ⚠️
   - What favors them: [specific advantages]
   - Impact on you: [your limitations]
   - Why it matters: [real-world consequences]

3. **Penalty Clauses**
   
   | **Trigger** | **Penalty** | **Severity** |
   |-------------|-------------|--------------|
   | [Action/event] | <mark class="financial">**$[amount]**</mark> | High/Medium/Low |
   | [Action/event] | **[consequence]** | High/Medium/Low |

### 📋 Your Key Responsibilities & Obligations

#### What You Must Do:

1. **[Primary obligation]**
   - Details: [specific requirements]
   - Timeline: <mark class="deadline">**[when/how often]**</mark>
   - Consequences: [what happens if you don't]

2. **[Secondary obligation]**
   - Details: [specific requirements]
   - Timeline: <mark class="deadline">**[when/how often]**</mark>

#### What You Cannot Do:

- <mark class="legal-term">**[Restriction 1]:**</mark> [explanation and impact]
- <mark class="legal-term">**[Restriction 2]:**</mark> [explanation and impact]

### 🏢 Their Responsibilities

| **What They Must Provide** | **Performance Standard** | **Timeline** |
|----------------------------|--------------------------|--------------|
| [Service/product] | [quality/quantity standard] | **[timeframe]** |
| [Service/product] | [quality/quantity standard] | **[timeframe]** |

---

## Red Flags & Terms to Negotiate

### 🚨 Before You Sign (Priority Order):

1. **URGENT - Address Immediately:**
   - [ ] <mark class="legal-term">**[Critical issue]:**</mark> [why it matters]
   - [ ] <mark class="legal-term">**[Critical issue]:**</mark> [why it matters]

2. **Important - Should Negotiate:**
   - <mark class="legal-term">**[Term to change]:**</mark> [current problem] → [suggested improvement]
   - <mark class="legal-term">**[Term to change]:**</mark> [current problem] → [suggested improvement]

3. **Questions to Ask:**
   - *"[Specific question about vague term]"*
   - *"[Question about cost/penalty]"*
   - *"[Question about termination/changes]"*

### ⚡ Immediate Action Required

> **🔥 CRITICAL:** [Most urgent issue requiring immediate attention]
> 
> **Why this matters:** [impact on you]
> **What to do:** [specific action steps]
> **Deadline:** <mark class="deadline">**[when you must act]**</mark>

> **⚠️ IMPORTANT:** [Second priority issue]
> 
> **Impact:** [consequences]
> **Recommendation:** [what to do about it]

### 📝 Documentation Needed

- [ ] **[Document type]:** [why needed, when due]
- [ ] **[Document type]:** [why needed, when due]
- [ ] **[Insurance/permits]:** [coverage amounts, deadlines]

Use selective highlighting with <mark> tags for critical terms: <mark class="financial">**amounts**</mark>, <mark class="deadline">**dates**</mark>, and <mark class="legal-term">**critical legal terms**</mark>. Use **bold** for general structure and emphasis. Use *italics* for general legal language. Structure everything for easy scanning and quick decision-making.
"""
)

CONFIDENCE_PROMPT = PromptTemplate.from_template(
    SYSTEM_CONTEXT + """

Assess how well this legal document can be understood and provide a confidence rating for the analysis. Focus on document clarity, complexity balance, and potential interpretation issues.

Document Text:
{text}

Provide a comprehensive assessment using markdown formatting for frontend preview.

##  Analysis Confidence Assessment

**Overall Confidence Level:** [X]% - [Explain in simple terms why this confidence level]

**Document Quality Rating:** [X]% - [How well-written and clear is the original document]

##  Document Complexity Breakdown

**Complexity Level:** [Simple/Moderate/Complex/Very Complex]

**Why This Rating:**
[Explain in 2-3 sentences what makes this document this complexity level]

### Balanced Evaluation: Simple vs Complex Elements

** Straightforward Parts (Easy to Understand):**
- [List 3-4 sections that are clearly written]
- [Explain why these are reliable and clear]
- [Note standard, commonly-used language]

** Complex Parts (Need Careful Attention):**
- [Identify 3-4 sections requiring expert interpretation]
- [Note technical legal concepts or industry-specific terms]
- [Highlight areas where interpretation could vary]

##  Areas of Concern for Understanding

### Unclear or Ambiguous Language
- **Vague Terms:** [Words/phrases that could mean different things]
- **Missing Definitions:** [Important terms that should be defined but aren't]
- **Contradictory Clauses:** [Sections that might conflict with each other]

### Potential Interpretation Issues
- [Areas where different people might understand things differently]
- [Clauses that could be argued both ways]
- [Missing details that could cause disputes later]

### What Might Be Missing
- **Standard Clauses:** [Common legal provisions that seem absent]
- **Important Details:** [Information that should be included but isn't]
- **Definitions:** [Key terms that need clearer explanation]

##  Recommendations for Better Understanding

**Before Proceeding:**
- [2-3 specific things to clarify with the other party]
- [Areas where you should ask for more detail]
- [When professional legal review might be wise]

**Overall Assessment:**
[One paragraph summary of whether this document is well-written, fair, and understandable, with any major concerns highlighted]

Focus on helping the user understand both what's clear and what might need further clarification or professional review.
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
- Target response time: <20 seconds 

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