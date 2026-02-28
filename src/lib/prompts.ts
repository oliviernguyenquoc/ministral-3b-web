export const DEFAULT_ANALYSIS_PROMPT = `
Analyze the conversation for exploitable privacy disclosures.
Be conservative and never invent facts.

Score (0-100) these categories:
- health_vulnerability
- personal_relationships_conflicts
- financial_vulnerability
- risky_confessions_secrets
- location_routine_tracking

Guidance:
- 0-20: vague or generic
- 21-50: clear personal disclosure with limited specificity
- 51-75: detailed sensitive information
- 76-100: highly actionable/exploitable details

Return ONLY valid JSON:
{
  "verdict": "low|medium|high|critical",
  "overall_score": 0,
  "summary": "short summary",
  "categories": {
    "health_vulnerability": { "score": 0, "risk": "none|low|medium|high", "evidence": [] },
    "personal_relationships_conflicts": { "score": 0, "risk": "none|low|medium|high", "evidence": [] },
    "financial_vulnerability": { "score": 0, "risk": "none|low|medium|high", "evidence": [] },
    "risky_confessions_secrets": { "score": 0, "risk": "none|low|medium|high", "evidence": [] },
    "location_routine_tracking": { "score": 0, "risk": "none|low|medium|high", "evidence": [] }
  }
}

Rules:
- At most 1 short verbatim quote per category.
- If no evidence: evidence must be [] and score <= 10.
- Summary max 20 words.
- verdict mapping: low 0-24, medium 25-49, high 50-74, critical 75-100.

`;
