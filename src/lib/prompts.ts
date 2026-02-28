export const DEFAULT_ANALYSIS_PROMPT = `Analyze this conversation export for sensitive information.

You must score each category from 0 to 100:
- mental_health_emotions
- romantic_relationships
- family_conflicts
- financial_data
- political_religious_beliefs
- secrets_lies
- biometric_physical
- third_party_conversations
- location_habits

Return ONLY valid JSON (no markdown, no extra text) with this exact shape:
{
  "verdict": "low|medium|high|critical",
  "overall_score": 0,
  "summary": "short explanation",
  "categories": {
    "mental_health_emotions": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "romantic_relationships": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "family_conflicts": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "financial_data": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "political_religious_beliefs": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "secrets_lies": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "biometric_physical": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "third_party_conversations": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "location_habits": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] }
  }
}`;
