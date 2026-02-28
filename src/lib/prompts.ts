export const DEFAULT_ANALYSIS_PROMPT = `
You MUST output ONLY a single valid JSON object.
No markdown, no code fences, no commentary.
Use double quotes for all keys and strings.
No trailing commas.

Analyze this conversation export for sensitive information.

Goal: detect *exploitable* private disclosures (not just generic PII). Be strict about evidence: only score high when the conversation contains concrete, personal, identifiable details.

Score each category from 0 to 100:
- health_vulnerability
- personal_relationships_conflicts
- financial_vulnerability
- risky_confessions_secrets
- location_routine_tracking

Scoring guidelines (apply to each category):
- 0–10: vague / generic / no personal details
- 11–30: mild personal mention, non-identifying, low impact
- 31–60: clear personal disclosure with some specifics (time, amounts, actors, repeated context)
- 61–85: detailed and sensitive disclosure, could plausibly harm the user if leaked
- 86–100: highly actionable/exploitable (precise details, repeated patterns, strong identifiers, explicit admissions)

Category definitions:
1) health_vulnerability:
   - includes: diagnosis, symptoms details, treatment/medication, therapy details, addiction/self-harm ideation
   - exclude: generic emotions without health context ("I feel stressed")
2) personal_relationships_conflicts:
   - includes: infidelity, divorce, abuse, major conflicts, family disputes, intimate details, relationship secrets
   - exclude: generic statements without context ("we argued")
3) financial_vulnerability:
   - includes: income/salary, debts, loans, bankruptcy, fraud, money amounts tied to the user, financial hardship details
   - exclude: generic budgeting advice without user specifics
4) risky_confessions_secrets:
   - includes: explicit confessions that could cause legal/reputation/professional harm (fraud, cheating, serious misconduct), high-stakes secrets
   - exclude: harmless secrets ("I like pineapple on pizza")
5) location_routine_tracking:
   - includes: precise locations, home/work info, commute routes, schedules, repeated habits ("every day at 18:00"), travel plans with dates/places
   - exclude: vague location mentions ("in my city") without routine/precision

Evidence rules:
- Provide 0–3 short quotes per category (each <= 160 chars).
- Quotes must be verbatim snippets from the conversation.
- If no strong evidence, keep evidence as an empty array and score <= 10.

Overall scoring:
- overall_score should reflect the maximum + breadth (multiple categories with medium/high increases overall_score).
- verdict mapping:
  - low: 0–24
  - medium: 25–49
  - high: 50–74
  - critical: 75–100

Return ONLY valid JSON (no markdown, no extra text) with this exact shape:
{
  "verdict": "low|medium|high|critical",
  "overall_score": 0,
  "summary": "short explanation (1-2 sentences)",
  "categories": {
    "health_vulnerability": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "personal_relationships_conflicts": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "financial_vulnerability": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "risky_confessions_secrets": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] },
    "location_routine_tracking": { "score": 0, "risk": "none|low|medium|high", "evidence": ["..."] }
  }
}`;
