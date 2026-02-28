# Prompt Memory

Use this file as baseline context for future prompts on this project.

## Product/UX decisions already validated

1. Keep the experience frictionless:
- No manual "Run analysis" button.
- Analysis starts automatically after upload, once model download is complete.

2. Model loading UX:
- Do not show loading UI on first page arrival.
- Show model download progress in a modal only after a valid upload triggers it.

3. Upload vs processing UI:
- While processing, replace the upload card with a status card indicating model evaluation is in progress.

4. Copy/content style:
- Public UI copy should be English-only.
- Remove unnecessary section titles and extra CTA friction.
- Keep layout visually continuous (no heavy segmented header/section cards).

5. Trust posture:
- Explicit "don't trust us blindly" message.
- State the project is open source and replicable.
- Keep visible GitHub link: `https://github.com/oliviernguyenquoc/ministral-3b-web`.

6. Prompt configurability:
- Hide the "Analysis prompt" section from end users.
- Keep internal default prompt in code.

7. Conversation-level analysis:
- Parse uploads into individual conversations (not one result per file).
- Keep `conversationId` and `conversationLabel` per analyzed item for future API delete actions.
- Results table includes source file + conversation metadata.

8. Risk rendering rules:
- Show category tags only when detection is above `none` (`risk !== none`).
- Keep category tag colors based on category score intensity.
- Verdict badge color is based on verdict enum, not score:
  - `low` = green
  - `medium` = amber
  - `high` = red
  - `critical` = strong rose/red

9. CSV export format:
- Include `SourceFile`, `ConversationId`, `ConversationLabel`.
- Include `Verdict`, `OverallScore`, per-category scores, `Summary`, and raw model output.

10. Fake data strategy:
- Use a realistic Gemini Activity HTML mock that mimics `MonActivité.html` card structure:
  - `outer-cell` -> `header-cell` -> 3 `content-cell` blocks.
- Keep some entries as standard prompt/response cards and some as “Gemini Canvas item created” cards.

## Quick instruction snippet for future prompts

Apply changes while preserving the existing frictionless flow:
- upload -> (if needed) model download modal -> automatic analysis -> results
- no manual run button
- clear processing status
- English UI copy
- open-source/replicable trust messaging with GitHub link

Use conversation-level outputs and preserve category-specific risk visualization behavior:
- parse by conversation
- keep conversation identifiers
- show only non-`none` category tags
- color verdict by `low|medium|high|critical`

## Current classification schema (must stay aligned in prompt + parser + UI)

Use these exact category keys in model JSON:
1. `mental_health_emotions` (Santé mentale/émotions)
2. `romantic_relationships` (Vie sentimentale)
3. `family_conflicts` (Conflits familiaux)
4. `financial_data` (Données financières personnelles)
5. `political_religious_beliefs` (Croyances/opinions politiques/religieuses)
6. `secrets_lies` (Secrets/mensonges)
7. `biometric_physical` (Données biométriques/physiques)
8. `third_party_conversations` (Conversations sur des tiers)
9. `location_habits` (Localisation/habitudes)

Model output contract:
- Return strict JSON only (no markdown prose)
- Top-level fields: `verdict`, `overall_score`, `summary`, `categories`
- Per-category fields: `score`, `risk`, `evidence`
