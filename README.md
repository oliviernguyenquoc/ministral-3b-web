## Elevator pitch

Find out what your ChatGPT conversations reveal about you in 30 seconds.

"Deal signed with the Department of War." Sam Altman's latest tweet confirms the shift: your data now lives in the cloud infrastructure of a military giant.

Have you shared secrets with ChatGPT that you'd regret if your boss, the military, or a hacker read them? AmIScrewed.eu scans your history and reveals overly private details you shared without thinking.
Why is it safe?

Unlike OpenAI, your data never leaves your computer. Thanks to WebGPU, our AI (Ministral) runs directly in your browser. Nothing is sent to our servers.

How does it work?

- Copy your conversations or upload your archive.
- Ultra-fast local analysis powered by your own graphics card.
- Unfiltered verdict: "Highly private conversation: Your exchange from 03/12 contains sensitive health details."

In the cloud, nothing is truly secret. Take back control.
→ Try it for free on AmIScrewed.eu

## How to import AI conversation

See: [how_to_import_ai_conversation.md]

## Prompt memory

For future prompt context and validated UX decisions, see: [PROMPT_MEMORY.md]

## Target Audience And Language

This product is aimed at non-technical users first.
- The homepage should read like a privacy awareness page, not a developer tool.
- Wording should stay simple, concrete, and reassuring for everyday users.
- The upload and results flow must be visible early on the page.

All public-facing product content must be in English.
- UI labels, headings, helper text, and calls-to-action are English-only.
- README and product copy should avoid mixing French and English.

## Which type of conversation we detect

- Santé mentale/émotions
- Vie très personnelle: Relations amoureuses/vie sentimentale
- Problèmes familiaux/conflits
- Données financières personnelles
- Croyances/opinions politiques/religieuses
- Secrets/mensonges
- Données biométriques/physiques
- Conversations sur des tiers
- Localisation/habitudes

## How to run the project

Run batch image analysis using the [Ministral-3-3B-Instruct-2512-ONNX](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-ONNX) model in the browser.

Upload a ZIP file containing multiple images, and each image will be run against the provided prompt. Results are displayed in a tabular format with the option to download the results as a CSV file.

This project runs entirely in the browser using WebGPU. No images are sent to a server; the model download occurs once and is cached locally.

### Prerequisites

You will need a browser with **WebGPU support** enabled (e.g. recent versions of Chrome).

### Development

1.  Clone the repository and install dependencies:

    ```bash
    npm install
    ```

2.  Start the development server:

    ```bash
    npm run dev
    ```

3.  Open `http://localhost:5173` in your browser.

The first time you load the model, the application will download approximately **3GB** of model weights. Subsequent loads will use the browser cache.

Performance (tokens per second) varies based on your device's GPU capabilities.
