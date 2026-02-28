<script lang="ts">
	import JSZip from 'jszip';
	import { slide, fade } from 'svelte/transition';
	import { cubicOut } from 'svelte/easing';
	import { ministral } from '$lib/ministral';
	import { DEFAULT_ANALYSIS_PROMPT } from '$lib/prompts';

	let isModelReady = $state(false);
	let isProcessing = $state(false);
	let isDownloading = $state(false);
	let showLoadingModal = $state(false);
	let dragActive = $state(false);
	let modelLoadPromise: Promise<void> | null = null;

	let loadProgress = $state(0);
	let loadStatus = $state('Preparing...');

	let prompt = $state(DEFAULT_ANALYSIS_PROMPT);
	type CategoryKey =
		| 'mental_health_emotions'
		| 'romantic_relationships'
		| 'family_conflicts'
		| 'financial_data'
		| 'political_religious_beliefs'
		| 'secrets_lies'
		| 'biometric_physical'
		| 'third_party_conversations'
		| 'location_habits';
	type RiskLevel = 'none' | 'low' | 'medium' | 'high';
	type CategoryScore = { score: number; risk: RiskLevel; evidence: string[] };
	type ParsedAnalysis = {
		verdict: 'low' | 'medium' | 'high' | 'critical';
		overallScore: number;
		summary: string;
		categories: Record<CategoryKey, CategoryScore>;
	};
	type AnalysisResult = {
		sourceFile: string;
		conversationId: string;
		conversationLabel: string;
		textContent: string;
		response: string;
		parsed: ParsedAnalysis | null;
	};

	const CATEGORY_KEYS: CategoryKey[] = [
		'mental_health_emotions',
		'romantic_relationships',
		'family_conflicts',
		'financial_data',
		'political_religious_beliefs',
		'secrets_lies',
		'biometric_physical',
		'third_party_conversations',
		'location_habits'
	];
	const CATEGORY_LABELS: Record<CategoryKey, string> = {
		mental_health_emotions: 'Sante mentale / emotions',
		romantic_relationships: 'Relations amoureuses / vie sentimentale',
		family_conflicts: 'Problemes familiaux / conflits',
		financial_data: 'Donnees financieres personnelles',
		political_religious_beliefs: 'Croyances / opinions politiques / religieuses',
		secrets_lies: 'Secrets / mensonges',
		biometric_physical: 'Donnees biometriques / physiques',
		third_party_conversations: 'Conversations sur des tiers',
		location_habits: 'Localisation / habitudes'
	};
	let results = $state<AnalysisResult[]>([]);
	const ACCEPTED_UPLOAD_EXTENSIONS = ['.zip', '.json', '.html', '.htm'];

	let fileInput = $state<HTMLInputElement>();

	async function loadModel() {
		if (ministral.isLoaded) {
			isModelReady = true;
			return;
		}
		if (modelLoadPromise) return modelLoadPromise;

		isDownloading = true;
		modelLoadPromise = (async () => {
			try {
				await ministral.load((msg, percentage) => {
					loadStatus = msg;
					loadProgress = Math.round(percentage);
				});
				isModelReady = true;
			} catch (e) {
				console.error(e);
				alert('Model loading failed. Check the console for details.');
			} finally {
				isDownloading = false;
				showLoadingModal = false;
				modelLoadPromise = null;
			}
		})();

		return modelLoadPromise;
	}

	function triggerFileInput() {
		fileInput?.click();
	}

	function handleDrag(e: DragEvent) {
		e.preventDefault();
		e.stopPropagation();
		if (e.type === 'dragenter' || e.type === 'dragover') {
			dragActive = true;
		} else if (e.type === 'dragleave') {
			dragActive = false;
		}
	}

	async function handleDrop(e: DragEvent) {
		e.preventDefault();
		e.stopPropagation();
		dragActive = false;

		if (e.dataTransfer?.files && e.dataTransfer.files[0]) {
			await processFile(e.dataTransfer.files[0]);
		}
	}

	async function handleFileSelect(e: Event) {
		const target = e.target as HTMLInputElement;
		const file = target.files?.[0];
		if (file) {
			await processFile(file);
		}
	}

	function getFileExtension(fileName: string) {
		const match = fileName.toLowerCase().match(/\.[^.]+$/);
		return match?.[0] ?? '';
	}

	function isAcceptedUpload(fileName: string) {
		return ACCEPTED_UPLOAD_EXTENSIONS.includes(getFileExtension(fileName));
	}

	function clampScore(value: unknown) {
		const score = Number(value);
		if (!Number.isFinite(score)) return 0;
		return Math.max(0, Math.min(100, Math.round(score)));
	}

	function deriveRiskFromScore(score: number): RiskLevel {
		if (score <= 0) return 'none';
		if (score < 35) return 'low';
		if (score < 70) return 'medium';
		return 'high';
	}

	function normalizeRisk(value: unknown, score: number): RiskLevel {
		if (value === 'none' || value === 'low' || value === 'medium' || value === 'high') {
			return value;
		}
		return deriveRiskFromScore(score);
	}

	function normalizeVerdict(value: unknown, overallScore: number): ParsedAnalysis['verdict'] {
		if (value === 'low' || value === 'medium' || value === 'high' || value === 'critical') {
			return value;
		}
		if (overallScore < 30) return 'low';
		if (overallScore < 60) return 'medium';
		if (overallScore < 85) return 'high';
		return 'critical';
	}

	function extractJSONPayload(raw: string): unknown | null {
		const trimmed = raw.trim();
		if (!trimmed) return null;

		try {
			return JSON.parse(trimmed);
		} catch {
			// Continue with fallback parsing.
		}

		const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
		if (fencedMatch?.[1]) {
			try {
				return JSON.parse(fencedMatch[1].trim());
			} catch {
				// Continue with next fallback.
			}
		}

		const start = trimmed.indexOf('{');
		const end = trimmed.lastIndexOf('}');
		if (start >= 0 && end > start) {
			try {
				return JSON.parse(trimmed.slice(start, end + 1));
			} catch {
				return null;
			}
		}

		return null;
	}

	function parseAnalysis(raw: string): ParsedAnalysis | null {
		const payload = extractJSONPayload(raw);
		if (!payload || typeof payload !== 'object') return null;

		const data = payload as Record<string, unknown>;
		const categoriesPayload =
			typeof data.categories === 'object' && data.categories ? (data.categories as Record<string, unknown>) : {};

		const categories = {} as Record<CategoryKey, CategoryScore>;
		for (const key of CATEGORY_KEYS) {
			const entry =
				typeof categoriesPayload[key] === 'object' && categoriesPayload[key]
					? (categoriesPayload[key] as Record<string, unknown>)
					: {};
			const score = clampScore(entry.score);
			categories[key] = {
				score,
				risk: normalizeRisk(entry.risk, score),
				evidence: Array.isArray(entry.evidence)
					? entry.evidence.map((item) => String(item)).filter(Boolean).slice(0, 2)
					: []
			};
		}

		const average =
			Math.round(CATEGORY_KEYS.reduce((sum, key) => sum + categories[key].score, 0) / CATEGORY_KEYS.length) || 0;
		const overallScore = clampScore(data.overall_score ?? average);

		return {
			verdict: normalizeVerdict(data.verdict, overallScore),
			overallScore,
			summary: String(data.summary ?? '').trim(),
			categories
		};
	}

	function scoreBadgeClass(score: number) {
		if (score >= 75) return 'border-red-300 bg-red-50 text-red-700';
		if (score >= 45) return 'border-amber-300 bg-amber-50 text-amber-700';
		if (score > 0) return 'border-yellow-300 bg-yellow-50 text-yellow-700';
		return 'border-emerald-300 bg-emerald-50 text-emerald-700';
	}

	function verdictBadgeClass(verdict: ParsedAnalysis['verdict']) {
		if (verdict === 'critical') return 'border-rose-400 bg-rose-100 text-rose-800';
		if (verdict === 'high') return 'border-red-300 bg-red-50 text-red-700';
		if (verdict === 'medium') return 'border-amber-300 bg-amber-50 text-amber-700';
		return 'border-emerald-300 bg-emerald-50 text-emerald-700';
	}

	function normalizeExtractedText(value: string) {
		return value.replace(/\s+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
	}

	async function buildImageFromText(rawText: string) {
		const maxChars = 18000;
		const normalizedText = rawText.trim();
		const truncated =
			normalizedText.length > maxChars
				? `${normalizedText.slice(0, maxChars)}\n\n[Text truncated for analysis]`
				: normalizedText;
		const source = truncated || '[Empty file]';

		return new Promise<string>((resolve) => {
			const canvas = document.createElement('canvas');
			const ctx = canvas.getContext('2d');
			if (!ctx) {
				resolve('');
				return;
			}

			const width = 1200;
			const padding = 40;
			const lineHeight = 28;
			const font = '20px "Space Grotesk", "Avenir Next", sans-serif';

			ctx.font = font;

			const lines: string[] = [];
			for (const paragraph of source.split('\n')) {
				const words = paragraph.split(/\s+/).filter(Boolean);
				if (!words.length) {
					lines.push('');
					continue;
				}

				let line = words[0];
				for (let i = 1; i < words.length; i++) {
					const candidate = `${line} ${words[i]}`;
					if (ctx.measureText(candidate).width > width - padding * 2) {
						lines.push(line);
						line = words[i];
					} else {
						line = candidate;
					}
				}
				lines.push(line);
			}

			const maxLines = 180;
			const finalLines =
				lines.length > maxLines
					? [...lines.slice(0, maxLines), '[... contenu tronque ...]']
					: lines;
			const height = Math.max(420, padding * 2 + finalLines.length * lineHeight + 30);

			canvas.width = width;
			canvas.height = height;

			ctx.fillStyle = '#ffffff';
			ctx.fillRect(0, 0, width, height);
			ctx.fillStyle = '#111827';
			ctx.font = font;
			ctx.textBaseline = 'top';

			finalLines.forEach((line, index) => {
				ctx.fillText(line, padding, padding + index * lineHeight, width - padding * 2);
			});

			resolve(canvas.toDataURL('image/png'));
		});
	}

	function extractConversationsFromHtml(fileName: string, html: string) {
		const parser = new DOMParser();
		const doc = parser.parseFromString(html, 'text/html');
		const scripts = doc.querySelectorAll('script, style, noscript');
		scripts.forEach((node) => node.remove());

		const cards = Array.from(doc.querySelectorAll('.outer-cell'));
		if (!cards.length) {
			const wholeText = normalizeExtractedText(doc.body?.textContent ?? '');
			return wholeText
				? [
						{
							conversationId: `${fileName}#conv-1`,
							conversationLabel: 'Conversation 1',
							textContent: wholeText
						}
					]
				: [];
		}

		return cards
			.map((card, index) => {
				const title = normalizeExtractedText(
					(card.querySelector('.mdl-typography--title')?.textContent ?? '').replace(/\s+/g, ' ')
				);
				const textContent = normalizeExtractedText((card as HTMLElement).innerText || card.textContent || '');
				if (!textContent) return null;
				return {
					conversationId: `${fileName}#conv-${index + 1}`,
					conversationLabel: title ? `${title} #${index + 1}` : `Conversation ${index + 1}`,
					textContent
				};
			})
			.filter(Boolean) as Array<{ conversationId: string; conversationLabel: string; textContent: string }>;
	}

	function extractConversationsFromJson(fileName: string, rawJson: string) {
		let parsed: unknown;
		try {
			parsed = JSON.parse(rawJson);
		} catch {
			return [
				{
					conversationId: `${fileName}#conv-1`,
					conversationLabel: 'Conversation 1',
					textContent: rawJson
				}
			];
		}

		const container = parsed as Record<string, unknown>;
		let entries: unknown[] = [];
		if (Array.isArray(parsed)) {
			entries = parsed;
		} else if (Array.isArray(container.conversations)) {
			entries = container.conversations;
		} else if (Array.isArray(container.chats)) {
			entries = container.chats;
		} else if (Array.isArray(container.items)) {
			entries = container.items;
		} else {
			entries = [parsed];
		}

		const chunks = entries
			.map((entry, index) => {
				const obj = entry as Record<string, unknown>;
				const candidateId =
					typeof obj?.id === 'string'
						? obj.id
						: typeof obj?.conversation_id === 'string'
							? obj.conversation_id
							: typeof obj?.uuid === 'string'
								? obj.uuid
								: `${fileName}#conv-${index + 1}`;
				const candidateTitle =
					typeof obj?.title === 'string'
						? obj.title
						: typeof obj?.name === 'string'
							? obj.name
							: `Conversation ${index + 1}`;
				const textContent = normalizeExtractedText(JSON.stringify(entry, null, 2));
				if (!textContent) return null;
				return {
					conversationId: candidateId,
					conversationLabel: normalizeExtractedText(candidateTitle) || `Conversation ${index + 1}`,
					textContent
				};
			})
			.filter(Boolean) as Array<{ conversationId: string; conversationLabel: string; textContent: string }>;

		if (chunks.length) return chunks;
		return [
			{
				conversationId: `${fileName}#conv-1`,
				conversationLabel: 'Conversation 1',
				textContent: normalizeExtractedText(JSON.stringify(parsed, null, 2))
			}
		];
	}

	async function processTextFile(fileName: string, content: string) {
		const extension = getFileExtension(fileName);
		const baseName = fileName.split('/').pop() || fileName;

		let chunks: Array<{ conversationId: string; conversationLabel: string; textContent: string }> = [];
		if (extension === '.json') {
			chunks = extractConversationsFromJson(baseName, content);
		} else if (extension === '.html' || extension === '.htm') {
			chunks = extractConversationsFromHtml(baseName, content);
		} else {
			const fallback = normalizeExtractedText(content);
			if (fallback) {
				chunks = [
					{
						conversationId: `${baseName}#conv-1`,
						conversationLabel: 'Conversation 1',
						textContent: fallback
					}
				];
			}
		}

		for (const chunk of chunks) {
			if (!chunk.textContent.trim()) continue;
			results.push({
				sourceFile: baseName,
				conversationId: chunk.conversationId,
				conversationLabel: chunk.conversationLabel,
				textContent: chunk.textContent,
				response: '',
				parsed: null
			});
		}
	}

	async function processFile(file: File) {
		if (!isAcceptedUpload(file.name)) {
			alert('Please upload a valid ZIP, JSON, or HTML file.');
			return;
		}

		// Start model download as soon as a valid file is uploaded.
		showLoadingModal = !isModelReady;
		const ensureModelReady = isModelReady ? Promise.resolve() : loadModel();
		results = [];

		try {
			const extension = getFileExtension(file.name);

			if (extension === '.zip') {
				const zip = new JSZip();
				const zipData = await zip.loadAsync(file);
				const textPromises: Promise<void>[] = [];

				zipData.forEach((relativePath, zipEntry) => {
					const isMacArtifact = /^__MACOSX|\/\._/.test(relativePath);
					const isTextFile = /\.(json|html|htm)$/i.test(relativePath);

					if (isTextFile && !zipEntry.dir && !isMacArtifact) {
						textPromises.push(
							zipEntry
								.async('text')
								.then(async (text) => processTextFile(relativePath, text))
								.catch((error) => {
									console.error(`Impossible de lire ${relativePath}:`, error);
								})
						);
					}
				});

				if (textPromises.length) {
					await Promise.all(textPromises);
				}
			} else if (extension === '.json' || extension === '.html' || extension === '.htm') {
				const raw = await file.text();
				await processTextFile(file.name, raw);
			}

			if (!results.length) {
				alert('No usable content was found in this file.');
				return;
			}

			await ensureModelReady;
			if (isModelReady && results.length) {
				await runInference();
			}
		} catch (e) {
			console.error(e);
			alert('Import failed. Check the console for details.');
		}
	}

	async function runInference() {
		if (!results.length || !isModelReady) return;

		isProcessing = true;

		try {
			for (let i = 0; i < results.length; i++) {
				const result = results[i];
				const imageSrc = await buildImageFromText(result.textContent);
				if (!imageSrc) {
					results[i].response = 'Error: unable to prepare content for analysis.';
					continue;
				}
				const img = new Image();
				img.src = imageSrc;

				await new Promise<void>((resolve) => {
					img.onload = async () => {
						try {
							const finalText = await ministral.generate(img, prompt, (updatedText) => {
								results[i].response = updatedText;
							});
							results[i].parsed = parseAnalysis(finalText);
						} catch (e) {
							results[i].response = 'Error: ' + e;
							results[i].parsed = null;
						}
						resolve();
					};
					img.onerror = () => {
						results[i].response = 'Error: unable to load content for analysis.';
						results[i].parsed = null;
						resolve();
					};
				});
			}
		} catch (e) {
			console.error('Error during batch processing: ' + e);
		} finally {
			isProcessing = false;
		}
	}

	function downloadCSV() {
		if (!results.length) return;

		const escapeCSV = (str: string) => {
			if (str.includes(',') || str.includes('"') || str.includes('\n')) {
				return `"${str.replace(/"/g, '""')}"`;
			}
			return str;
		};

		const headers = [
			'SourceFile',
			'ConversationId',
			'ConversationLabel',
			'Verdict',
			'OverallScore',
			...CATEGORY_KEYS.map((key) => CATEGORY_LABELS[key]),
			'Summary',
			'Raw'
		];
		const csvRows = [
			headers.join(','),
			...results.map((r) => {
				const parsed = r.parsed;
				return [
					escapeCSV(r.sourceFile),
					escapeCSV(r.conversationId),
					escapeCSV(r.conversationLabel),
					escapeCSV(parsed?.verdict ?? ''),
					escapeCSV(String(parsed?.overallScore ?? '')),
					...CATEGORY_KEYS.map((key) => escapeCSV(String(parsed?.categories[key].score ?? ''))),
					escapeCSV(parsed?.summary ?? ''),
					escapeCSV(r.response)
				].join(',');
			})
		];

		const csvContent = csvRows.join('\n');
		const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
		const url = URL.createObjectURL(blob);

		const link = document.createElement('a');
		link.href = url;
		link.download = `vision-results-${new Date().getTime()}.csv`;
		link.click();

		URL.revokeObjectURL(url);
	}

	const hasResponses = $derived(results.some((r) => r.response));
</script>

<div
	class="min-h-screen bg-[radial-gradient(circle_at_10%_10%,#fef3c7_0%,#fff7ed_35%,#f8fafc_75%)] text-slate-900 selection:bg-amber-200"
	style="font-family: 'Space Grotesk', 'Avenir Next', 'Segoe UI', sans-serif;"
>
	<div class="mx-auto max-w-7xl px-6 py-10 sm:py-14">
		<header
			class="relative mb-4 overflow-hidden px-1 py-2 sm:py-4"
		>
			<div class="relative">
				<h1
					class="max-w-5xl text-3xl leading-tight font-black tracking-tight text-slate-900 sm:text-5xl"
				>
					See in 30 seconds what your ChatGPT conversations reveal about you.
				</h1>
				<p class="mt-5 max-w-4xl text-base leading-relaxed text-slate-600 sm:text-lg">
					You may have shared personal information without realizing it: health, money, work, and
					private life details.
				</p>
				<p class="mt-3 max-w-3xl text-sm leading-relaxed text-slate-500 sm:text-base">
					This page helps you review your conversation exports and quickly spot what might be too
					sensitive.
				</p>
				<p class="mt-3 text-xs font-medium text-slate-500">
					Works best on Google Chrome (desktop)
				</p>
			</div>
		</header>

		{#if isModelReady}
			<div
				transition:fade
				class="mb-8 flex items-center justify-center gap-2 rounded-full border border-green-200 bg-green-50 py-1.5 text-sm font-medium text-green-700"
			>
				<span class="relative flex h-2.5 w-2.5">
					<span
						class="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75"
					></span>
				<span class="relative inline-flex h-2.5 w-2.5 rounded-full bg-green-500"></span>
				</span>
				Analysis ready
			</div>
		{/if}

		<!-- SECTION: Interaction -->
		<section
			id="local-scan"
			class="mb-8 transition-opacity duration-500 {isModelReady
				? 'opacity-100'
				: 'opacity-100'}"
		>
			<div class="p-6">
				<!-- Styled Dropzone -->
				<div class="space-y-6">
					{#if isProcessing}
						<div class="rounded-xl border border-amber-200 bg-amber-50 px-6 py-8 text-center">
							<div class="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm">
								<svg
									class="h-5 w-5 animate-spin text-amber-600"
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
								>
									<circle
										class="opacity-25"
										cx="12"
										cy="12"
										r="10"
										stroke="currentColor"
										stroke-width="4"
									></circle>
									<path
										class="opacity-75"
										fill="currentColor"
										d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
									></path>
								</svg>
							</div>
							<p class="mt-4 text-sm font-semibold text-amber-900">
								Model is evaluating your conversations...
							</p>
							<p class="mt-1 text-xs text-amber-700">
								{results.length} conversation(s) in progress.
							</p>
						</div>
					{:else}
						<div>
							<label for="dropzone" class="mb-2 block text-sm font-medium text-gray-700"
								>Upload conversations (ZIP, JSON, HTML)</label
							>
							<div
								role="button"
								tabindex="0"
								onclick={triggerFileInput}
								ondragenter={handleDrag}
								ondragleave={handleDrag}
								ondragover={handleDrag}
								ondrop={handleDrop}
								onkeydown={(e) => {
									if (e.key === 'Enter' || e.key === ' ') {
										triggerFileInput();
									}
								}}
								class="group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 transition-all duration-200 ease-in-out
								{dragActive
									? 'border-amber-500 bg-amber-50/70'
									: 'border-gray-300 bg-gray-50 hover:border-amber-400 hover:bg-white'}"
							>
								<input
									id="zip-upload"
									type="file"
									accept=".zip,.json,.html,.htm"
									bind:this={fileInput}
									onchange={handleFileSelect}
									class="hidden"
								/>
								<div
									class="flex h-12 w-12 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-gray-900/5 transition-transform group-hover:scale-110 group-hover:text-amber-600"
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="1.5"
										stroke="currentColor"
										class="h-6 w-6 text-gray-500 group-hover:text-amber-600"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
										/>
									</svg>
								</div>
								<div class="mt-4 flex text-sm text-gray-600">
									<span class="font-semibold text-amber-700 hover:text-amber-600"
										>Click to upload</span
									>
									<span class="pl-1">or drag and drop a ZIP, JSON, or HTML file</span>
								</div>
								<p class="mt-1 text-xs text-gray-500">Accepted formats: ZIP, JSON, HTML</p>
							</div>
						</div>
					{/if}
				</div>

				<!-- Results Table -->
				{#if results.length > 0}
					<div class="animate-in fade-in slide-in-from-bottom-4 mt-10 duration-500">
						<div class="mb-4 flex items-center justify-between">
							<h3 class="text-base leading-6 font-semibold text-gray-900">
								Results <span
									class="ml-2 rounded-md bg-gray-100 px-2 py-0.5 text-xs text-gray-600"
									>{results.length}</span
								>
							</h3>
							<button
								onclick={downloadCSV}
								disabled={!hasResponses}
								class="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 disabled:opacity-50"
							>
								<svg
									xmlns="http://www.w3.org/2000/svg"
									viewBox="0 0 20 20"
									fill="currentColor"
									class="h-4 w-4"
								>
									<path
										fill-rule="evenodd"
										d="M4.5 2A1.5 1.5 0 003 3.5v13A1.5 1.5 0 004.5 18h11a1.5 1.5 0 001.5-1.5V7.621a1.5 1.5 0 00-.44-1.06l-4.12-4.122A1.5 1.5 0 0011.378 2H4.5zm2.25 8.5a.75.75 0 000 1.5h6.5a.75.75 0 000-1.5h-6.5zm0 3a.75.75 0 000 1.5h6.5a.75.75 0 000-1.5h-6.5z"
										clip-rule="evenodd"
									/>
								</svg>
								Download CSV
							</button>
						</div>

						<div class="overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm">
							<table class="min-w-full divide-y divide-gray-200">
								<thead class="bg-gray-50">
									<tr>
										<th
											scope="col"
											class="px-6 py-3 text-left text-xs font-medium tracking-wider text-gray-500 uppercase"
											>Source</th
										>
										<th
											scope="col"
											class="px-6 py-3 text-left text-xs font-medium tracking-wider text-gray-500 uppercase"
											>Conversation</th
										>
										<th
											scope="col"
											class="px-6 py-3 text-left text-xs font-medium tracking-wider text-gray-500 uppercase"
											>Analysis</th
										>
									</tr>
								</thead>
									<tbody class="divide-y divide-gray-200 bg-white">
										{#each results as result}
											<tr class="transition-colors hover:bg-gray-50/50">
												<td class="px-6 py-4 whitespace-nowrap">
													<span class="font-mono text-xs text-gray-500">{result.sourceFile}</span>
												</td>
												<td class="px-6 py-4 whitespace-nowrap">
													<div class="space-y-1">
														<p class="text-xs font-medium text-slate-700">{result.conversationLabel}</p>
														<p class="font-mono text-[11px] text-slate-500">{result.conversationId}</p>
													</div>
												</td>
												<td class="px-6 py-4 text-sm text-gray-700">
												{#if result.response}
													{#if result.parsed}
														<div class="space-y-3">
																<div class="flex flex-wrap items-center gap-2">
																	<span
																		class="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold uppercase {verdictBadgeClass(result.parsed.verdict)}"
																	>
																		{result.parsed.verdict}
																	</span>
																<span class="text-xs font-medium text-slate-500">
																	Overall {result.parsed.overallScore}/100
																</span>
															</div>
															<div class="flex flex-wrap gap-2">
																{#each CATEGORY_KEYS as category}
																	{#if result.parsed.categories[category].risk !== 'none'}
																		<span
																			class="inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium {scoreBadgeClass(result.parsed.categories[category].score)}"
																		>
																			{CATEGORY_LABELS[category]} {result.parsed.categories[category].score} ({result.parsed.categories[category].risk})
																		</span>
																	{/if}
																{/each}
															</div>
															{#each CATEGORY_KEYS as category}
																{#if result.parsed.categories[category].evidence.length > 0}
																	<div class="space-y-1">
																		<p class="text-xs font-semibold text-slate-500">{CATEGORY_LABELS[category]} evidence</p>
																		{#each result.parsed.categories[category].evidence as excerpt}
																			<p class="text-xs leading-relaxed text-slate-600">&quot;{excerpt}&quot;</p>
																		{/each}
																	</div>
																{/if}
															{/each}
															{#if result.parsed.summary}
																<p class="leading-relaxed text-slate-700">{result.parsed.summary}</p>
															{/if}
														</div>
													{:else}
														<p class="leading-relaxed">{result.response}</p>
													{/if}
												{:else if isProcessing}
													<span class="inline-flex items-center gap-2 text-gray-400">
														<span
															class="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 delay-0"
														></span>
														<span
															class="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 delay-150"
														></span>
														<span
															class="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400 delay-300"
														></span>
													</span>
												{:else}
													<span class="text-xs text-gray-300 italic">Pending</span>
												{/if}
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					</div>
				{/if}
			</div>
		</section>

		<section class="mb-10 space-y-4 text-sm leading-relaxed text-slate-700">
			<p>
				Your files stay on your device during analysis. Nothing is sent to our servers.
			</p>
			<p>
				How it works: upload a ZIP, JSON, or HTML file, automatic analysis starts, then review the
				flagged sensitive passages.
			</p>
			<p class="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-amber-900">
				Don&apos;t trust us blindly. This project is open source and fully replicable.
				<a
					href="https://github.com/oliviernguyenquoc/ministral-3b-web"
					target="_blank"
					rel="noopener noreferrer"
					class="ml-1 font-semibold underline decoration-amber-500 underline-offset-2 hover:text-amber-700"
				>
					Review the code on GitHub
				</a>
				and run it yourself.
			</p>
			<p class="font-medium text-slate-900">
				In the cloud, nothing is truly secret. Take 30 seconds to check what you shared.
			</p>
		</section>
	</div>
</div>

{#if showLoadingModal && isDownloading}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/45 px-4"
		transition:fade={{ duration: 150 }}
	>
		<div
			class="w-full max-w-lg overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_20px_70px_-30px_rgba(15,23,42,0.55)]"
			transition:slide={{ duration: 220, easing: cubicOut, axis: 'y' }}
		>
			<div class="border-b border-slate-100 bg-slate-50 px-6 py-4">
				<h2 class="text-base font-semibold text-slate-900">Preparing analysis</h2>
			</div>
			<div class="p-6">
				<p class="mb-2 text-sm text-slate-600">Downloading model files (~3GB).</p>
				<p class="mb-5 text-xs text-slate-500">
					This step is only required for first-time use in this browser.
				</p>
				<div class="mb-4 h-3 overflow-hidden rounded-full bg-slate-200">
					<div
						class="h-full rounded-full bg-amber-500 transition-all duration-300 ease-out"
						style="width: {loadProgress}%"
					></div>
				</div>
				<div class="flex items-center justify-between gap-4 text-sm text-slate-700">
					<span class="truncate">{loadStatus}</span>
					<span class="font-mono font-semibold">{loadProgress}%</span>
				</div>
			</div>
		</div>
	</div>
{/if}
