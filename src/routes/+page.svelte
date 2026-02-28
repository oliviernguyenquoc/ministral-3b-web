<script lang="ts">
	import JSZip from 'jszip';
	import { onMount } from 'svelte';
	import { slide, fade } from 'svelte/transition';
	import { cubicOut } from 'svelte/easing';
	import { ministral, type GenerationMetrics } from '$lib/ministral';
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
		| 'health_vulnerability'
		| 'personal_relationships_conflicts'
		| 'financial_vulnerability'
		| 'risky_confessions_secrets'
		| 'location_routine_tracking';
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
		'health_vulnerability',
		'personal_relationships_conflicts',
		'financial_vulnerability',
		'risky_confessions_secrets',
		'location_routine_tracking'
	];
	const CATEGORY_LABELS: Record<CategoryKey, string> = {
		health_vulnerability: 'Health vulnerability',
		personal_relationships_conflicts: 'Personal relationships/conflicts',
		financial_vulnerability: 'Financial vulnerability',
		risky_confessions_secrets: 'Risky confessions/secrets',
		location_routine_tracking: 'Location/routine tracking'
	};
	let results = $state<AnalysisResult[]>([]);
	let activeInferenceIndex = $state<number | null>(null);
	let completedAnalyses = $state(0);
	let totalAnalyses = $state(0);
	const ACCEPTED_UPLOAD_EXTENSIONS = ['.zip', '.json', '.html', '.htm'];

	let fileInput = $state<HTMLInputElement>();
	let runtimeStatusMessage = $state('Detecting WebGPU...');

	onMount(async () => {
		const runtime = await ministral.inspectRuntime();
		runtimeStatusMessage = runtime.webgpuSupported
			? `WebGPU active${runtime.adapterName ? ` (${runtime.adapterName})` : ''}`
			: 'WebGPU not detected. Inference will be very slow or unavailable.';
	});

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
				const message = e instanceof Error ? e.message : 'Unknown error';
				alert(`Model loading failed. ${message}`);
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

	function prepareTextForAnalysis(rawText: string) {
		const maxChars = 3500;
		const normalizedText = rawText.trim();
		if (!normalizedText) return '[Empty file]';

		if (normalizedText.length <= maxChars) return normalizedText;

		const headChars = 2200;
		const tailChars = 1000;
		const truncated = `${normalizedText.slice(0, headChars)}\n\n[...middle truncated for faster local analysis...]\n\n${normalizedText.slice(-tailChars)}`;
		return truncated || '[Empty file]';
	}

	function cleanGeminiActivityText(raw: string) {
		const lines = raw
			.split('\n')
			.map((line) => line.trim())
			.filter(Boolean);

		const filtered = lines.filter((line) => {
			if (/^Produits\s*:/i.test(line)) return false;
			if (/^Que fait cette information ici/i.test(line)) return false;
			if (/^Cette activite a ete enregistree/i.test(line)) return false;
			if (/^Vous pouvez controler ces parametres/i.test(line)) return false;
			if (/^Applications Gemini$/i.test(line)) return false;
			if (/googleusercontent\.com\/gemini_canvas_content/i.test(line)) return false;
			if (/\bCET$/i.test(line)) return false;
			return true;
		});

		const promptLine = filtered.find((line) => /^Prompt\s*:/i.test(line));
		const prompt = promptLine?.replace(/^Prompt\s*:\s*/i, '').trim() ?? '';
		const assistantLines = filtered.filter((line) => line !== promptLine);

		const compactAssistant = normalizeExtractedText(assistantLines.join('\n'));
		if (prompt && compactAssistant) {
			return `user: ${prompt}\n\nassistant: ${compactAssistant}`;
		}
		if (prompt) return `user: ${prompt}`;
		return compactAssistant;
	}

	function extractConversationsFromHtml(fileName: string, html: string) {
		const parser = new DOMParser();
		const doc = parser.parseFromString(html, 'text/html');
		const scripts = doc.querySelectorAll('script, style, noscript');
		scripts.forEach((node) => node.remove());

		const cards = Array.from(doc.querySelectorAll('.outer-cell'));
		if (!cards.length) {
			// Fallback for exported activity formats where selectors may differ.
			const blockMatches = html.match(/<div class="outer-cell[\s\S]*?<\/div>\s*<\/div>\s*<\/div>/gi) ?? [];
			if (blockMatches.length) {
				const regexChunks = blockMatches
					.map((block, index) => {
						const blockDoc = parser.parseFromString(block, 'text/html');
						const textContent = normalizeExtractedText(
							cleanGeminiActivityText(blockDoc.body?.textContent ?? '')
						);
						if (!textContent) return null;
						return {
							conversationId: `${fileName}#conv-${index + 1}`,
							conversationLabel: `Conversation ${index + 1}`,
							textContent
						};
					})
					.filter(Boolean) as Array<{
					conversationId: string;
					conversationLabel: string;
					textContent: string;
				}>;
				if (regexChunks.length) return regexChunks;
			}

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
				const bodyCells = Array.from(
					card.querySelectorAll('.content-cell.mdl-cell.mdl-typography--body-1:not(.mdl-typography--text-right)')
				);
				const primaryText = bodyCells
					.map((cell) => cleanGeminiActivityText((cell as HTMLElement).innerText || cell.textContent || ''))
					.filter(Boolean)
					.join('\n\n');
				const fallbackText = cleanGeminiActivityText(
					(card as HTMLElement).innerText || card.textContent || ''
				);
				const textContent = normalizeExtractedText(primaryText || fallbackText);
				if (!textContent) return null;
				return {
					conversationId: `${fileName}#conv-${index + 1}`,
					conversationLabel: title ? `${title} #${index + 1}` : `Conversation ${index + 1}`,
					textContent
				};
			})
			.filter(Boolean) as Array<{ conversationId: string; conversationLabel: string; textContent: string }>;
	}

	type ConversationTurn = {
		role: string;
		text: string;
		createdAt: number | null;
		order: number;
	};

	function normalizeRole(role: unknown) {
		if (typeof role !== 'string') return 'speaker';
		const normalized = role.trim().toLowerCase();
		if (!normalized) return 'speaker';
		if (normalized === 'assistant' || normalized === 'user' || normalized === 'system') {
			return normalized;
		}
		return 'speaker';
	}

	function extractMessageText(content: unknown): string {
		if (typeof content === 'string') return content;

		if (Array.isArray(content)) {
			return content
				.map((part) => extractMessageText(part))
				.filter(Boolean)
				.join('\n');
		}

		if (content && typeof content === 'object') {
			const obj = content as Record<string, unknown>;
			if (Array.isArray(obj.parts)) {
				return obj.parts
					.map((part) => extractMessageText(part))
					.filter(Boolean)
					.join('\n');
			}
			if (typeof obj.text === 'string') return obj.text;
			if (obj.content) return extractMessageText(obj.content);
		}

		return '';
	}

	function toTimestamp(value: unknown) {
		const candidate = Number(value);
		return Number.isFinite(candidate) ? candidate : null;
	}

	function extractTurnsFromMapping(entry: Record<string, unknown>) {
		const mapping =
			entry.mapping && typeof entry.mapping === 'object'
				? (entry.mapping as Record<string, unknown>)
				: null;
		if (!mapping) return [] as ConversationTurn[];

		const turns: ConversationTurn[] = [];
		let order = 0;

		for (const node of Object.values(mapping)) {
			const nodeRecord = node && typeof node === 'object' ? (node as Record<string, unknown>) : null;
			const message =
				nodeRecord?.message && typeof nodeRecord.message === 'object'
					? (nodeRecord.message as Record<string, unknown>)
					: null;
			if (!message) continue;

			const author =
				message.author && typeof message.author === 'object'
					? (message.author as Record<string, unknown>)
					: null;
			const role = normalizeRole(author?.role ?? message.role);
			const text = normalizeExtractedText(
				extractMessageText(message.content ?? message.text ?? message.parts)
			);
			if (!text) continue;

			const createdAt = toTimestamp(message.create_time ?? nodeRecord?.create_time);
			turns.push({
				role,
				text,
				createdAt,
				order
			});
			order += 1;
		}

		turns.sort((a, b) => {
			if (a.createdAt === null && b.createdAt === null) return a.order - b.order;
			if (a.createdAt === null) return 1;
			if (b.createdAt === null) return -1;
			if (a.createdAt === b.createdAt) return a.order - b.order;
			return a.createdAt - b.createdAt;
		});

		return turns;
	}

	function extractTurnsFromMessages(entry: Record<string, unknown>) {
		const messages = Array.isArray(entry.messages)
			? entry.messages
			: Array.isArray(entry.turns)
				? entry.turns
				: [];

		return messages
			.map((item, index) => {
				const obj = item && typeof item === 'object' ? (item as Record<string, unknown>) : {};
				const author =
					obj.author && typeof obj.author === 'object'
						? (obj.author as Record<string, unknown>)
						: null;
				const text = normalizeExtractedText(
					extractMessageText(obj.content ?? obj.text ?? obj.parts ?? obj.message)
				);
				if (!text) return null;

				return {
					role: normalizeRole(obj.role ?? author?.role),
					text,
					createdAt: toTimestamp(obj.create_time ?? obj.timestamp ?? obj.time),
					order: index
				};
			})
			.filter(Boolean) as ConversationTurn[];
	}

	function extractConversationText(entry: unknown) {
		if (typeof entry === 'string') return normalizeExtractedText(entry);
		if (!entry || typeof entry !== 'object') return '';

		const obj = entry as Record<string, unknown>;
		let turns = extractTurnsFromMapping(obj);

		if (!turns.length) {
			turns = extractTurnsFromMessages(obj);
		}

		if (!turns.length) {
			const promptText = extractMessageText(obj.prompt ?? obj.input ?? obj.question);
			const responseText = extractMessageText(obj.response ?? obj.output ?? obj.answer);
			const stitched = [promptText && `user: ${promptText}`, responseText && `assistant: ${responseText}`]
				.filter(Boolean)
				.join('\n\n');
			return normalizeExtractedText(stitched);
		}

		return normalizeExtractedText(turns.map((turn) => `${turn.role}: ${turn.text}`).join('\n\n'));
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
				const extractedText = extractConversationText(entry);
				const fallbackJson = JSON.stringify(entry);
				const textContent = normalizeExtractedText(
					extractedText ||
						(fallbackJson.length > 12000
							? `${fallbackJson.slice(0, 12000)}\n\n[JSON truncated for analysis]`
							: fallbackJson)
				);
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
				textContent: normalizeExtractedText(JSON.stringify(parsed))
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
			console.log(`Extracted ${chunks.length} conversation(s) from ${fileName} | Example text snippet: "${chunks[0]?.textContent.slice(0, 100)}"`);
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

		console.log(`[parse] ${baseName}: extracted ${chunks.length} conversation(s)`);

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
		activeInferenceIndex = null;
		completedAnalyses = 0;
		totalAnalyses = 0;

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
		activeInferenceIndex = null;
		completedAnalyses = 0;
		totalAnalyses = results.length;

		try {
			for (let i = 0; i < results.length; i++) {
				activeInferenceIndex = i;
				const result = results[i];
				const textForAnalysis = prepareTextForAnalysis(result.textContent);
				if (!textForAnalysis) {
					results[i].response = 'Error: unable to prepare content for analysis.';
					results[i].parsed = null;
					completedAnalyses = i + 1;
					continue;
				}
				try {
					results[i].response = '';
					results[i].parsed = null;

					const startTime = performance.now();
					const metrics: GenerationMetrics = {
						ttftMs: null,
						generationMs: 0,
						generatedChars: 0,
						emittedPieces: 0,
						piecesPerSecond: 0,
						charsPerSecond: 0
					};
					const finalText = await ministral.generate(textForAnalysis, prompt, undefined, (generationMetrics) => {
						Object.assign(metrics, generationMetrics);
					});
					const endTime = performance.now();

					results[i].response = finalText;
					results[i].parsed = parseAnalysis(finalText);
					console.log('[inference-metrics]', {
						sourceFile: results[i].sourceFile,
						conversationId: results[i].conversationId,
						conversationLabel: results[i].conversationLabel,
						runtimeStatusMessage,
						totalMs: Number((endTime - startTime).toFixed(2)),
						ttftMs: metrics.ttftMs === null ? null : Number(metrics.ttftMs.toFixed(2)),
						generationMs: Number(metrics.generationMs.toFixed(2)),
						emittedPieces: metrics.emittedPieces,
						piecesPerSecond: Number(metrics.piecesPerSecond.toFixed(2)),
						generatedChars: metrics.generatedChars || finalText.length,
						charsPerSecond: Number(metrics.charsPerSecond.toFixed(2))
					});
					console.log('[analysis-json]', {
						sourceFile: results[i].sourceFile,
						conversationId: results[i].conversationId,
						conversationLabel: results[i].conversationLabel,
						rawResponse: finalText,
						extractedJson: extractJSONPayload(finalText)
					});
				} catch (e) {
					results[i].response = 'Error: ' + e;
					results[i].parsed = null;
				} finally {
					completedAnalyses = i + 1;
				}
			}
		} catch (e) {
			console.error('Error during batch processing: ' + e);
		} finally {
			activeInferenceIndex = null;
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
	class="min-h-screen bg-[radial-gradient(circle_at_10%_10%,#dbeafe_0%,#eff6ff_38%,#f8fafc_80%)] text-slate-900 selection:bg-sky-200"
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
					Upload your export and quickly spot sensitive details you may have shared without thinking.
				</p>
				<p class="mt-3 text-xs font-medium text-slate-500">
					Works best on Google Chrome (desktop)
				</p>
				<p class="mt-2 text-xs font-medium text-slate-500">
					{runtimeStatusMessage}
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
						<div class="rounded-xl border border-sky-200 bg-sky-50 px-6 py-8 text-center">
							<div class="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm">
								<svg
									class="h-5 w-5 animate-spin text-sky-600"
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
								<p class="mt-4 text-sm font-semibold text-sky-900">
									Model is evaluating your conversations...
								</p>
								<p class="mt-1 text-xs text-sky-700">
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
										? 'border-sky-500 bg-sky-50/70'
										: 'border-gray-300 bg-gray-50 hover:border-sky-400 hover:bg-white'}"
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
									class="flex h-12 w-12 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-gray-900/5 transition-transform group-hover:scale-110 group-hover:text-sky-600"
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="1.5"
										stroke="currentColor"
										class="h-6 w-6 text-gray-500 group-hover:text-sky-600"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
										/>
									</svg>
								</div>
								<div class="mt-4 flex text-sm text-gray-600">
										<span class="font-semibold text-sky-700 hover:text-sky-600"
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
						{#if isProcessing && totalAnalyses > 0}
							<p class="mb-3 text-xs font-medium text-slate-500">
								Generating analysis... conversation {Math.min((activeInferenceIndex ?? completedAnalyses) + 1, totalAnalyses)} / {totalAnalyses}
							</p>
						{/if}

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
										{#each results as result, i (`${result.sourceFile}-${result.conversationId}-${i}`)}
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
												{:else if isProcessing && activeInferenceIndex === i}
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
												{:else if isProcessing}
													<span class="text-xs text-gray-300 italic">Queued</span>
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
			<p class="rounded-xl border border-sky-200 bg-sky-50 px-4 py-3 text-sky-900">
				Don&apos;t trust us blindly. This project is open source and fully replicable.
				<a
					href="https://github.com/oliviernguyenquoc/ministral-3b-web"
					target="_blank"
					rel="noopener noreferrer"
					class="ml-1 font-semibold underline decoration-sky-500 underline-offset-2 hover:text-sky-700"
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
						class="h-full rounded-full bg-sky-500 transition-all duration-300 ease-out"
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
