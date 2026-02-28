import {
	AutoProcessor,
	AutoModelForImageTextToText,
	TextStreamer,
	type PreTrainedModel,
	type Processor,
	type ProgressInfo
} from '@huggingface/transformers';

const MODEL_ID = 'mistralai/Ministral-3-3B-Instruct-2512-ONNX';
const STREAM_UPDATE_INTERVAL_MS = 150;
const ANALYSIS_MAX_NEW_TOKENS = 220;

export type GenerationMetrics = {
	ttftMs: number | null;
	generationMs: number;
	generatedChars: number;
	emittedPieces: number;
	piecesPerSecond: number;
	charsPerSecond: number;
};

type RuntimeInfo = {
	webgpuSupported: boolean;
	adapterName: string | null;
};

export class MinistralEngine {
	private processor: Processor | null = null;
	private model: PreTrainedModel | null = null;
	private isWarmedUp = false;
	private runtimeInfo: RuntimeInfo = {
		webgpuSupported: false,
		adapterName: null
	};
	public isLoaded = false;
	public isLoading = false;

	private buildAnalysisMessages(conversationText: string, promptText: string) {
		return [
			{
				role: 'system',
				content:
					'You are a privacy risk analyzer. Return ONLY valid JSON (no markdown, no commentary, no explanations outside JSON)'
			},
			{
				role: 'user',
				content: `${promptText}\n\nConversation to analyze:\n${conversationText}`
			}
		];
	}

	async inspectRuntime(): Promise<RuntimeInfo> {
		const gpu = (
			globalThis.navigator as
				| undefined
				| {
						gpu?: {
							requestAdapter?: (options?: { powerPreference?: 'low-power' | 'high-performance' }) => Promise<unknown>;
						};
				  }
		)?.gpu;

		if (!gpu?.requestAdapter) {
			this.runtimeInfo = {
				webgpuSupported: false,
				adapterName: null
			};
			return this.runtimeInfo;
		}

		try {
			const adapter = (await gpu.requestAdapter({ powerPreference: 'high-performance' })) as
				| undefined
				| {
						info?: { description?: string };
						name?: string;
				  };

			this.runtimeInfo = {
				webgpuSupported: Boolean(adapter),
				adapterName: adapter?.info?.description ?? adapter?.name ?? null
			};
		} catch {
			this.runtimeInfo = {
				webgpuSupported: false,
				adapterName: null
			};
		}

		return this.runtimeInfo;
	}

	private async warmupIfNeeded(onProgress?: (message: string, percentage: number) => void) {
		if (this.isWarmedUp || !this.model || !this.processor?.tokenizer) return;

		onProgress?.('Compiling WebGPU kernels...', 98);
		const warmupMessages = this.buildAnalysisMessages(
			'user: I live at 10 Main Street and take insulin daily.\nassistant: Thanks, I noted it.',
			'Return compact JSON with verdict, overall_score, summary, and categories.'
		);
		const warmupInputs = this.processor.tokenizer.apply_chat_template(warmupMessages, {
			add_generation_prompt: true,
			tokenize: true,
			return_dict: true
		}) as Record<string, unknown>;

		await this.model.generate({
			...warmupInputs,
			max_new_tokens: 24,
			do_sample: false,
			repetition_penalty: 1.1
		} as any);
		onProgress?.('Finalizing runtime...', 99);
		this.isWarmedUp = true;
	}

	// Load the model and processor
	async load(onProgress?: (message: string, percentage: number) => void) {
		if (this.isLoaded) return;
		this.isLoading = true;

		// Track max progress to prevent UI jitter
		let maxProgress = 0;

		try {
			const runtime = await this.inspectRuntime();
			if (!runtime.webgpuSupported) {
				throw new Error(
					'WebGPU is unavailable in this browser/device. Use a recent Chrome/Edge desktop build with hardware acceleration enabled.'
				);
			}

			// 1. Load Processor
			onProgress?.('Initializing...', 5);
			this.processor = await AutoProcessor.from_pretrained(MODEL_ID);

			if (this.processor.image_processor) {
				this.processor.image_processor.size = { longest_edge: 480 };
			}

			// 2. Load Model
			onProgress?.('Preparing model...', 10);

			this.model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
				dtype: {
					embed_tokens: 'fp16',
					vision_encoder: 'q4',
					decoder_model_merged: 'q4f16'
				},
				device: 'webgpu',
				progress_callback: (info: ProgressInfo) => {
					// Only update UI for the actual download progress
					if (info.status === 'progress') {
						// The 'decoder' .onnx_data file is the huge ~3GB one.
						// We ignore the smaller files (like tokenizer.json) to prevent the bar jumping.
						if (info.file.includes('decoder') && info.file.endsWith('.onnx_data')) {
							const pct = info.loaded / info.total;

							// Map this file's progress (0-100%) to the UI's (10-100%)
							const currentProgress = 10 + pct * 90;

							// Prevent backward jumps
							if (currentProgress > maxProgress) {
								maxProgress = currentProgress;
								onProgress?.('Downloading weights (~3GB)...', maxProgress);
							}
						}
					}
				}
				});

				await this.warmupIfNeeded(onProgress);

				this.isLoaded = true;
				onProgress?.('Ready', 100);
			} catch (err) {
			console.error(err);
			throw err;
		} finally {
			this.isLoading = false;
		}
	}

	// Run inference directly on conversation text (no image preprocessing)
	async generate(
		conversationText: string,
		promptText: string,
		onTokenChunk?: (chunk: string) => void,
		onMetrics?: (metrics: GenerationMetrics) => void
	) {
		if (!this.model || !this.processor) throw new Error('Model not loaded');
		if (!this.processor.tokenizer) throw new Error('Tokenizer not loaded');

		// 1. Prepare chat-formatted text inputs
		const messages = this.buildAnalysisMessages(conversationText, promptText);

		const inputs = this.processor.tokenizer.apply_chat_template(messages, {
			add_generation_prompt: true,
			tokenize: true,
			return_dict: true
		});
		const generationInputs = inputs as Record<string, unknown>;

		// 2. Setup timing and optional streaming
		const generationStart = globalThis.performance?.now() ?? Date.now();
		let firstTokenAt: number | null = null;
		let generatedText = '';
		let emittedPieces = 0;

		// 3. Generate
		if (onTokenChunk) {
			let bufferedChunk = '';
			let lastUiUpdate = 0;
			const flushBufferedChunk = (now: number) => {
				if (!bufferedChunk) return;
				onTokenChunk(bufferedChunk);
				bufferedChunk = '';
				lastUiUpdate = now;
			};

			const streamer = new TextStreamer(this.processor.tokenizer!, {
				skip_prompt: true,
				skip_special_tokens: true,
				callback_function: (token: string) => {
					if (!token) return;
					const now = globalThis.performance?.now() ?? Date.now();
					if (firstTokenAt === null) {
						firstTokenAt = now;
					}

					generatedText += token;
					bufferedChunk += token;
					emittedPieces += 1;

					if (now - lastUiUpdate >= STREAM_UPDATE_INTERVAL_MS) {
						flushBufferedChunk(now);
					}
				}
			});

			const generateParams = {
				...generationInputs,
				max_new_tokens: ANALYSIS_MAX_NEW_TOKENS,
				do_sample: false,
				streamer,
				repetition_penalty: 1.2
			} as any;
			await this.model.generate(generateParams);
			const generationEnd = globalThis.performance?.now() ?? Date.now();
			flushBufferedChunk(generationEnd);
		} else {
			const outputTokenIds = await this.model.generate({
				...generationInputs,
				max_new_tokens: ANALYSIS_MAX_NEW_TOKENS,
				do_sample: false,
				repetition_penalty: 1.2
			} as any);

			const decoded = this.processor.tokenizer.batch_decode(outputTokenIds as any, {
				skip_special_tokens: true
			})[0] ?? '';
			const promptDecoded = this.processor.tokenizer.batch_decode((generationInputs as any).input_ids, {
				skip_special_tokens: true
			})[0] ?? '';

			generatedText = decoded.slice(promptDecoded.length).trim();
		}

		const generationEnd = globalThis.performance?.now() ?? Date.now();
		const generationMs =
			firstTokenAt === null || !onTokenChunk ? generationEnd - generationStart : generationEnd - firstTokenAt;
		const metrics: GenerationMetrics = {
			ttftMs: firstTokenAt === null ? null : firstTokenAt - generationStart,
			generationMs,
			generatedChars: generatedText.length,
			emittedPieces,
			piecesPerSecond: generationMs > 0 ? emittedPieces / (generationMs / 1000) : 0,
			charsPerSecond: generationMs > 0 ? generatedText.length / (generationMs / 1000) : 0
		};

		onMetrics?.(metrics);

		return generatedText;
	}
}

// Singleton instance to prevent reloading during navigation
export const ministral = new MinistralEngine();
