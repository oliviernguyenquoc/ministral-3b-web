<script lang="ts">
	import JSZip from 'jszip';
	import { slide, fade } from 'svelte/transition';
	import { cubicOut } from 'svelte/easing';
	import { ministral } from '$lib/ministral';

	let isModelReady = $state(false);
	let isProcessing = $state(false);
	let isDownloading = $state(false);
	let dragActive = $state(false);

	let loadProgress = $state(0);
	let loadStatus = $state('Preparation...');

	let prompt = $state(
		"Repere les informations privees ou sensibles dans cette conversation (sante, argent, identite, travail) et donne un verdict clair."
	);
	let results = $state<Array<{ fileName: string; imageSrc: string; response: string }>>([]);

	let fileInput = $state<HTMLInputElement>();

	async function loadModel() {
		if (ministral.isLoaded || isDownloading) return;

		isDownloading = true;

		try {
			await ministral.load((msg, percentage) => {
				loadStatus = msg;
				loadProgress = Math.round(percentage);
			});
			isModelReady = true;
		} catch (e) {
			console.error(e);
			alert("Le chargement a echoue. Consultez la console pour plus de details.");
			isDownloading = false;
		}
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

	async function processFile(file: File) {
		if (!file.name.endsWith('.zip')) {
			alert('Merci d importer un fichier ZIP valide.');
			return;
		}

		try {
			const zip = new JSZip();
			const zipData = await zip.loadAsync(file);
			// Reset results
			results = [];
			const imagePromises: Promise<void>[] = [];

			zipData.forEach((relativePath, zipEntry) => {
				// Filter for images and ignore MACOSX artifacts
				const isImage =
					/\.(jpg|jpeg|png|gif|webp)$/i.test(relativePath) && !/^__MACOSX|\/\._/.test(relativePath);

				if (isImage && !zipEntry.dir) {
					imagePromises.push(
						zipEntry.async('blob').then((blob) => {
							return new Promise<void>((resolve) => {
								const reader = new FileReader();
								reader.onload = (evt) => {
									const imageSrc = evt.target?.result as string;
									results.push({
										fileName: relativePath.split('/').pop() || relativePath,
										imageSrc,
										response: ''
									});
									resolve();
								};
								reader.readAsDataURL(blob);
							});
						})
					);
				}
			});

			await Promise.all(imagePromises);
		} catch (e) {
			console.error(e);
		}
	}

	async function runInference() {
		if (!results.length || !isModelReady) return;

		isProcessing = true;

		try {
			for (let i = 0; i < results.length; i++) {
				const result = results[i];
				const img = new Image();
				img.src = result.imageSrc;

				await new Promise<void>((resolve) => {
					img.onload = async () => {
						try {
							await ministral.generate(img, prompt, (updatedText) => {
								results[i].response = updatedText;
							});
						} catch (e) {
							results[i].response = 'Error: ' + e;
						}
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

		const headers = ['Fichier', 'Analyse'];
		const csvRows = [
			headers.join(','),
			...results.map((r) => `${escapeCSV(r.fileName)},${escapeCSV(r.response)}`)
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
		<header class="relative mb-10 overflow-hidden rounded-3xl border border-amber-200/80 bg-white/85 p-6 shadow-[0_20px_70px_-40px_rgba(180,83,9,0.5)] backdrop-blur sm:p-10">
			<div class="pointer-events-none absolute -top-20 -right-12 h-56 w-56 rounded-full bg-amber-200/60 blur-3xl"></div>
			<div class="pointer-events-none absolute -bottom-16 -left-10 h-48 w-48 rounded-full bg-orange-200/70 blur-3xl"></div>
			<div class="relative">
				<p class="inline-flex rounded-full border border-amber-300 bg-amber-100 px-3 py-1 text-xs font-semibold tracking-[0.16em] text-amber-900 uppercase">
					Test de confidentialite
				</p>
				<h1 class="mt-4 max-w-5xl text-3xl leading-tight font-black tracking-tight text-slate-900 sm:text-5xl">
					Decouvrez en 30 secondes ce que vos conversations ChatGPT revelent sur vous.
				</h1>
				<p class="mt-5 max-w-4xl text-base leading-relaxed text-slate-600 sm:text-lg">
					Vous avez peut-etre partage des informations personnelles sans vous en rendre compte:
					sante, argent, travail, vie privee.
				</p>
				<p class="mt-3 max-w-3xl text-sm leading-relaxed text-slate-500 sm:text-base">
					Cette page vous aide a verifier vos captures de conversation et a identifier rapidement ce
					qui pourrait etre trop sensible.
				</p>
				<div class="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center">
					<a
						href="#local-scan"
						class="rounded-xl border border-slate-900 bg-slate-900 px-5 py-3 text-sm font-semibold text-white transition-colors hover:bg-slate-800"
					>
						Commencer l'analyse
					</a>
				</div>
				<p class="mt-3 text-xs font-medium text-slate-500">
					Fonctionne mieux sur Google Chrome (ordinateur)
				</p>
			</div>
		</header>

		<!-- SECTION: Model Loader (Slides away when ready) -->
		{#if !isModelReady}
			<section
				transition:slide={{ duration: 500, easing: cubicOut, axis: 'y' }}
				class="mx-auto mb-8 max-w-2xl overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm"
			>
				<div class="border-b border-gray-100 bg-gray-50/50 px-6 py-4">
					<h2 class="flex items-center gap-2 text-lg font-semibold text-gray-800">
						<span
							class="flex h-6 w-6 items-center justify-center rounded-full bg-amber-600 text-xs text-white"
							>1</span
						>
						Preparation de l'analyse
					</h2>
				</div>

				<div class="p-8 text-center">
					<p class="mb-2 text-gray-600">Telechargez le module d'analyse.</p>
					<p class="mb-6 text-gray-600">
						Cette etape se fait une seule fois et reste en memoire dans votre navigateur.
					</p>

					<button
						onclick={loadModel}
						disabled={isDownloading}
						class="relative inline-flex w-full cursor-pointer items-center justify-center overflow-hidden rounded-xl bg-gray-900 px-6 py-4 text-sm font-semibold text-white shadow-sm transition-all hover:bg-black hover:shadow-md disabled:cursor-not-allowed disabled:shadow-none"
					>
						{#if isDownloading}
							<!-- Progress Bar Background -->
							<div
								class="absolute top-0 left-0 h-full bg-gray-700 transition-all duration-300 ease-out"
								style="width: {loadProgress}%"
							></div>

							<!-- Content (sitting on top of progress bar) -->
							<div class="relative z-10 flex items-center gap-3">
								{#if loadProgress < 99}
									<svg
										class="h-4 w-4 animate-spin text-gray-400"
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
											d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
										></path>
									</svg>
								{/if}
								<span class="font-mono">{loadProgress}%</span>
								<span class="border-l border-gray-600 pl-3 text-gray-300">{loadStatus}</span>
							</div>
						{:else}
							Telecharger et demarrer (~3GB)
						{/if}
					</button>
				</div>
			</section>
		{:else}
			<!-- Small status badge when model is loaded -->
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
				Analyse prete
			</div>
		{/if}

		<!-- SECTION: Interaction (Only visible when loaded or fading in) -->
		<section
			id="local-scan"
			class="mb-8 rounded-2xl border border-gray-200 bg-white shadow-sm transition-opacity duration-500 {isModelReady
				? 'opacity-100'
				: 'pointer-events-none opacity-40 blur-sm filter'}"
		>
			<div class="border-b border-gray-100 px-6 py-5">
				<h2 class="text-xl font-bold tracking-tight text-gray-900">Analyse de confidentialite</h2>
				<p class="text-sm text-gray-500">
					Importez vos captures de conversation (ZIP), puis obtenez un resultat clair.
				</p>
			</div>

			<div class="p-6">
				<!-- Styled Dropzone -->
				<div class="space-y-6">
					<div>
						<label for="dropzone" class="mb-2 block text-sm font-medium text-gray-700"
							>Importer des captures (ZIP)</label
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
								accept=".zip"
								bind:this={fileInput}
								onchange={handleFileSelect}
								class="hidden"
								disabled={!isModelReady}
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
									>Cliquez pour importer</span
								>
								<span class="pl-1">ou glissez-deposez votre archive ZIP</span>
							</div>
							<p class="mt-1 text-xs text-gray-500">
								Formats acceptes: JPG, PNG, GIF, WebP
							</p>
						</div>
					</div>

					<!-- Prompt Area -->
					<div>
						<label for="prompt" class="mb-2 block text-sm font-medium text-gray-700"
							>Consigne d'analyse</label
						>
						<div class="relative">
							<textarea
								id="prompt"
								bind:value={prompt}
								disabled={!isModelReady}
								rows="2"
								class="block w-full rounded-xl border border-gray-300 bg-white px-4 py-3 shadow-sm transition-colors focus:border-amber-500 focus:ring-2 focus:ring-amber-500/20 disabled:cursor-not-allowed disabled:bg-gray-100 disabled:text-gray-500 sm:text-sm"
								placeholder="Exemple: repere les infos personnelles, sante ou finance..."
							></textarea>
						</div>
					</div>

					<!-- Action Button -->
					<button
						onclick={runInference}
						disabled={!isModelReady || !results.length || isProcessing}
						class="w-full cursor-pointer rounded-xl bg-amber-600 px-4 py-3.5 text-sm font-semibold text-white shadow-sm hover:bg-amber-500 focus-visible:outline focus-visible:outline-offset-2 focus-visible:outline-amber-600 disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-400"
					>
						{#if isProcessing}
							<span class="flex items-center justify-center gap-2">
								<svg
									class="h-4 w-4 animate-spin text-white"
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
										d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
									></path>
								</svg>
								Analyse en cours ({results.length} image(s))...
							</span>
						{:else}
							Lancer l'analyse
						{/if}
					</button>
				</div>

				<!-- Results Table -->
				{#if results.length > 0}
					<div class="animate-in fade-in slide-in-from-bottom-4 mt-10 duration-500">
						<div class="mb-4 flex items-center justify-between">
							<h3 class="text-base leading-6 font-semibold text-gray-900">
								Resultats <span class="ml-2 rounded-md bg-gray-100 px-2 py-0.5 text-xs text-gray-600"
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
								Telecharger CSV
							</button>
						</div>

						<div class="overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm">
							<table class="min-w-full divide-y divide-gray-200">
								<thead class="bg-gray-50">
									<tr>
										<th
											scope="col"
											class="px-6 py-3 text-left text-xs font-medium tracking-wider text-gray-500 uppercase"
											>Apercu</th
										>
										<th
											scope="col"
											class="px-6 py-3 text-left text-xs font-medium tracking-wider text-gray-500 uppercase"
											>Fichier</th
										>
										<th
											scope="col"
											class="px-6 py-3 text-left text-xs font-medium tracking-wider text-gray-500 uppercase"
											>Analyse</th
										>
									</tr>
								</thead>
								<tbody class="divide-y divide-gray-200 bg-white">
									{#each results as result}
										<tr class="transition-colors hover:bg-gray-50/50">
											<td class="px-6 py-4 whitespace-nowrap">
												<div class="h-24 w-24 overflow-hidden rounded-lg border border-gray-100">
													<img
														src={result.imageSrc}
														alt={result.fileName}
														class="h-full w-full object-cover"
													/>
												</div>
											</td>
											<td class="px-6 py-4 whitespace-nowrap">
												<span class="font-mono text-xs text-gray-500">{result.fileName}</span>
											</td>
											<td class="px-6 py-4 text-sm text-gray-700">
												{#if result.response}
													<p class="leading-relaxed">{result.response}</p>
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
													<span class="text-xs text-gray-300 italic">En attente</span>
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

		<section class="mb-8 grid gap-4 lg:grid-cols-3">
			<article class="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
				<h2 class="text-lg font-bold text-slate-900">Pourquoi c'est rassurant</h2>
				<p class="mt-3 text-sm leading-relaxed text-slate-600">
					Vos fichiers restent sur votre ordinateur pendant l'analyse. Rien n'est envoye vers nos
					serveurs.
				</p>
			</article>
			<article class="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
				<h2 class="text-lg font-bold text-slate-900">Comment ca marche</h2>
				<ul class="mt-3 space-y-2 text-sm leading-relaxed text-slate-600">
					<li>1. Importez votre archive ZIP.</li>
					<li>2. Lancez l'analyse en un clic.</li>
					<li>3. Lisez les passages a risque detectes.</li>
				</ul>
			</article>
			<article class="rounded-2xl border border-rose-200 bg-rose-50 p-6 shadow-sm">
				<h2 class="text-lg font-bold text-rose-900">Exemple de resultat</h2>
				<p class="mt-3 text-sm leading-relaxed text-rose-800">
					&quot;Conversation tres privee: l'echange du 03/12 contient des details de sante
					sensibles.&quot;
				</p>
			</article>
		</section>

		<section class="mb-10 rounded-2xl border border-slate-900 bg-slate-900 px-6 py-5 text-center text-white">
			<p class="text-xs font-semibold tracking-[0.16em] text-amber-300 uppercase">Sensibilisation</p>
			<p class="mt-2 text-lg font-bold sm:text-xl">Dans le cloud, rien n'est vraiment secret.</p>
			<p class="mt-2 text-sm text-slate-300">Prenez 30 secondes pour verifier ce que vous avez partage.</p>
		</section>
	</div>
</div>
