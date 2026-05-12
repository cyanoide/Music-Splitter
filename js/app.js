/**
 * Music Splitter — Main Application Controller
 * Handles UI state, file loading, Demucs ML processing, and stem player interaction.
 *
 * Uses demucs-web (timcsy/demucs-web, MIT) for real ML-based source separation
 * via ONNX Runtime Web. Falls back to frequency-band splitting if the model
 * fails to load (offline / no SharedArrayBuffer support).
 */

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.mjs';
import { DemucsProcessor, CONSTANTS as DEMUCS_CONST } from './demucs/index.js';

// ── Stem definitions (4 stems Demucs v4 + 3 sous-bandes optionnelles d'"autre") ──
const STEM_DEFS = {
    vocals:     { label: 'Voix',           color: '#f472b6', icon: '🎤' },
    drums:      { label: 'Batterie',       color: '#facc15', icon: '🥁' },
    bass:       { label: 'Basse',          color: '#fb923c', icon: '🎸' },
    other:      { label: 'Autre',          color: '#a78bfa', icon: '🎵' },
    other_low:  { label: 'Autre · grave',  color: '#7c3aed', icon: '🎻' },
    other_mid:  { label: 'Autre · médium', color: '#a78bfa', icon: '🎹' },
    other_high: { label: 'Autre · aigu',   color: '#c4b5fd', icon: '✨' },
};

// ── Sous-bandes pour le post-traitement d'"autre" ──
const OTHER_BANDS = [
    { name: 'other_low',  type: 'lowpass',  freq: 800,   Q: 0.7 },
    { name: 'other_mid',  type: 'bandpass', freq: 1700,  Q: 0.7 },  // ~800-3000 Hz
    { name: 'other_high', type: 'highpass', freq: 3000,  Q: 0.7 },
];

// ── State ──
const engine = new AudioEngine();
let masterWaveform = null;
const stemWaveforms = {};
let stemPeaks = {};
let masterPeaks = null;
let currentFileName = '';

// ── Demucs ML state ──
let demucsProcessor = null;
let demucsLoaded = false;
let demucsLoading = false;
let demucsLoadError = null;
let demucsBackend = 'wasm';

// ── DOM refs ──
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const views = {
    upload: $('#view-upload'),
    processing: $('#view-processing'),
    player: $('#view-player'),
};

const dom = {
    dropZone: $('#drop-zone'),
    fileInput: $('#file-input'),
    stemsInput: $('#stems-input'),
    btnLoadStems: $('#btn-load-stems'),
    btnNewSession: $('#btn-new-session'),
    btnCancel: $('#btn-cancel'),
    btnPlay: $('#btn-play'),
    btnStop: $('#btn-stop'),
    btnSkipBack: $('#btn-skip-back'),
    btnSkipForward: $('#btn-skip-forward'),
    btnLoop: $('#btn-loop'),
    btnExport: $('#btn-export'),
    btnDoExport: $('#btn-do-export'),
    btnCloseExport: $('#btn-close-export'),
    exportModal: $('#export-modal'),
    iconPlay: $('#icon-play'),
    iconPause: $('#icon-pause'),
    progressRing: $('#progress-ring'),
    progressPercent: $('#progress-percent'),
    processingTitle: $('#processing-title'),
    processingStatus: $('#processing-status'),
    trackName: $('#track-name'),
    trackDuration: $('#track-duration'),
    timeCurrent: $('#time-current'),
    timeTotal: $('#time-total'),
    masterWaveformCanvas: $('#master-waveform'),
    masterContainer: $('.master-waveform-container'),
    playhead: $('#playhead'),
    stemTracks: $('#stem-tracks'),
    optSplitOther: $('#opt-split-other'),
};

// ── View management ──
function showView(name) {
    for (const [key, el] of Object.entries(views)) {
        el.classList.toggle('active', key === name);
    }
    dom.btnNewSession.style.display = name === 'player' ? '' : 'none';
}

// ── Time formatting ──
function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// ── Progress ring ──
function setProgress(pct) {
    const circumference = 2 * Math.PI * 54; // r=54
    const offset = circumference * (1 - pct / 100);
    dom.progressRing.style.strokeDashoffset = offset;
    dom.progressPercent.textContent = `${Math.round(pct)}%`;
}

function setStep(stepId) {
    $$('.processing-steps .step').forEach(el => {
        el.classList.remove('active', 'done');
    });
    const steps = ['step-model', 'step-decode', 'step-separate', 'step-finalize'];
    const idx = steps.indexOf(stepId);
    for (let i = 0; i < idx; i++) {
        $(`#${steps[i]}`).classList.add('done');
    }
    if (idx >= 0) $(`#${stepId}`).classList.add('active');
}

// ── Stem UI generation ──
function buildStemTracks(stemNames) {
    dom.stemTracks.innerHTML = '';
    for (const name of stemNames) {
        const def = STEM_DEFS[name] || { label: name, color: '#a78bfa', icon: '🎵' };
        const row = document.createElement('div');
        row.className = 'stem-row';
        row.dataset.stem = name;
        row.innerHTML = `
            <div class="stem-info">
                <div class="stem-color" style="background:${def.color}"></div>
                <div>
                    <div class="stem-name">${def.label}</div>
                </div>
                <div class="stem-controls">
                    <button class="btn-stem btn-mute" data-stem="${name}" title="Mute">M</button>
                    <button class="btn-stem btn-solo" data-stem="${name}" title="Solo">S</button>
                </div>
            </div>
            <div class="stem-waveform-container" data-stem="${name}">
                <canvas class="stem-waveform" data-stem="${name}"></canvas>
                <div class="stem-playhead" data-stem="${name}"></div>
            </div>
            <div class="stem-volume-container">
                <input type="range" class="stem-volume" data-stem="${name}" min="0" max="100" value="100">
                <span class="stem-volume-val" data-stem="${name}">100</span>
            </div>
        `;
        dom.stemTracks.appendChild(row);
    }
}

function drawStemWaveforms(stemBuffers) {
    for (const [name, buffer] of Object.entries(stemBuffers)) {
        const canvas = $(`.stem-waveform[data-stem="${name}"]`);
        if (!canvas) continue;
        const def = STEM_DEFS[name] || { color: '#a78bfa' };
        const wf = new Waveform(canvas, { color: def.color, barWidth: 1.5, barGap: 0.5 });
        const peaks = Waveform.computePeaks(buffer, 300);
        wf.draw(peaks);
        stemWaveforms[name] = wf;
        stemPeaks[name] = peaks;
    }
}

function drawMasterWaveform(stemBuffers) {
    const names = Object.keys(stemBuffers);
    if (names.length === 0) return;
    const first = stemBuffers[names[0]];
    const sampleRate = first.sampleRate;
    const length = Math.max(...Object.values(stemBuffers).map(b => b.length));
    const numChannels = first.numberOfChannels;

    const offline = new OfflineAudioContext(numChannels, length, sampleRate);
    for (const buffer of Object.values(stemBuffers)) {
        const source = offline.createBufferSource();
        source.buffer = buffer;
        source.connect(offline.destination);
        source.start(0);
    }
    offline.startRendering().then(mixedBuffer => {
        masterPeaks = Waveform.computePeaks(mixedBuffer, 400);
        if (masterWaveform) masterWaveform.destroy();
        masterWaveform = new Waveform(dom.masterWaveformCanvas, {
            color: '#8b5cf6',
            barWidth: 2,
            barGap: 1,
        });
        masterWaveform.draw(masterPeaks);
    });
}

// ── Update play/pause icon ──
function updatePlayIcon() {
    dom.iconPlay.style.display = engine.playing ? 'none' : '';
    dom.iconPause.style.display = engine.playing ? '' : 'none';
}

// ── ONNX Runtime backend detection ──
async function detectBackend() {
    if ('gpu' in navigator) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) return 'webgpu';
        } catch (e) {
            console.warn('WebGPU non disponible:', e);
        }
    }
    return 'wasm';
}

// ── Demucs model loading ──
async function ensureDemucsModelLoaded() {
    if (demucsLoaded) return true;
    if (demucsLoadError) throw demucsLoadError;
    if (demucsLoading) {
        // Wait for in-flight load
        while (demucsLoading) await new Promise(r => setTimeout(r, 100));
        if (demucsLoadError) throw demucsLoadError;
        return demucsLoaded;
    }

    demucsLoading = true;
    try {
        setStep('step-model');
        dom.processingTitle.textContent = 'Chargement du modèle IA...';
        dom.processingStatus.textContent = 'Initialisation de ONNX Runtime';
        setProgress(0);

        demucsBackend = await detectBackend();
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        if (demucsBackend === 'webgpu') {
            ort.env.webgpu = ort.env.webgpu || {};
            ort.env.webgpu.powerPreference = 'high-performance';
        }

        demucsProcessor = new DemucsProcessor({
            ort,
            sessionOptions: {
                executionProviders: demucsBackend === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
                graphOptimizationLevel: 'basic',
            },
            onLog: (phase, msg) => console.log(`[Demucs/${phase}] ${msg}`),
            onDownloadProgress: (loaded, total) => {
                const pct = total > 0 ? (loaded / total) * 100 : 0;
                const loadedMB = (loaded / 1024 / 1024).toFixed(1);
                const totalMB = (total / 1024 / 1024).toFixed(1);
                dom.processingStatus.textContent =
                    `Téléchargement du modèle Demucs : ${loadedMB} Mo / ${totalMB} Mo (${pct.toFixed(0)}%) — backend : ${demucsBackend.toUpperCase()}`;
                setProgress(pct * 0.2); // model load is 0-20%
            },
        });

        await demucsProcessor.loadModel(DEMUCS_CONST.DEFAULT_MODEL_URL);
        demucsLoaded = true;
        demucsLoading = false;
        return true;
    } catch (err) {
        demucsLoading = false;
        demucsLoadError = err;
        console.error('Échec chargement Demucs:', err);
        throw err;
    }
}

// ── Resampling helper (linear interp) ──
function resample(channel, srcRate, dstRate) {
    if (srcRate === dstRate) return channel;
    const ratio = dstRate / srcRate;
    const newLength = Math.floor(channel.length * ratio);
    const out = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
        const srcIdx = i / ratio;
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(idx0 + 1, channel.length - 1);
        const frac = srcIdx - idx0;
        out[i] = channel[idx0] * (1 - frac) + channel[idx1] * frac;
    }
    return out;
}

// ── Float32 channel pair → AudioBuffer ──
function channelsToAudioBuffer(left, right, sampleRate) {
    const buffer = engine.ctx.createBuffer(2, left.length, sampleRate);
    buffer.getChannelData(0).set(left);
    buffer.getChannelData(1).set(right);
    return buffer;
}

// ── Demucs separation (real ML) ──
async function separateWithDemucs(audioBuffer) {
    try {
        await ensureDemucsModelLoaded();
    } catch (err) {
        console.warn('Modèle Demucs indisponible — fallback sur séparation fréquentielle. Cause :', err.message);
        dom.processingTitle.textContent = 'Séparation (mode dégradé)';
        dom.processingStatus.textContent =
            'Modèle IA indisponible — repli sur filtrage fréquentiel. Vérifie ta connexion ou ton navigateur.';
        setStep('step-separate');
        await new Promise(r => setTimeout(r, 1500));
        return await frequencyBandSeparation(audioBuffer);
    }

    setStep('step-separate');
    dom.processingTitle.textContent = 'Séparation par IA Demucs...';
    const targetRate = DEMUCS_CONST.SAMPLE_RATE; // 44100
    let left = audioBuffer.getChannelData(0);
    let right = audioBuffer.numberOfChannels > 1
        ? audioBuffer.getChannelData(1)
        : left;

    if (audioBuffer.sampleRate !== targetRate) {
        dom.processingStatus.textContent =
            `Ré-échantillonnage ${audioBuffer.sampleRate} Hz → ${targetRate} Hz`;
        left = resample(left, audioBuffer.sampleRate, targetRate);
        right = resample(right, audioBuffer.sampleRate, targetRate);
    }

    // Copy if same reference (mono)
    if (left === right) {
        right = new Float32Array(left);
    }

    const startTime = Date.now();
    demucsProcessor.onProgress = ({ progress, currentSegment, totalSegments }) => {
        // Map 0..1 → 25..90
        setProgress(25 + progress * 65);
        const elapsed = (Date.now() - startTime) / 1000;
        const speed = elapsed > 0 ? ((currentSegment / totalSegments) * (left.length / targetRate)) / elapsed : 0;
        dom.processingStatus.textContent =
            `Segment ${currentSegment}/${totalSegments} · ${demucsBackend.toUpperCase()} · ${speed.toFixed(2)}x temps réel`;
    };

    const result = await demucsProcessor.separate(left, right);

    // result = { drums, bass, other, vocals } each {left, right} Float32Array
    const stems = {};
    for (const name of ['drums', 'bass', 'other', 'vocals']) {
        stems[name] = channelsToAudioBuffer(result[name].left, result[name].right, targetRate);
    }
    return stems;
}

/**
 * Frequency-band separation: graceful fallback only.
 * Used when ONNX Runtime / WebGPU / SharedArrayBuffer are unavailable.
 * Not a real separation — just bandpass filters applied to the full mix.
 */
async function frequencyBandSeparation(audioBuffer) {
    const sampleRate = audioBuffer.sampleRate;
    const length = audioBuffer.length;
    const numChannels = audioBuffer.numberOfChannels;

    const bands = {
        bass:   { low: 20,   high: 250 },
        drums:  { low: 60,   high: 8000 },
        vocals: { low: 300,  high: 5000 },
        other:  { low: 5000, high: 20000 },
    };

    const stems = {};
    const total = Object.keys(bands).length;
    let processed = 0;

    for (const [name, band] of Object.entries(bands)) {
        const offline = new OfflineAudioContext(numChannels, length, sampleRate);
        const source = offline.createBufferSource();
        source.buffer = audioBuffer;

        if (name === 'drums') {
            const bp = offline.createBiquadFilter();
            bp.type = 'bandpass';
            bp.frequency.value = Math.sqrt(band.low * band.high);
            bp.Q.value = 0.5;
            const comp = offline.createDynamicsCompressor();
            comp.threshold.value = -20;
            comp.ratio.value = 4;
            comp.attack.value = 0.001;
            comp.release.value = 0.05;
            source.connect(bp); bp.connect(comp); comp.connect(offline.destination);
        } else if (name === 'vocals') {
            const bp = offline.createBiquadFilter();
            bp.type = 'bandpass';
            bp.frequency.value = 1200;
            bp.Q.value = 1.2;
            const peak = offline.createBiquadFilter();
            peak.type = 'peaking';
            peak.frequency.value = 3000;
            peak.gain.value = 3;
            peak.Q.value = 1;
            source.connect(bp); bp.connect(peak); peak.connect(offline.destination);
        } else if (name === 'bass') {
            const lp = offline.createBiquadFilter();
            lp.type = 'lowpass';
            lp.frequency.value = band.high;
            lp.Q.value = 0.7;
            source.connect(lp); lp.connect(offline.destination);
        } else {
            const hp = offline.createBiquadFilter();
            hp.type = 'highpass';
            hp.frequency.value = band.low;
            hp.Q.value = 0.7;
            source.connect(hp); hp.connect(offline.destination);
        }

        source.start(0);
        stems[name] = await offline.startRendering();

        processed++;
        setProgress(30 + (processed / total) * 60);
        dom.processingStatus.textContent = `Stem ${processed}/${total} : ${STEM_DEFS[name]?.label || name}`;
        await new Promise(r => setTimeout(r, 50));
    }

    return stems;
}

/**
 * Sous-divise un AudioBuffer "autre" en 3 sous-bandes fréquentielles
 * (grave, médium, aigu) via OfflineAudioContext + filtres biquad.
 * Retourne un dict { other_low, other_mid, other_high } d'AudioBuffers.
 */
async function splitOtherIntoBands(otherBuffer) {
    const sampleRate = otherBuffer.sampleRate;
    const length = otherBuffer.length;
    const numChannels = otherBuffer.numberOfChannels;
    const result = {};

    for (const band of OTHER_BANDS) {
        const offline = new OfflineAudioContext(numChannels, length, sampleRate);
        const source = offline.createBufferSource();
        source.buffer = otherBuffer;

        if (band.type === 'bandpass') {
            // Bande médium plus large : enchaîne highpass 800Hz + lowpass 3kHz
            const hp = offline.createBiquadFilter();
            hp.type = 'highpass';
            hp.frequency.value = 800;
            hp.Q.value = 0.7;
            const lp = offline.createBiquadFilter();
            lp.type = 'lowpass';
            lp.frequency.value = 3000;
            lp.Q.value = 0.7;
            source.connect(hp);
            hp.connect(lp);
            lp.connect(offline.destination);
        } else {
            const filter = offline.createBiquadFilter();
            filter.type = band.type;
            filter.frequency.value = band.freq;
            filter.Q.value = band.Q;
            source.connect(filter);
            filter.connect(offline.destination);
        }

        source.start(0);
        result[band.name] = await offline.startRendering();
    }

    return result;
}

// ── File handling ──
async function handleAudioFile(file) {
    if (!file || (!file.type.startsWith('audio/') && !file.name.match(/\.(mp3|flac|wav|ogg|m4a)$/i))) {
        alert('Format non supporté. Utilise MP3, FLAC, WAV, OGG ou M4A.');
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        alert('Fichier trop volumineux (max 50 Mo).');
        return;
    }

    currentFileName = file.name.replace(/\.[^.]+$/, '');
    showView('processing');
    setProgress(0);
    dom.processingStatus.textContent = file.name;

    try {
        engine.init();

        // Step 1: load Demucs model (no-op if already loaded). Failure is caught
        // and surfaced inside separateWithDemucs() which falls back gracefully.
        try {
            await ensureDemucsModelLoaded();
        } catch (e) {
            // Continue — separateWithDemucs() will handle fallback.
        }

        // Step 2: decode the audio
        setStep('step-decode');
        dom.processingTitle.textContent = 'Décodage audio...';
        const arrayBuffer = await file.arrayBuffer();
        setProgress(22);
        const audioBuffer = await engine.decodeAudio(arrayBuffer);

        // Step 3: separate
        const stemBuffers = await separateWithDemucs(audioBuffer);

        // Step 3.5 (optional): sub-divide "other" into frequency bands
        if (dom.optSplitOther?.checked && stemBuffers.other) {
            setProgress(92);
            dom.processingTitle.textContent = 'Sous-division d\'"Autre"...';
            dom.processingStatus.textContent = 'Découpage en grave / médium / aigu';
            const subBands = await splitOtherIntoBands(stemBuffers.other);
            delete stemBuffers.other;
            Object.assign(stemBuffers, subBands);
        }

        // Finalize
        setStep('step-finalize');
        setProgress(95);
        dom.processingTitle.textContent = 'Finalisation...';
        dom.processingStatus.textContent = 'Préparation du lecteur';

        await initPlayer(stemBuffers);
        setProgress(100);
        await new Promise(r => setTimeout(r, 300));
        showView('player');
    } catch (err) {
        console.error('Processing error:', err);
        alert('Erreur lors du traitement : ' + err.message);
        showView('upload');
    }
}

async function handlePreSeparatedStems(files) {
    if (files.length === 0) return;

    showView('processing');
    setProgress(0);
    setStep('step-decode');
    dom.processingTitle.textContent = 'Chargement des stems...';

    engine.init();
    const stemBuffers = {};
    let i = 0;

    for (const file of files) {
        const name = detectStemName(file.name);
        dom.processingStatus.textContent = file.name;
        const arrayBuffer = await file.arrayBuffer();
        stemBuffers[name] = await engine.decodeAudio(arrayBuffer);
        i++;
        setProgress((i / files.length) * 90);
    }

    currentFileName = 'Multi-stems';
    setStep('step-finalize');
    setProgress(95);

    await initPlayer(stemBuffers);
    setProgress(100);
    await new Promise(r => setTimeout(r, 300));
    showView('player');
}

function detectStemName(filename) {
    const lower = filename.toLowerCase();
    if (lower.includes('vocal') || lower.includes('voix') || lower.includes('voice')) return 'vocals';
    if (lower.includes('bass') || lower.includes('basse')) return 'bass';
    if (lower.includes('drum') || lower.includes('batterie') || lower.includes('perc')) return 'drums';
    if (lower.includes('other') || lower.includes('autre')) return 'other';
    return filename.replace(/\.[^.]+$/, '').toLowerCase().replace(/[^a-z0-9]/g, '_');
}

// ── Player initialization ──
async function initPlayer(stemBuffers) {
    engine.loadStems(stemBuffers);

    dom.trackName.textContent = currentFileName;
    dom.trackDuration.textContent = formatTime(engine.duration);
    dom.timeTotal.textContent = formatTime(engine.duration);
    dom.timeCurrent.textContent = '0:00';

    buildStemTracks(engine.stemOrder);
    drawStemWaveforms(stemBuffers);
    drawMasterWaveform(stemBuffers);

    engine._onTimeUpdate = (current, duration) => {
        dom.timeCurrent.textContent = formatTime(current);
        const progress = duration > 0 ? current / duration : 0;
        dom.playhead.style.left = `${progress * 100}%`;
        if (masterWaveform && masterPeaks) {
            masterWaveform.draw(masterPeaks, progress);
        }
        // Per-stem playhead + bar progress
        for (const name of engine.stemOrder) {
            const ph = $(`.stem-playhead[data-stem="${name}"]`);
            if (ph) ph.style.left = `${progress * 100}%`;
            const wf = stemWaveforms[name];
            const peaks = stemPeaks[name];
            if (wf && peaks) wf.draw(peaks, progress);
        }
    };

    engine._onEnded = () => {
        updatePlayIcon();
        dom.timeCurrent.textContent = '0:00';
        dom.playhead.style.left = '0%';
        if (masterWaveform && masterPeaks) {
            masterWaveform.draw(masterPeaks);
        }
        for (const name of engine.stemOrder) {
            const ph = $(`.stem-playhead[data-stem="${name}"]`);
            if (ph) ph.style.left = '0%';
            const wf = stemWaveforms[name];
            const peaks = stemPeaks[name];
            if (wf && peaks) wf.draw(peaks);
        }
    };

    buildExportStemCheckboxes(engine.stemOrder);
}

function buildExportStemCheckboxes(stemNames) {
    const container = $('#export-stem-select');
    container.innerHTML = '';
    for (const name of stemNames) {
        const def = STEM_DEFS[name] || { label: name };
        const label = document.createElement('label');
        label.innerHTML = `<input type="checkbox" value="${name}" checked> ${def.label}`;
        container.appendChild(label);
    }
}

// ── Event listeners ──
function initEvents() {
    dom.dropZone.addEventListener('click', () => dom.fileInput.click());
    dom.dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dom.dropZone.classList.add('dragover');
    });
    dom.dropZone.addEventListener('dragleave', () => {
        dom.dropZone.classList.remove('dragover');
    });
    dom.dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dom.dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleAudioFile(file);
    });
    dom.fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleAudioFile(file);
    });

    dom.btnLoadStems.addEventListener('click', () => dom.stemsInput.click());
    dom.stemsInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handlePreSeparatedStems(Array.from(e.target.files));
        }
    });

    dom.btnCancel.addEventListener('click', () => showView('upload'));
    dom.btnNewSession.addEventListener('click', () => {
        engine.stop();
        showView('upload');
    });

    dom.btnPlay.addEventListener('click', () => {
        if (engine.playing) engine.pause();
        else engine.play();
        updatePlayIcon();
    });

    dom.btnStop.addEventListener('click', () => {
        engine.stop();
        updatePlayIcon();
        dom.timeCurrent.textContent = '0:00';
        dom.playhead.style.left = '0%';
        if (masterWaveform && masterPeaks) {
            masterWaveform.draw(masterPeaks);
        }
        for (const name of engine.stemOrder) {
            const ph = $(`.stem-playhead[data-stem="${name}"]`);
            if (ph) ph.style.left = '0%';
            const wf = stemWaveforms[name];
            const peaks = stemPeaks[name];
            if (wf && peaks) wf.draw(peaks);
        }
    });

    dom.btnSkipBack.addEventListener('click', () => engine.skip(-5));
    dom.btnSkipForward.addEventListener('click', () => engine.skip(5));

    dom.btnLoop.addEventListener('click', () => {
        const looping = engine.toggleLoop();
        dom.btnLoop.classList.toggle('active', looping);
    });

    dom.masterContainer.addEventListener('click', (e) => {
        const rect = dom.masterContainer.getBoundingClientRect();
        const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        engine.seek(ratio * engine.duration);
        if (!engine.playing) {
            dom.timeCurrent.textContent = formatTime(engine.currentTime);
            dom.playhead.style.left = `${ratio * 100}%`;
            for (const name of engine.stemOrder) {
                const ph = $(`.stem-playhead[data-stem="${name}"]`);
                if (ph) ph.style.left = `${ratio * 100}%`;
                const wf = stemWaveforms[name];
                const peaks = stemPeaks[name];
                if (wf && peaks) wf.draw(peaks, ratio);
            }
            if (masterWaveform && masterPeaks) {
                masterWaveform.draw(masterPeaks, ratio);
            }
        }
    });

    dom.stemTracks.addEventListener('click', (e) => {
        // Priority 1 : mute/solo buttons
        const btn = e.target.closest('.btn-mute, .btn-solo');
        if (btn) {
            const stemName = btn.dataset.stem;
            if (btn.classList.contains('btn-mute')) {
                const muted = engine.toggleMute(stemName);
                btn.classList.toggle('active-mute', muted);
                btn.closest('.stem-row').classList.toggle('muted', muted);
            } else if (btn.classList.contains('btn-solo')) {
                const soloed = engine.toggleSolo(stemName);
                btn.classList.toggle('active-solo', soloed);
                btn.closest('.stem-row').classList.toggle('solo', soloed);
                updateSoloVisuals();
            }
            return;
        }

        // Priority 2 : click sur waveform de stem → seek global
        const wfContainer = e.target.closest('.stem-waveform-container');
        if (wfContainer) {
            const rect = wfContainer.getBoundingClientRect();
            const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            engine.seek(ratio * engine.duration);
            if (!engine.playing) {
                dom.timeCurrent.textContent = formatTime(engine.currentTime);
                dom.playhead.style.left = `${ratio * 100}%`;
                // Repercute sur tous les stems
                for (const name of engine.stemOrder) {
                    const ph = $(`.stem-playhead[data-stem="${name}"]`);
                    if (ph) ph.style.left = `${ratio * 100}%`;
                    const wf = stemWaveforms[name];
                    const peaks = stemPeaks[name];
                    if (wf && peaks) wf.draw(peaks, ratio);
                }
                if (masterWaveform && masterPeaks) {
                    masterWaveform.draw(masterPeaks, ratio);
                }
            }
        }
    });

    dom.stemTracks.addEventListener('input', (e) => {
        if (e.target.classList.contains('stem-volume')) {
            const stemName = e.target.dataset.stem;
            const vol = parseInt(e.target.value) / 100;
            engine.setVolume(stemName, vol);
            const valSpan = $(`.stem-volume-val[data-stem="${stemName}"]`);
            if (valSpan) valSpan.textContent = e.target.value;
        }
    });

    dom.btnExport.addEventListener('click', () => {
        dom.exportModal.style.display = 'flex';
    });
    dom.btnCloseExport.addEventListener('click', () => {
        dom.exportModal.style.display = 'none';
    });
    dom.exportModal.addEventListener('click', (e) => {
        if (e.target === dom.exportModal) dom.exportModal.style.display = 'none';
    });
    $$('input[name="export-type"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            $('#export-stem-select').style.display = e.target.value === 'selected' ? '' : 'none';
        });
    });
    dom.btnDoExport.addEventListener('click', doExport);

    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (!views.player.classList.contains('active')) return;
        switch (e.code) {
            case 'Space':
                e.preventDefault();
                dom.btnPlay.click();
                break;
            case 'ArrowLeft':
                engine.skip(-5);
                break;
            case 'ArrowRight':
                engine.skip(5);
                break;
            case 'KeyL':
                dom.btnLoop.click();
                break;
        }
    });
}

function updateSoloVisuals() {
    const anySoloed = Object.values(engine.stems).some(s => s.soloed);
    for (const name of engine.stemOrder) {
        const row = $(`.stem-row[data-stem="${name}"]`);
        if (!row) continue;
        const stem = engine.stems[name];
        if (anySoloed && !stem.soloed && !stem.muted) {
            row.style.opacity = '0.35';
        } else if (stem.muted) {
            row.style.opacity = '0.35';
        } else {
            row.style.opacity = '1';
        }
    }
}

// ── Export ──
async function doExport() {
    const exportType = $('input[name="export-type"]:checked').value;

    if (exportType === 'all') {
        for (const name of engine.stemOrder) {
            const blob = engine.exportStem(name);
            if (blob) downloadBlob(blob, `${currentFileName}_${name}.wav`);
            await new Promise(r => setTimeout(r, 200));
        }
    } else if (exportType === 'mix') {
        const blob = engine.exportMix();
        downloadBlob(blob, `${currentFileName}_mix.wav`);
    } else if (exportType === 'selected') {
        const selected = Array.from($$('#export-stem-select input:checked')).map(cb => cb.value);
        if (selected.length === 0) {
            alert('Sélectionne au moins un stem.');
            return;
        }
        for (const name of selected) {
            const blob = engine.exportStem(name);
            if (blob) downloadBlob(blob, `${currentFileName}_${name}.wav`);
            await new Promise(r => setTimeout(r, 200));
        }
    }

    dom.exportModal.style.display = 'none';
}

function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 5000);
}

// ── Init ──
function init() {
    initEvents();
    showView('upload');
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
