/**
 * AudioEngine — Web Audio API based multi-stem player
 * Handles synchronized playback, per-stem volume/mute/solo, and export.
 */
class AudioEngine {
    constructor() {
        this.ctx = null;
        this.stems = {};       // { name: { buffer, source, gain, muted, soloed, volume } }
        this.stemOrder = [];
        this.playing = false;
        this.looping = false;
        this.startTime = 0;    // AudioContext time when playback started
        this.pauseOffset = 0;  // seconds into the track when paused
        this.duration = 0;
        this._animFrame = null;
        this._onTimeUpdate = null;
        this._onEnded = null;
    }

    /** Initialize AudioContext (must be called from user gesture) */
    init() {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
        }
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    /**
     * Load stems from AudioBuffers
     * @param {Object<string, AudioBuffer>} stemBuffers  { "vocals": AudioBuffer, ... }
     */
    loadStems(stemBuffers) {
        this.stop();
        this.stems = {};
        this.stemOrder = [];
        this.duration = 0;
        this.pauseOffset = 0;

        for (const [name, buffer] of Object.entries(stemBuffers)) {
            const gain = this.ctx.createGain();
            gain.connect(this.ctx.destination);
            gain.gain.value = 1.0;

            this.stems[name] = {
                buffer,
                source: null,
                gain,
                muted: false,
                soloed: false,
                volume: 1.0,
            };
            this.stemOrder.push(name);
            if (buffer.duration > this.duration) {
                this.duration = buffer.duration;
            }
        }
        this._updateGains();
    }

    /** Start or resume playback */
    play() {
        if (this.playing) return;
        this.init();

        for (const name of this.stemOrder) {
            const stem = this.stems[name];
            const source = this.ctx.createBufferSource();
            source.buffer = stem.buffer;
            source.loop = this.looping;
            source.connect(stem.gain);
            source.start(0, this.pauseOffset);
            stem.source = source;
        }

        this.startTime = this.ctx.currentTime - this.pauseOffset;
        this.playing = true;
        this._startTimeUpdates();

        // Handle end of track
        const firstStem = this.stems[this.stemOrder[0]];
        if (firstStem?.source) {
            firstStem.source.onended = () => {
                if (!this.looping && this.playing) {
                    this.playing = false;
                    this.pauseOffset = 0;
                    this._stopTimeUpdates();
                    this._onEnded?.();
                }
            };
        }
    }

    /** Pause playback */
    pause() {
        if (!this.playing) return;
        this.pauseOffset = this.currentTime;
        this._stopSources();
        this.playing = false;
        this._stopTimeUpdates();
    }

    /** Stop and reset to beginning */
    stop() {
        this._stopSources();
        this.playing = false;
        this.pauseOffset = 0;
        this._stopTimeUpdates();
    }

    /** Seek to a position in seconds */
    seek(time) {
        const wasPlaying = this.playing;
        if (wasPlaying) {
            this._stopSources();
        }
        this.pauseOffset = Math.max(0, Math.min(time, this.duration));
        if (wasPlaying) {
            this.playing = false;
            this.play();
        }
    }

    /** Skip forward/backward by delta seconds */
    skip(delta) {
        this.seek(this.currentTime + delta);
    }

    /** Get current playback time in seconds */
    get currentTime() {
        if (this.playing) {
            return Math.min(this.ctx.currentTime - this.startTime, this.duration);
        }
        return this.pauseOffset;
    }

    /** Toggle loop mode */
    toggleLoop() {
        this.looping = !this.looping;
        for (const stem of Object.values(this.stems)) {
            if (stem.source) {
                stem.source.loop = this.looping;
            }
        }
        return this.looping;
    }

    /** Set volume for a stem (0-1) */
    setVolume(name, vol) {
        if (!this.stems[name]) return;
        this.stems[name].volume = vol;
        this._updateGains();
    }

    /** Toggle mute on a stem */
    toggleMute(name) {
        if (!this.stems[name]) return;
        this.stems[name].muted = !this.stems[name].muted;
        this._updateGains();
        return this.stems[name].muted;
    }

    /** Toggle solo on a stem */
    toggleSolo(name) {
        if (!this.stems[name]) return;
        this.stems[name].soloed = !this.stems[name].soloed;
        this._updateGains();
        return this.stems[name].soloed;
    }

    /** Recalculate all gain values based on mute/solo/volume */
    _updateGains() {
        const anySoloed = Object.values(this.stems).some(s => s.soloed);
        for (const [name, stem] of Object.entries(this.stems)) {
            let effectiveVol = stem.volume;
            if (stem.muted) {
                effectiveVol = 0;
            } else if (anySoloed && !stem.soloed) {
                effectiveVol = 0;
            }
            stem.gain.gain.setTargetAtTime(effectiveVol, this.ctx.currentTime, 0.02);
        }
    }

    /** Stop all source nodes */
    _stopSources() {
        for (const stem of Object.values(this.stems)) {
            if (stem.source) {
                try { stem.source.stop(); } catch(e) {}
                stem.source.onended = null;
                stem.source = null;
            }
        }
    }

    /** Time update animation loop */
    _startTimeUpdates() {
        const tick = () => {
            this._onTimeUpdate?.(this.currentTime, this.duration);
            this._animFrame = requestAnimationFrame(tick);
        };
        tick();
    }

    _stopTimeUpdates() {
        if (this._animFrame) {
            cancelAnimationFrame(this._animFrame);
            this._animFrame = null;
        }
    }

    /**
     * Decode an audio file (ArrayBuffer) into an AudioBuffer
     * @param {ArrayBuffer} arrayBuffer
     * @returns {Promise<AudioBuffer>}
     */
    async decodeAudio(arrayBuffer) {
        this.init();
        return this.ctx.decodeAudioData(arrayBuffer);
    }

    /**
     * Export a mix (respecting current mute/solo/volume) as a WAV Blob
     * @param {Object} opts  { stems: string[]|null, format: 'wav' }
     * @returns {Blob}
     */
    exportMix(opts = {}) {
        const stemsToExport = opts.stems || this.stemOrder;
        const anySoloed = Object.values(this.stems).some(s => s.soloed);

        // Find max length
        let maxLength = 0;
        for (const name of stemsToExport) {
            const s = this.stems[name];
            if (s && s.buffer.length > maxLength) maxLength = s.buffer.length;
        }

        const numChannels = 2;
        const sampleRate = 44100;
        const mixed = [new Float32Array(maxLength), new Float32Array(maxLength)];

        for (const name of stemsToExport) {
            const stem = this.stems[name];
            if (!stem) continue;
            let vol = stem.volume;
            if (stem.muted) vol = 0;
            else if (anySoloed && !stem.soloed) vol = 0;
            if (vol === 0) continue;

            for (let ch = 0; ch < Math.min(numChannels, stem.buffer.numberOfChannels); ch++) {
                const data = stem.buffer.getChannelData(ch);
                for (let i = 0; i < data.length; i++) {
                    mixed[ch][i] += data[i] * vol;
                }
            }
        }

        return this._encodeWav(mixed, sampleRate);
    }

    /**
     * Export a single stem as WAV Blob
     * @param {string} stemName
     * @returns {Blob}
     */
    exportStem(stemName) {
        const stem = this.stems[stemName];
        if (!stem) return null;
        const channels = [];
        for (let ch = 0; ch < stem.buffer.numberOfChannels; ch++) {
            channels.push(stem.buffer.getChannelData(ch));
        }
        return this._encodeWav(channels, stem.buffer.sampleRate);
    }

    /**
     * Encode Float32 channel data to WAV Blob
     */
    _encodeWav(channels, sampleRate) {
        const numChannels = channels.length;
        const length = channels[0].length;
        const bitsPerSample = 16;
        const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
        const blockAlign = numChannels * (bitsPerSample / 8);
        const dataSize = length * blockAlign;
        const buffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(buffer);

        // WAV header
        const writeString = (offset, str) => {
            for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
        };
        writeString(0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeString(8, 'WAVE');
        writeString(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);            // PCM
        view.setUint16(22, numChannels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        writeString(36, 'data');
        view.setUint32(40, dataSize, true);

        // Interleave and convert to 16-bit
        let offset = 44;
        for (let i = 0; i < length; i++) {
            for (let ch = 0; ch < numChannels; ch++) {
                let sample = channels[ch][i];
                sample = Math.max(-1, Math.min(1, sample));
                view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                offset += 2;
            }
        }

        return new Blob([buffer], { type: 'audio/wav' });
    }
}
