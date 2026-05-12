/**
 * Waveform — Canvas-based waveform rendering for audio buffers
 */
class Waveform {
    /**
     * @param {HTMLCanvasElement} canvas
     * @param {object} opts
     */
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.color = opts.color || '#8b5cf6';
        this.bgColor = opts.bgColor || 'transparent';
        this.barWidth = opts.barWidth || 2;
        this.barGap = opts.barGap || 1;
        this.data = null;
        this._resize();
        this._resizeObserver = new ResizeObserver(() => this._resize());
        this._resizeObserver.observe(canvas);
    }

    _resize() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.width = rect.width;
        this.height = rect.height;
        if (this.data) this.draw(this.data);
    }

    /**
     * Compute peaks from an AudioBuffer (averages all channels)
     * @param {AudioBuffer} buffer
     * @returns {Float32Array} normalized peak values
     */
    static computePeaks(buffer, numBars = 200) {
        const channels = buffer.numberOfChannels;
        const length = buffer.length;
        const samplesPerBar = Math.floor(length / numBars);
        const peaks = new Float32Array(numBars);

        for (let i = 0; i < numBars; i++) {
            let sum = 0;
            const start = i * samplesPerBar;
            const end = Math.min(start + samplesPerBar, length);
            for (let ch = 0; ch < channels; ch++) {
                const channelData = buffer.getChannelData(ch);
                for (let s = start; s < end; s++) {
                    sum += Math.abs(channelData[s]);
                }
            }
            peaks[i] = sum / ((end - start) * channels);
        }

        // Normalize
        let max = 0;
        for (let i = 0; i < numBars; i++) {
            if (peaks[i] > max) max = peaks[i];
        }
        if (max > 0) {
            for (let i = 0; i < numBars; i++) {
                peaks[i] /= max;
            }
        }
        return peaks;
    }

    /**
     * Draw waveform bars from peak data
     * @param {Float32Array} peaks
     * @param {number} [progress] 0-1 to highlight played portion
     */
    draw(peaks, progress = -1) {
        this.data = peaks;
        const { ctx, width, height, barWidth, barGap, color, bgColor } = this;

        ctx.clearRect(0, 0, width, height);
        if (bgColor !== 'transparent') {
            ctx.fillStyle = bgColor;
            ctx.fillRect(0, 0, width, height);
        }

        const totalBarWidth = barWidth + barGap;
        const numBars = Math.floor(width / totalBarWidth);
        const step = peaks.length / numBars;
        const mid = height / 2;
        const maxBarHeight = height * 0.8;

        for (let i = 0; i < numBars; i++) {
            const peakIndex = Math.floor(i * step);
            const value = peaks[Math.min(peakIndex, peaks.length - 1)];
            const barHeight = Math.max(2, value * maxBarHeight);
            const x = i * totalBarWidth;

            if (progress >= 0 && (i / numBars) <= progress) {
                ctx.fillStyle = color;
                ctx.globalAlpha = 1;
            } else if (progress >= 0) {
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.3;
            } else {
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.6;
            }

            // Draw symmetric bar around center
            ctx.beginPath();
            ctx.roundRect(x, mid - barHeight / 2, barWidth, barHeight, 1);
            ctx.fill();
        }
        ctx.globalAlpha = 1;
    }

    destroy() {
        this._resizeObserver?.disconnect();
    }
}

// Polyfill roundRect if not available
if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
        this.rect(x, y, w, h);
    };
}
