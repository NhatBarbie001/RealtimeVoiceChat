// static/pcmWorkletProcessor.js
// Dynamic downsampling to produce 16kHz PCM for the STT server.

class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sourceSampleRate = sampleRate; // global variable in AudioWorkletGlobalScope
    this.targetSampleRate = 16000;
    this.lastSample = 0.0;
    this.lastPosition = 0.0;
  }

  process(inputs) {
    const in32 = inputs[0][0];
    if (in32 && in32.length > 0) {
      const ratio = this.sourceSampleRate / this.targetSampleRate;
      const outSamples = [];
      let pos = this.lastPosition;

      while (pos < in32.length) {
        const index = Math.floor(pos);
        const nextIndex = index + 1;
        const weight = pos - index;

        const sample = index >= 0 ? in32[index] : this.lastSample;
        const nextSample = nextIndex < in32.length ? in32[nextIndex] : in32[in32.length - 1];

        // Linear interpolation
        const interpolated = sample + weight * (nextSample - sample);

        // Convert float to Int16 range
        let s = interpolated < -1 ? -1 : interpolated > 1 ? 1 : interpolated;
        const val = s < 0 ? s * 0x8000 : s * 0x7FFF;
        outSamples.push(val);

        pos += ratio;
      }

      // Save state for next block
      this.lastPosition = pos - in32.length;
      if (in32.length > 0) {
        this.lastSample = in32[in32.length - 1];
      }

      if (outSamples.length > 0) {
        const int16 = new Int16Array(outSamples);
        // send raw ArrayBuffer, transferable
        this.port.postMessage(int16.buffer, [int16.buffer]);
      }
    }
    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);
