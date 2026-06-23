// static/ttsPlaybackProcessor.js

class TTSPlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferQueue = [];
    this.readOffset = 0;
    this.samplesRemaining = 0;
    this.isPlaying = false;

    // We resample from 48000 Hz to the AudioContext's current sampleRate
    this.sourceSampleRate = 48000;
    this.targetSampleRate = sampleRate; // global variable in AudioWorkletGlobalScope

    // Fractional position in the current buffer
    this.fractionalPosition = 0.0;
    this.lastSample = 0.0;

    // Listen for incoming messages
    this.port.onmessage = (event) => {
      // Check if this is a control message (object with a "type" property).
      if (event.data && typeof event.data === "object" && event.data.type === "clear") {
        // Clear the TTS buffer and reset playback state.
        this.bufferQueue = [];
        this.readOffset = 0;
        this.samplesRemaining = 0;
        this.isPlaying = false;
        this.fractionalPosition = 0.0;
        this.lastSample = 0.0;
        return;
      }
      
      // Otherwise assume it's a PCM chunk (an Int16Array)
      this.bufferQueue.push(event.data);
      this.samplesRemaining += event.data.length;
    };
  }

  process(inputs, outputs) {
    const outputChannel = outputs[0][0];

    if (this.samplesRemaining === 0 && this.bufferQueue.length === 0) {
      outputChannel.fill(0);
      if (this.isPlaying) {
        this.isPlaying = false;
        this.port.postMessage({ type: 'ttsPlaybackStopped' });
      }
      return true;
    }

    if (!this.isPlaying) {
      this.isPlaying = true;
      this.port.postMessage({ type: 'ttsPlaybackStarted' });
    }

    const ratio = this.sourceSampleRate / this.targetSampleRate;
    let outIdx = 0;

    while (outIdx < outputChannel.length) {
      if (this.bufferQueue.length === 0) {
        break;
      }

      const currentBuffer = this.bufferQueue[0];
      const index = Math.floor(this.fractionalPosition);
      const nextIndex = index + 1;
      const weight = this.fractionalPosition - index;

      let sample = 0;
      let nextSample = 0;

      if (index < currentBuffer.length) {
        sample = currentBuffer[index] / 32768.0;
      } else {
        sample = this.lastSample;
      }

      if (nextIndex < currentBuffer.length) {
        nextSample = currentBuffer[nextIndex] / 32768.0;
      } else {
        // If next sample is in the next buffer in queue
        if (this.bufferQueue.length > 1) {
          nextSample = this.bufferQueue[1][nextIndex - currentBuffer.length] / 32768.0;
        } else {
          nextSample = sample;
        }
      }

      // Linear interpolation
      const interpolatedSample = sample + weight * (nextSample - sample);
      outputChannel[outIdx++] = interpolatedSample;

      // Advance position
      this.fractionalPosition += ratio;

      // Check if we consumed the current buffer
      if (this.fractionalPosition >= currentBuffer.length) {
        this.fractionalPosition -= currentBuffer.length;
        this.samplesRemaining -= currentBuffer.length;
        this.lastSample = currentBuffer[currentBuffer.length - 1] / 32768.0;
        this.bufferQueue.shift();
      }
    }

    // Fill remaining output channel with silence if queue ran out mid-block
    while (outIdx < outputChannel.length) {
      outputChannel[outIdx++] = 0;
    }

    return true;
  }
}

registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
