import os
import uuid
import wave
import numpy as np
import sherpa_onnx
import logging

logger = logging.getLogger("vad_module")

class StreamingVAD:
    def __init__(self, model_path: str, temp_dir: str, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Configure VAD model
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = model_path
        config.silero_vad.threshold = 0.5
        config.silero_vad.min_silence_duration = 0.5  # seconds of silence to split utterance
        config.silero_vad.min_speech_duration = 0.25  # seconds of speech to be considered valid
        config.sample_rate = sample_rate
        
        logger.info(f"Initializing Silero VAD with model: {model_path}")
        self.detector = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)
        
    def feed(self, pcm_chunk: bytes) -> list:
        """
        Feeds a chunk of raw PCM int16 audio from client.
        Returns a list of tuples: (utterance_id, wav_path) of detected speech utterances.
        """
        if not pcm_chunk:
            return []
            
        # Convert raw int16 PCM bytes to float32 normalized [-1.0, 1.0]
        samples_int16 = np.frombuffer(pcm_chunk, dtype=np.int16)
        if len(samples_int16) == 0:
            return []
            
        samples_float32 = samples_int16.astype(np.float32) / 32768.0
        
        # Feed to the VAD detector
        self.detector.accept_waveform(samples_float32)
        
        utterances = []
        
        # Retrieve all completed speech segments from VAD queue
        while not self.detector.empty():
            segment = self.detector.front
            # segment.samples contains the float32 waveform
            # segment.start contains the start index (in samples)
            
            utterance_id = str(uuid.uuid4())
            wav_path = os.path.join(self.temp_dir, f"{utterance_id}.wav")
            
            # Save speech segment as WAV file
            self.save_wav(wav_path, segment.samples)
            
            logger.info(f"VAD detected speech segment. Saved to {wav_path} (length: {len(segment.samples)/self.sample_rate:.2f}s)")
            
            utterances.append((utterance_id, wav_path))
            self.detector.pop()
            
        return utterances

    def save_wav(self, path: str, samples: np.ndarray):
        # Clip to ensure valid float values, then scale to 16-bit integer
        int_samples = np.clip(samples, -1.0, 1.0)
        int_samples = (int_samples * 32767).astype(np.int16)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(int_samples.tobytes())
            
    def flush(self) -> list:
        """
        Flush remaining audio in detector to finalize any ongoing speech segment.
        """
        self.detector.flush()
        return self.feed(b"") # Run feed with empty bytes to drain any final segments
