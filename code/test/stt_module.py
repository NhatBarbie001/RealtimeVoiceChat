import os
import wave
import asyncio
import numpy as np
import sherpa_onnx
import logging
import time

logger = logging.getLogger("stt_module")

def get_best_provider() -> str:
    """
    Detects if CUDA or TensorRT execution providers are available in onnxruntime,
    and returns 'cuda' or 'cpu'.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        logger.info(f"Available ONNXRuntime providers: {available}")
        if 'TensorrtExecutionProvider' in available:
            return 'trt'
        elif 'CUDAExecutionProvider' in available:
            return 'cuda'
    except Exception as e:
        logger.warning(f"Could not import onnxruntime or detect providers: {e}")
    return 'cpu'

class ResultsRegistry:
    def __init__(self):
        self.results = {}
        self.events = {}

    def register_utterance(self, utterance_id: str):
        self.results[utterance_id] = None
        self.events[utterance_id] = asyncio.Event()

    def set_result(self, utterance_id: str, text: str):
        self.results[utterance_id] = text
        if utterance_id in self.events:
            self.events[utterance_id].set()

    async def get_result(self, utterance_id: str) -> str:
        if utterance_id not in self.events:
            return ""
        await self.events[utterance_id].wait()
        text = self.results.get(utterance_id, "")
        # Clean up registry memory
        self.results.pop(utterance_id, None)
        self.events.pop(utterance_id, None)
        return text

class GipformerSTT:
    def __init__(self, model_dir: str):
        encoder = os.path.join(model_dir, "encoder.onnx")
        decoder = os.path.join(model_dir, "decoder.onnx")
        joiner = os.path.join(model_dir, "joiner.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")
        
        for p in [encoder, decoder, joiner, tokens]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing required Gipformer model file: {p}")
                
        provider = get_best_provider()
        logger.info(f"Initializing Gipformer STT with provider: {provider}")
        
        # Gipformer is an offline (non-streaming) model
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=4,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="modified_beam_search",
            provider=provider
        )
        logger.info("Gipformer STT successfully initialized.")

    def transcribe_batch_sync(self, wav_paths: list[str]) -> list[str]:
        """
        Synchronously transcribes a batch of audio files.
        Must be run in a thread executor if called from async loops.
        """
        if not wav_paths:
            return []
            
        streams = []
        for path in wav_paths:
            stream = self.recognizer.create_stream()
            # Read wave file samples
            with wave.open(path, 'rb') as wf:
                params = wf.getparams()
                frames = wf.readframes(params.nframes)
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            stream.accept_waveform(16000, samples)
            streams.append(stream)
            
        # Perform batch inference
        start_time = time.time()
        self.recognizer.decode_streams(streams)
        latency = time.time() - start_time
        logger.info(f"Decoded batch of {len(wav_paths)} items in {latency:.4f}s")
        
        results = []
        for stream in streams:
            results.append(stream.result.text.strip())
            
        return results

class BatchTranscriptionWorker:
    def __init__(self, stt: GipformerSTT, registry: ResultsRegistry):
        self.stt = stt
        self.registry = registry
        self.queue = asyncio.Queue()
        self.is_running = False
        self.task = None

    def start(self):
        self.is_running = True
        self.task = asyncio.create_task(self._worker_loop())
        logger.info("BatchTranscriptionWorker started.")

    async def stop(self):
        self.is_running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("BatchTranscriptionWorker stopped.")

    async def _worker_loop(self):
        loop = asyncio.get_running_loop()
        while self.is_running:
            try:
                # Wait for the first item in the batch
                item = await self.queue.get()
                batch = [item]
                
                # Wait up to 15ms to accumulate up to 4 items total
                start_wait = time.time()
                while len(batch) < 4:
                    elapsed = (time.time() - start_wait) * 1000
                    timeout = max(0.0, 15.0 - elapsed) / 1000.0
                    try:
                        next_item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        batch.append(next_item)
                    except asyncio.TimeoutError:
                        break
                
                # Prepare batch items
                utterance_ids = [b[0] for b in batch]
                wav_paths = [b[1] for b in batch]
                
                # Perform sync inference in thread executor to prevent event loop blocking
                logger.info(f"Worker processing batch of {len(batch)} items: {utterance_ids}")
                transcriptions = await loop.run_in_executor(
                    None, 
                    self.stt.transcribe_batch_sync, 
                    wav_paths
                )
                
                # Populate registry and delete files
                for u_id, path, text in zip(utterance_ids, wav_paths, transcriptions):
                    # Save results to registry
                    self.registry.set_result(u_id, text)
                    # Delete temporary WAV file
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception as e:
                        logger.error(f"Error removing temp WAV file {path}: {e}")
                        
                # Mark items as processed in the queue
                for _ in range(len(batch)):
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch transcription worker loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)
