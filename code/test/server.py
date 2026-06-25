import os
import shutil
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Import our custom modules
from download_models import download_all_models, MODELS_DIR
from vad_module import StreamingVAD
from stt_module import GipformerSTT, ResultsRegistry, BatchTranscriptionWorker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("server")

TEMP_DIR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_utterances")

''' <- Đây là một Decorator. Nó đóng gói hàm lifespan phía dưới thành một "quản lý ngữ cảnh bất đồng bộ".
Nó báo cho FastAPI biết: "Hãy coi hàm này là nơi quản lý vòng đời của app. Khi nào mở app thì chạy phần đầu,
khi nào đóng app thì chạy phần đuôi".
'''
@asynccontextmanager 
async def lifespan(app: FastAPI):
    '''Hiểu một cách đơn giản, đoạn code này kiểm soát những gì sẽ xảy ra 
    ngay trước khi server bắt đầu nhận request và ngay sau khi server chuẩn bị tắt hẳn.'''
    # 1. Download models if not exists
    logger.info("Checking and downloading models...")
    download_all_models()
    
    # 2. Initialize STT and VAD paths
    gipformer_dir = os.path.join(MODELS_DIR, "gipformer")
    vad_model_path = os.path.join(MODELS_DIR, "vad", "silero_vad.onnx")
    
    logger.info("Initializing Gipformer STT engine...")
    '''Cú pháp app.state. <tên_biến>: Vai trò: app.state là một "kho lưu trữ toàn cục" (Global Storage) do FastAPI cung cấp.
    Tại sao phải dùng? Vì hàm lifespan này chạy độc lập. Để các hàm xử lý API khác (sau này bạn viết) có thể dùng chung
    bộ AI GipformerSTT hay worker này, bạn phải "gắn" chúng vào app.state. Khi đó, ở bất kỳ API endpoint nào,
    bạn chỉ cần gọi request.app.state.stt là dùng được ngay, không phải khởi tạo lại từ đầu (tiết kiệm RAM và CPU tối đa).
    '''
    app.state.stt = GipformerSTT(gipformer_dir)
    app.state.vad_model_path = vad_model_path
    
    # 3. Initialize Registry and Batch Worker
    '''
        app.state.worker.start(): Kích hoạt một tiến trình chạy ngầm (background worker) 
        để gom các file âm thanh vào xử lý hàng loạt (batch).
    '''
    app.state.registry = ResultsRegistry()
    app.state.worker = BatchTranscriptionWorker(app.state.stt, app.state.registry)
    app.state.worker.start()
    
    # Clean temporary folder on startup
    if os.path.exists(TEMP_DIR_ROOT):
        shutil.rmtree(TEMP_DIR_ROOT)
    os.makedirs(TEMP_DIR_ROOT, exist_ok=True)
    
    yield # <- ngăn cách app thành 2 giai đoạn start -> running -> shutdown
    
    # Cleanup on shutdown
    logger.info("Shutting down worker...")
    '''
        Cú pháp await: Vì đây là hàm bất đồng bộ (async def), các hành động tốn thời gian
        (như đóng một worker đang chạy ngầm) cần có await để Python biết rằng nó cần đợi
        hành động này hoàn thành xong xuôi rồi mới đi tiếp.
    '''
    await app.state.worker.stop()
    if os.path.exists(TEMP_DIR_ROOT):
        shutil.rmtree(TEMP_DIR_ROOT, ignore_errors=True)
# ============================================================================================================
'''
Tạo FastAPI app
Và mount thư mục static để phục vụ file HTML, CSS, JS:
'''
app = FastAPI(lifespan=lifespan)

# Mount static folder
static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")
# ============================================================================================================
'''
Đoạn code này của bạn là một API endpoint cơ bản trong FastAPI
 dùng để đọc một file HTML từ ổ cứng và trả về cho trình duyệt hiển thị (giao diện trang chủ).
'''

'''
@app: Gọi đến instance của FastAPI mà bạn đã khởi tạo trước đó.

+ .get("/"):
+  get: Định nghĩa HTTP method là GET (dùng khi trình duyệt muốn yêu cầu lấy dữ liệu/giao diện).
+  "/": Đây là Path (Đường dẫn). Dấu gạch chéo đơn độc này đại diện cho trang chủ (Root URL), ví dụ: http://127.0.0.1:8000/.
'''
@app.get("/")
async def get_index():
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)

#================================================================================================================
'''
+ @app.websocket("/ws"): Định nghĩa một đường ống giao tiếp hai chiều (Full-duplex) qua giao thức WebSocket tại đường dẫn /ws. 
Khác với HTTP thông thường (vào rồi ra), WebSocket giữ kết nối luôn mở. 
+ await websocket.accept(): Chấp nhận kết nối từ client. Từ giây phút này, server và client có thể chủ động "nói chuyện" 
với nhau bất cứ lúc nào.
'''
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    #==================================================================================
    '''
    Vai trò: Tạo ra một ID ngẫu nhiên (session_id) và một thư mục tạm thời riêng biệt
     cho mỗi người dùng kết nối vào. Mục đích là để lưu trữ các file âm thanh cắt nhỏ 
     của riêng người đó, tránh lẫn lộn dữ liệu giữa các client.
    '''
    session_id = os.urandom(8).hex()
    session_temp_dir = os.path.join(TEMP_DIR_ROOT, session_id)
    os.makedirs(session_temp_dir, exist_ok=True)

    logger.info(f"Client connected. Session: {session_id}")
    #==================================================================================
    '''
        Mối liên kết với câu hỏi trước: Bạn có nhớ app.state ở câu hỏi đầu tiên không?
        Ở đây, ta rút registry, worker, và cấu hình model từ websocket.app.state ra để dùng.

        VAD (Voice Activity Detection): Đây là bộ phát hiện tiếng người. Nó có nhiệm vụ "nghe" 
        luồng âm thanh liên tục, nhận biết khi nào người dùng bắt đầu nói và dừng nói 
        để cắt thành từng câu (utterance).
    '''
    # Initialize VAD for this connection session
    vad = StreamingVAD(
        model_path=websocket.app.state.vad_model_path,
        temp_dir=session_temp_dir
    )
    
    registry = websocket.app.state.registry
    worker = websocket.app.state.worker
    # =================================================================================
    
    # Queue for tracking utterance IDs generated in this session
    session_utterance_queue = asyncio.Queue()
    
    # Consumer task to wait for results and send them back to client
    # =================================================================================
    async def consumer():
        try:
            while True:
                u_id = await session_utterance_queue.get()
                # Wait for transcription to complete in registry
                logger.info(f"Consumer waiting for result of utterance: {u_id}")
                text = await registry.get_result(u_id)
                logger.info(f"Consumer sending result for utterance {u_id}: {text}")
                
                # Send back to client
                await websocket.send_json({
                    "text": text,
                    "is_final": True
                })
                session_utterance_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in WebSocket consumer task: {e}")

    consumer_task = asyncio.create_task(consumer())
    # =================================================================================
    try:
        while True:
            # Receive raw binary PCM from client
            data = await websocket.receive_bytes()
            
            # Feed to VAD to detect segments
            utterances = vad.feed(data)
            for u_id, wav_path in utterances:
                # Register in results registry
                registry.register_utterance(u_id)
                # Queue for STT batch worker
                await worker.queue.put((u_id, wav_path))
                # Track in session queue so the consumer task can wait for it
                await session_utterance_queue.put(u_id)
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected. Session: {session_id}")
    except Exception as e:
        logger.error(f"Error in websocket loop: {e}")
    finally:
        # Flush any remaining audio segments
        try:
            remaining = vad.flush()
            for u_id, wav_path in remaining:
                registry.register_utterance(u_id)
                await worker.queue.put((u_id, wav_path))
                await session_utterance_queue.put(u_id)
        except Exception as e:
            logger.error(f"Error flushing VAD on disconnect: {e}")
            
        # Cancel the consumer task
        consumer_task.cancel()
        try:
            await consumer_task
        except Exception:
            pass
            
        # Clean up session directory
        try:
            if os.path.exists(session_temp_dir):
                shutil.rmtree(session_temp_dir)
        except Exception as e:
            logger.error(f"Error clearing session folder {session_temp_dir}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8001, reload=False)
