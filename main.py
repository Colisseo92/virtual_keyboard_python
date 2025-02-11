from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse,StreamingResponse
import asyncio
import cv2
import mediapipe as mp 
import numpy as np
import asyncio
import base64
from BlinkDetector import detectBlink
from wordPredictor import init_module,getPredictions

app = FastAPI() #App declaration

latest_frame = None
is_searching = False
last_word = ""

@app.get("/")
async def get_home():
    """Serve the main HTML page with video streaming."""
    return HTMLResponse(content="""
    <html>
    <head>
        <title>Live Video Stream</title>
    </head>
    <body>
        <h1>Live Video Stream</h1>
        <img id="video-stream" src="/video-stream" style="width:80%; border:1px solid black;" />
    </body>
    </html>
    """)

@app.get("/video-stream")
async def video_stream():
    """Stream the latest video frame."""
    async def frame_generator():
        global latest_frame
        while True:
            if latest_frame is not None:
                # Encode the frame to JPEG and yield it
                _, encoded_frame = cv2.imencode(".jpg", latest_frame)
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       bytearray(encoded_frame) + b"\r\n")
            await asyncio.sleep(0.01)  # Approx 30 FPS

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global latest_frame
    await websocket.accept()
    print("Connection accepted !")
    try:
        while True:
            data = await websocket.receive_bytes()
            
            frame = np.frombuffer(data, dtype=np.uint8)
            image = cv2.imdecode(frame,cv2.IMREAD_COLOR)
            #print(f"IMAGE: {image}")
            latest_frame = image
            
    except Exception as e:
        print(f'{e}')
    
@app.websocket("/text")
async def text_endpoint(websocket: WebSocket):
    global is_searching, last_word
    await websocket.accept()
    print("Text connection accepted!")
    try:
        while True:
            text = await websocket.receive_text()
            print(f"TEXT: {text}")
            if is_searching == False and last_word != text:
                is_searching = True
                await websocket.send_json(getPredictions(text))
                print("Searched ended")
                last_word = text
                is_searching = False
    except Exception as e:
        print(f'Text error: {e}')     

        
@app.get('/')
async def get():
    return HTMLResponse('WebSocket server is running')

if __name__ == "__main__":
    import uvicorn
    init_module()
    uvicorn.run(app,host="127.0.0.1",port=8000)