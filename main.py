from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse,StreamingResponse
import cv2
import mediapipe as mp 
import numpy as np
import asyncio
from BlinkDetector import detectBlink

app = FastAPI() #App declaration

ear = None
total = None

latest_frame = None #current image

@app.websocket('/')
async def get_home():
    return HTMLResponse(content="""
    <html>
    <head>
        <title>Live Video From Flutter App</title>
    </head>
    <body>
        <h1>Live Camera From Flutter</h1>
        <img id="video-stream src="/video-stream" style="width:80%; border:1px solid black;" />
    </body>
    </html>                  
    """)
    
@app.get('/')
async def get_home():
    return HTMLResponse(content="""
    <html>
    <head>
        <title>Live Video From Flutter App</title>
    </head>
    <body>
        <h1>Live Camera From Flutter</h1>
        <img id="video-stream src="/video-stream" style="width:80%; border:1px solid black;" />
    </body>
    </html>                  
    """)
    
@app.get("/video-stream")
async def video_stream():
    async def frame_generator():
        global ear,total
        global latest_frame
        while True:
            if latest_frame is not None:
                frame_detected,ear,total = detectBlink(latest_frame)
                _,encoded_frame = cv2.imencode('.jpg',frame_detected)
                yield(b"--frame\r\n"
                      b"Content-Type: image/jpeg\r\n\r\n" +
                      bytearray(encoded_frame) + b"\r\n")
            await asyncio.sleep(0.03) #30FPS    
    return StreamingResponse(frame_generator(),media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global latest_frame
    await websocket.accept()
    print("Webscocket connection established successfully !")
    
    try:
        while True:
            data = await websocket.receive_bytes()
            frame = np.frombuffer(data,dtype=np.uint8)
            frame=cv2.imdecode(frame,cv2.IMREAD_COLOR)
            
            if frame is None:
                print("Can't decode Image")
                continue
            
            latest_frame = frame
            await websocket.send_text(f"EAR: {ear} | Total: {total}")
    except WebSocketDisconnect:
        print("WebSocket Disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Websocket connection closed")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)