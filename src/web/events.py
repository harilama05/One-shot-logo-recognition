import time
import traceback
from flask import request
from services.video_service import video_processor
from extensions import socketio

def register_events(app_socketio):
    @app_socketio.on('connect')
    def handle_connect():
        print(f"Client đã kết nối: {request.sid}")
        socketio.emit('status', {'message': 'Đã kết nối tới server'})

    @socketio.on('disconnect')
    def handle_disconnect():
        print(f"Client đã ngắt kết nối: {request.sid}")

def process_video_thread():
    try:
        frame_delay = 1.0 / video_processor.fps if video_processor.fps > 0 else 1.0 / 30
        
        while video_processor.is_processing and video_processor.cap:
            ret, frame = video_processor.cap.read()
            
            if not ret:
                socketio.emit('video_ended', {'message': 'Xử lý video hoàn tất'})
                break
            
            results = video_processor.process_frame(frame)
            socketio.emit('frame_processed', results)
            
            time.sleep(frame_delay)
            
    except Exception as e:
        print(f"Lỗi trong thread xử lý video: {e}")
        traceback.print_exc()
        socketio.emit('error', {'message': f'Lỗi xử lý video: {str(e)}'})
    
    finally:
        video_processor.is_processing = False
        if video_processor.cap:
            video_processor.cap.release()
