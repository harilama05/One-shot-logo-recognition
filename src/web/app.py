import os
from flask import Flask
from config import Config
from extensions import socketio

# Import Blueprints
from routes.api import api_bp
from routes.views import views_bp
from events import register_events

def create_app():
    # Khởi tạo Flask app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'mat-khau-bi-mat-cua-ban'
    
    # Đăng ký Blueprints
    app.register_blueprint(api_bp)
    app.register_blueprint(views_bp)
    
    # Khởi tạo SocketIO
    socketio.init_app(app)
    register_events(socketio)
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    # Tạo các thư mục cần thiết
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.DETECTED_DIR, exist_ok=True)
    
    print("Bắt đầu ứng dụng Web nhận diện Logo...")
    print(f"Thiết bị: {Config.DEVICE}")
    print("Truy cập ứng dụng tại: http://localhost:5000")
    
    # Chạy server với socketio
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)