import os
import json
import torch

class AppConfig:
    def __init__(self):
        self.load_from_file()
    
    def load_from_file(self):
        # Tải cấu hình từ file config.json
        if os.path.exists('config.json'):
            try:
                with open('config.json', 'r') as f:
                    config_data = json.load(f)
                
                video_config = config_data.get('video', {})
                self.VIDEO_PATH = video_config.get('default_path', '../../output/query.mp4')
                
                model_config = config_data.get('models', {})
                self.MODEL_PATH = model_config.get('yolo_path', '../../weights/best.pt')
                self.RECOG_WEIGHTS = model_config.get('recognition_path', '../../weights/arcface_logo_model_best_b4_64_06.pth')
                
                detect_config = config_data.get('detection', {})
                self.CONF_THRESHOLD = detect_config.get('confidence_threshold', 0.7)
                self.RECOGNITION_THRESHOLD = detect_config.get('recognition_threshold', 0.4)
                self.BATCH_SIZE = detect_config.get('batch_size', 20)
                
                dir_config = config_data.get('directories', {})
                self.SUPPORT_DIR = dir_config.get('support', '../../output/support')
                self.MASK_DIR = dir_config.get('mask', '../../output/mask')
                self.OUTPUT_DIR = dir_config.get('output', '../../output/output_yolo')
                self.DETECTED_DIR = dir_config.get('detected', '../../output/detected_frames')
                
                self.EMBED_DB_PATH = '../../output/embedding_db.pkl'
                
                device_setting = model_config.get('device', 'auto')
                if device_setting == 'auto':
                    self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    self.DEVICE = torch.device(device_setting)
                
                print("Đã tải cấu hình từ config.json")
                
            except Exception as e:
                print(f"Lỗi tải config.json: {e}")
                self.set_defaults()
        else:
            print("Không tìm thấy config.json, dùng cài đặt mặc định")
            self.set_defaults()
    
    def set_defaults(self):
        self.VIDEO_PATH = '../../output/query.mp4'
        self.MODEL_PATH = '../../weights/best.pt'
        self.RECOG_WEIGHTS = '../../weights/arcface_logo_model_best_b4_64_06.pth'
        self.SUPPORT_DIR = '../../output/support'
        self.MASK_DIR = '../../output/mask'
        self.EMBED_DB_PATH = '../../output/embedding_db.pkl'
        self.OUTPUT_DIR = '../../output/output_yolo'
        self.DETECTED_DIR = '../../output/detected_frames'
        self.BATCH_SIZE = 20
        self.CONF_THRESHOLD = 0.7
        self.RECOGNITION_THRESHOLD = 0.4
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Biến cấu hình toàn cục
Config = AppConfig()
