import os
from collections import defaultdict
from flask import Blueprint, jsonify, request

from config import Config
from utils import load_embeddings, image_to_thumb_base64
from services.video_service import video_processor
from services.registry_service import LogoRegistry

api_bp = Blueprint('api', __name__)

@api_bp.route('/api/load_models', methods=['POST'])
def load_models():
    try:
        success = video_processor.load_models()
        return jsonify({'success': success, 'message': 'Tải model thành công' if success else 'Tải model thất bại'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@api_bp.route('/api/register_support', methods=['POST'])
def register_support():
    try:
        data = request.get_json()
        support_dir = data.get('support_dir', Config.SUPPORT_DIR)
        mask_dir = data.get('mask_dir', Config.MASK_DIR)
        
        success = LogoRegistry.register_support_folder(support_dir, mask_dir)
        if success:
            video_processor.load_database()
            
        return jsonify({'success': success, 'message': 'Đăng ký support thành công' if success else 'Đăng ký support thất bại'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@api_bp.route('/api/register_detection', methods=['POST'])
def register_detection():
    try:
        data = request.get_json()
        detection_id = data.get('detection_id')
        class_name = data.get('class_name')
        
        if not detection_id or not class_name:
            return jsonify({'success': False, 'message': 'Thiếu detection_id hoặc class_name'})
        
        crop_path = os.path.join(Config.DETECTED_DIR, f"{detection_id}_crop.jpg")
        mask_path = os.path.join(Config.DETECTED_DIR, f"{detection_id}_mask.png")
        
        if not os.path.exists(crop_path):
            return jsonify({'success': False, 'message': 'Không tìm thấy ảnh crop'})
        
        success = LogoRegistry.register_single_logo(crop_path, mask_path if os.path.exists(mask_path) else None, class_name)
        
        if success:
            video_processor.load_database()
            
        return jsonify({'success': success, 'message': f'Đã đăng ký detection là {class_name}' if success else 'Đăng ký detection thất bại'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@api_bp.route('/api/start_video', methods=['POST'])
def start_video():
    from threading import Thread
    from events import process_video_thread
    try:
        data = request.get_json()
        video_path = data.get('video_path', Config.VIDEO_PATH)
        
        if video_processor.is_processing:
            return jsonify({'success': False, 'message': 'Video đang được xử lý'})
        
        success = video_processor.open_video(video_path)
        if success:
            video_processor.is_processing = True
            thread = Thread(target=process_video_thread)
            thread.daemon = True
            thread.start()
            
        return jsonify({'success': success, 'message': 'Bắt đầu video thành công' if success else 'Bắt đầu video thất bại'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@api_bp.route('/api/stop_video', methods=['POST'])
def stop_video():
    try:
        video_processor.is_processing = False
        if video_processor.cap:
            video_processor.cap.release()
        return jsonify({'success': True, 'message': 'Đã dừng video'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@api_bp.route('/api/get_database_info', methods=['GET'])
def get_database_info():
    try:
        db = load_embeddings(Config.EMBED_DB_PATH)
        class_counts = defaultdict(int)
        for _, class_name in db:
            class_counts[class_name] += 1
        
        return jsonify({
            'success': True,
            'total_embeddings': len(db),
            'classes': dict(class_counts)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@api_bp.route('/api/get_detections', methods=['GET'])
def get_detections():
    try:
        include_base64 = request.args.get('base64', '1') == '1'
        batch_start_time = request.args.get('batch_start_time', None)
        detections = []
        
        if os.path.exists(Config.DETECTED_DIR):
            try:
                files = os.listdir(Config.DETECTED_DIR)
            except (OSError, PermissionError):
                files = []
                
            for f in files:
                if not f.endswith('_crop.jpg'):
                    continue
                    
                detection_id = f[:-9]
                crop_path = os.path.join(Config.DETECTED_DIR, f)
                
                if not os.path.exists(crop_path) or not os.path.isfile(crop_path):
                    continue
                    
                try:
                    size = os.path.getsize(crop_path)
                    mtime = os.path.getmtime(crop_path)
                except (OSError, PermissionError):
                    continue
                
                if batch_start_time and mtime < float(batch_start_time):
                    continue
                    
                mask_file = f"{detection_id}_mask.png"
                mask_path = os.path.join(Config.DETECTED_DIR, mask_file)
                
                thumb_b64 = None
                if include_base64:
                    try:
                        thumb_b64 = image_to_thumb_base64(crop_path)
                    except Exception as e:
                        print(f"Lỗi thumbnail cho {crop_path}: {e}")
                        thumb_b64 = None
                
                detections.append({
                    'id': detection_id,
                    'crop_path': f"/detected/{f}?t={int(mtime)}",
                    'has_mask': os.path.exists(mask_path) and os.path.isfile(mask_path),
                    'size': size,
                    'mtime': mtime,
                    'thumb': thumb_b64
                })
                
        detections.sort(key=lambda x: x['mtime'], reverse=True)
        return jsonify({'success': True, 'detections': detections})
        
    except Exception as e:
        print(f"Lỗi trong get_detections: {e}")
        return jsonify({'success': False, 'message': str(e), 'detections': []})
