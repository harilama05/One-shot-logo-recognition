# setup.py

#!/usr/bin/env python3
"""
Script cài đặt cho Ứng dụng Web Nhận diện Logo
Kiểm tra môi trường và tạo các thư mục cần thiết
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    # Kiểm tra phiên bản Python có tương thích không
    if sys.version_info < (3, 8):
        print("Yêu cầu Python 3.8+")
        return False
    print(f"Python {sys.version.split()[0]} tương thích")
    return True

def check_and_install_requirements():
    # Kiểm tra và cài đặt các gói phụ thuộc từ requirements.txt
    print("\nKiểm tra các gói phụ thuộc...")
    try:
        import pkg_resources
        with open('requirements.txt', 'r') as f:
            requirements = f.read().splitlines()
        
        installed = {pkg.key for pkg in pkg_resources.working_set}
        missing = []
        
        # Tìm các gói còn thiếu
        for requirement in requirements:
            if '==' in requirement:
                package_name = requirement.split('==')[0].lower()
            elif '>=' in requirement:
                package_name = requirement.split('>=')[0].lower()
            else:
                package_name = requirement.lower()
            
            if package_name not in installed:
                missing.append(requirement)
        
        # Nếu thiếu thì cài đặt
        if missing:
            print(f"Các gói còn thiếu: {', '.join(missing)}")
            print("Đang cài đặt các gói còn thiếu...")
            for package in missing:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install',
                    '--trusted-host', 'pypi.org',
                    '--trusted-host', 'files.pythonhosted.org',
                    package
                ])
            print("Đã cài đặt tất cả các gói phụ thuộc")
        else:
            print("Tất cả các gói phụ thuộc đã được cài đặt")
        return True
        
    except Exception as e:
        print(f"Lỗi khi kiểm tra các gói phụ thuộc: {e}")
        return False

def create_directories():
    # Tạo các thư mục cần thiết cho ứng dụng
    print("\nĐang tạo các thư mục...")
    config_dirs = [
        'support',
        'output_yolo',
        'templates',
        'static'
    ]
    # Thử đọc thêm thư mục từ file config
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            dirs = config.get('directories', {})
            # Thêm các thư mục trong config vào danh sách
            for v in dirs.values():
                if v not in config_dirs:
                    config_dirs.append(v)
        except Exception as e:
            print(f"Không thể đọc thư mục từ config.json: {e}")
    for directory in config_dirs:
        Path(directory).mkdir(exist_ok=True)
        print(f"Đã tạo/xác minh: {directory}/")

def check_model_files():
    # Kiểm tra xem các file model có tồn tại không
    print("\nKiểm tra các file model...")
    model_files = [
        'best 2.pt',
        'arcface_logo_model_best.pth'
    ]
    
    missing_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"Tìm thấy: {model_file}")
        else:
            print(f"Thiếu: {model_file}")
            missing_models.append(model_file)
    
    if missing_models:
        print(f"\nThiếu các file model: {', '.join(missing_models)}")
        print("Vui lòng đảm bảo các file này có trong thư mục dự án trước khi chạy.")
        return False
    return True

def check_config_file():
    # Kiểm tra và xác thực file cấu hình
    print("\nKiểm tra cấu hình...")
    
    # Ưu tiên dùng file config_clean.json nếu có
    if os.path.exists('config_clean.json'):
        print("Tìm thấy config_clean.json")
        import shutil
        shutil.copy2('config_clean.json', 'config.json')
        print("Đã sao chép thành config.json")
    
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            print("File cấu hình hợp lệ")
            return True
        except json.JSONDecodeError as e:
            print(f"JSON không hợp lệ trong config.json: {e}")
            return False
    else:
        print("Không tìm thấy config.json")
        return False

def check_video_file():
    # Kiểm tra xem file video mẫu có tồn tại không
    print("\nKiểm tra file video...")
    if os.path.exists('query.mp4'):
        print("Tìm thấy: query.mp4")
        return True
    else:
        print("Thiếu: query.mp4")
        print("Vui lòng thêm một file video có tên 'query.mp4' để thử nghiệm.")
        return False

def main():
    # Hàm cài đặt chính
    print("Cài đặt Ứng dụng Web Nhận diện Logo")
    print("=" * 50)
    
    success = True
    
    success &= check_python_version()
    success &= check_and_install_requirements()
    create_directories()
    models_ok = check_model_files()
    config_ok = check_config_file()
    video_ok = check_video_file()
    
    print("\n" + "=" * 50)
    if success and config_ok:
        print("Cài đặt hoàn tất!")
        if not models_ok:
            print("Cảnh báo: Thiếu file model")
        if not video_ok:
            print("Cảnh báo: Thiếu file video")
    else:
        print("Cài đặt thất bại. Vui lòng sửa các lỗi trên.")
        sys.exit(1)

if __name__ == "__main__":
    main()