# One-shot Logo Recognition

Dự án này triển khai một hệ thống nhận diện logo one-shot trên video hoặc hình ảnh. Hệ thống kết hợp giữa phát hiện và phân vùng đối tượng (Detection & Segmentation) bằng YOLO với trích xuất đặc trưng và so khớp (Feature Extraction & Recognition) bằng ArcFace (sử dụng kiến trúc EfficientNet). Điều này cho phép hệ thống nhận diện các logo mới chỉ thông qua một ảnh mẫu (one-shot) mà không cần phải huấn luyện lại (retrain) toàn bộ mô hình.

---

## Kiến trúc Pipeline (Pipeline Architecture)

Quá trình xử lý luồng dữ liệu (video) đi qua các bước (Worker) như sau:

1. Input Worker (Đầu vào): Đọc luồng video/hình ảnh đầu vào và trích xuất từng khung hình (frame).
2. YOLO Detect Worker (Phát hiện & Phân vùng):
   - Sử dụng mô hình YOLO (`best.pt`) để phát hiện vùng chứa logo (Bounding Box).
   - Tiến hành phân vùng đối tượng (Segmentation Mask) nhằm loại bỏ nền (background) gây nhiễu, giúp việc trích xuất đặc trưng logo chính xác hơn.
3. ArcFace Recognition Worker (Trích xuất & So khớp đặc trưng):
   - Các vùng chứa logo đã được crop (và làm sạch bằng mask) sẽ được đưa qua mạng nơ-ron ArcFace (sử dụng file `arcface_logo_model_best_b4_64_06.pth`).
   - ArcFace chuyển đổi ảnh logo thành một vector đặc trưng (Embedding 1D size 512).
   - Hệ thống so sánh khoảng cách/độ tương đồng Cosine của embedding này với cơ sở dữ liệu các logo đã biết (`embedding_db.pkl`). Nếu độ tương đồng vượt ngưỡng (threshold), logo sẽ được gắn nhãn (label) tương ứng.
4. Post-process & Output Worker (Hậu xử lý & Đầu ra):
   - Tổng hợp kết quả nhận diện. Vẽ bounding box, label và điểm tin cậy (confidence score) lên frame gốc.
   - Ghi các khung hình đã được xử lý thành video đầu ra.

---

## Cấu trúc Repository (Mới cập nhật - Chuẩn OOP)

```text
one-shot-logo-recognition/
├── oslr/                  # Chứa toàn bộ core pipeline CLI (package & các Worker).
├── scripts/               # Chứa các script chạy nhanh (entrypoints) từ CLI.
├── web/                   # Ứng dụng Web Demo trực quan (Kiến trúc OOP/MVC).
│   ├── app.py             # Entrypoint khởi tạo Flask app & SocketIO.
│   ├── config.py          # Quản lý cấu hình toàn cục.
│   ├── utils.py           # Các hàm tiện ích (utilities) dùng chung.
│   ├── extensions.py      # Chứa các khởi tạo mở rộng (VD: SocketIO).
│   ├── events.py          # Quản lý sự kiện WebSocket (real-time).
│   ├── routes/            # Chứa định tuyến API & View.
│   │   ├── api.py         # Các REST API backend.
│   │   └── views.py       # Render giao diện HTML.
│   ├── services/          # Tầng nghiệp vụ (Business Logic).
│   │   ├── video_service.py    # Chuyên xử lý video inference (YOLO + ArcFace).
│   │   └── registry_service.py # Chuyên nghiệp vụ đăng ký/lưu trữ logo.
│   ├── static/            # Chứa các file tĩnh (CSS, JS).
│   └── templates/         # Chứa các file HTML.
├── training/              # Chứa các script dùng để huấn luyện model.
├── weights/               # Thư mục lưu trữ model weights (YOLO, ArcFace).
├── data/                  # Thư mục chứa dữ liệu video test, database embeddings.
├── dataset/               # Thư mục dataset.
├── requirements.txt       # Danh sách các thư viện Python cần thiết.
└── README.md              # Tài liệu hướng dẫn sử dụng này.
```

---

## Cài đặt Môi trường

Yêu cầu sử dụng **Python 3.8+**. Khuyến nghị sử dụng môi trường ảo (virtualenv hoặc conda).

```bash
# Cài đặt các thư viện cần thiết cho project
pip install -r requirements.txt
```

---

## Model Weights

Hệ thống cần 2 file trọng số (weights) để hoạt động. Vui lòng đặt các file này vào thư mục `weights/`:
1. `best.pt`: Trọng số của mô hình YOLO thực hiện object detection & segmentation.
2. `arcface_logo_model_best_b4_64_06.pth`: Trọng số của mô hình ArcFace thực hiện feature extraction.

---

## Hướng dẫn Sử dụng (Command Line Interface - CLI)

Bạn có thể chạy trực tiếp pipeline để xử lý video thông qua script helper.

Cú pháp cơ bản (đảm bảo bạn đang ở thư mục gốc của project):
```bash
python scripts/run_pipeline.py \
  --video "output/query.mp4" \
  --yolo-weights "weights/best.pt" \
  --recog-weights "weights/arcface_logo_model_best_b4_64_06.pth" \
  --embed-db "output/embedding_db.pkl" \
  --output "output/result.mp4" \
  --conf-threshold 0.7 \
  --recog-threshold 0.4
```

Hoặc sử dụng thư viện trực tiếp (yêu cầu chuyển vào thư mục `src`):
```bash
cd src
python -m oslr --help
```

### Các tham số chính:
- `--video`: Đường dẫn tới file video đầu vào.
- `--yolo-weights`: Đường dẫn tới file weight của YOLO.
- `--recog-weights`: Đường dẫn tới file weight của ArcFace.
- `--embed-db`: Đường dẫn tới database chứa vector của các logo (File Pickle lưu danh sách các tuple `(embedding, label)`).
- `--output`: Đường dẫn để xuất file video kết quả.
- `--conf-threshold`: Ngưỡng tự tin (confidence) của YOLO (mặc định: `0.7`).
- `--recog-threshold`: Ngưỡng độ tương đồng của ArcFace để chấp nhận nhận diện đúng (mặc định: `0.4`).
- `--device`: Chỉ định thiết bị chạy (VD: `cuda:0`, `cpu`). Mặc định tự động nhận diện.

---

## Chạy Web App Demo

Dự án cung cấp một giao diện Web đơn giản (Flask + SocketIO) cho phép người dùng upload video, trực tiếp theo dõi tiến độ xử lý và xem trước kết quả trực quan trên trình duyệt. Toàn bộ code Web App đã được đồng bộ để sử dụng chung mô hình EfficientNet-B4 với CLI.

```bash
cd src/web
python app.py
```
Sau khi khởi động, truy cập trình duyệt web tại địa chỉ: `http://localhost:5000`

---

## Ghi chú về Dữ liệu

- Thư mục `dataset/` mặc định đã được cấu hình ẩn trong `.gitignore` để tránh upload lên mã nguồn (GitHub).
- Khi kiểm thử trên máy cá nhân, hệ thống sẽ tự động tạo thư mục `output/` để lưu video đầu ra và thông tin cơ sở dữ liệu.
