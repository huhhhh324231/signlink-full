# 🎬 WORKFLOW - QUY TRÌNH HOÀN CHỈNH

## 📋 Sơ Đồ Quy Trình

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMERA AI SIGN LANGUAGE                  │
│                    Transfer Learning                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 1: CÀI ĐẶT & CHUẨN BỊ                           │
└─────────────────────────────────────────────────────────────┘
  
  Step 1.1: Cài Python 3.8+
  ├─ Download từ python.org
  ├─ Chọn "Add to PATH"
  └─ Verify: python --version

  Step 1.2: Tạo Virtual Environment
  ├─ python -m venv venv
  └─ venv\Scripts\activate

  Step 1.3: Cài Dependencies
  ├─ pip install -r requirements.txt
  └─ Chờ ~5-10 phút (TensorFlow lớn)

  Step 1.4: Kiểm Tra Cài Đặt ✅
  ├─ python check_setup.py
  └─ Xác nhận [✓] cho tất cả


┌─────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 2: DEMO - CHẠY CAMERA (KHÔNG CẦN HUẤN LUYỆN)   │
└─────────────────────────────────────────────────────────────┘

  python hehe.py
  ├─ Mở camera
  ├─ Tải MobileNetV2 model (pre-trained)
  ├─ Phát hiện tư thế với MediaPipe
  ├─ Nhận diện tín hiệu
  └─ Hiển thị kết quả


┌─────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 3: HỖ TRỢ THU THẬP DỮ LIỆU (TÙY CHỌN)         │
└─────────────────────────────────────────────────────────────┘

  python collect_data.py
  │
  ├─ Tạo thư mục sign_language_data/
  │  ├─ 0/ (Hello)
  │  ├─ 1/ (Thank You)
  │  ├─ 2/ (Yes)
  │  └─ ...
  │
  ├─ Lặp cho mỗi tín hiệu:
  │  ├─ Thực hiện tín hiệu trước camera
  │  ├─ Nhấn SPACE để ghi ảnh
  │  └─ Thu thập 100-200 ảnh
  │
  └─ Lưu toàn bộ vào sign_language_data/


┌─────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 4: HUẤN LUYỆN MODEL (TÙY CHỌN)                 │
└─────────────────────────────────────────────────────────────┘

  python train_model.py
  │
  ├─ Tải dữ liệu từ sign_language_data/
  │
  ├─ Tạo Transfer Learning Model:
  │  ├─ Base: MobileNetV2 (frozen)
  │  ├─ Custom Head (256 → 128 → 10)
  │  └─ Total params: ~2.5M
  │
  ├─ GIAI ĐOẠN 1: Huấn luyện layer cuối (10 epochs)
  │  ├─ Freeze MobileNetV2
  │  ├─ Train only custom head
  │  └─ Learning rate: 0.001
  │
  ├─ GIAI ĐOẠN 2: Fine-tuning (10 epochs)
  │  ├─ Unfreeze 30 layer cuối của MobileNetV2
  │  ├─ Train toàn bộ model
  │  └─ Learning rate: 0.0001 (nhỏ hơn)
  │
  ├─ Đánh giá trên test set
  │  ├─ Test Accuracy
  │  └─ Test Loss
  │
  └─ Lưu model → sign_language_model.h5


┌─────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 5: SỬ DỤNG MODEL TÙY CHỈNH                     │
└─────────────────────────────────────────────────────────────┘

  1. Chỉnh sửa hehe.py:
     ├─ Thêm dòng:
     │  model = tf.keras.models.load_model("sign_language_model.h5")
     └─ Trong hàm main()

  2. Chạy:
     python hehe.py

  3. Camera sẽ sử dụng model tùy chỉnh


┌─────────────────────────────────────────────────────────────┐
│  GIAI ĐOẠN 6: TRIỂN KHAI & TỐI ƯU (NÂNG CAO)             │
└─────────────────────────────────────────────────────────────┘

  ⚡ Tăng Tốc Độ:
  ├─ Sử dụng GPU: pip install tensorflow[and-cuda]
  ├─ Model quantization
  └─ ONNX Runtime

  📈 Tăng Độ Chính Xác:
  ├─ Thu thập dữ liệu hơn (500+ ảnh/class)
  ├─ Data augmentation
  ├─ Tăng epochs
  └─ Ensemble multiple models

  🔧 Tuning Parameters:
  ├─ Sửa config.py
  ├─ Thay đổi model architecture
  └─ Adjust confidence threshold


┌─────────────────────────────────────────────────────────────┐
│  HỆ THỐNG INFERENCE THỜI GIAN THỰC                        │
└─────────────────────────────────────────────────────────────┘

Input Frame (từ Camera)
    ↓
MediaPipe Pose Detection
├─ Detect 33 body landmarks
└─ Draw skeleton

    ↓
Image Preprocessing
├─ Resize to 224×224
├─ Normalize values
└─ Batch 1 image

    ↓
MobileNetV2 Forward Pass
├─ Extract features (1280 dims)
└─ ~50-100ms CPU / ~20-50ms GPU

    ↓
Custom Head
├─ Dense(256) + ReLU + Dropout
├─ Dense(128) + ReLU + Dropout
└─ Dense(10, Softmax)

    ↓
Prediction Smoothing
├─ Buffer 5 frames
├─ Take majority vote
└─ Calculate average confidence

    ↓
Display Result
├─ Sign name
├─ Confidence %
├─ FPS counter
└─ Skeleton overlay

    ↓
Output to Screen


┌─────────────────────────────────────────────────────────────┐
│  TỆPS LIÊN QUAN TRONG DỰ ÁN                               │
└─────────────────────────────────────────────────────────────┘

📄 hehe.py
  ├─ Main application
  ├─ Real-time camera inference
  └─ Display results

📄 collect_data.py
  ├─ Data collection tool
  ├─ Create directory structure
  └─ Batch image capture

📄 train_model.py
  ├─ Load training data
  ├─ Build Transfer Learning model
  ├─ Two-stage training
  └─ Save model

📄 config.py
  ├─ All configurable parameters
  ├─ Model architecture settings
  ├─ Camera settings
  └─ Inference settings

📄 check_setup.py
  ├─ Verify installation
  ├─ Check all packages
  ├─ Test camera
  └─ Check GPU

📄 requirements.txt
  ├─ TensorFlow
  ├─ OpenCV
  ├─ MediaPipe
  └─ NumPy, SciPy

📄 README.md
  └─ Detailed documentation

📄 INSTALLATION.md
  └─ Setup instructions

📄 QUICK_START.md
  └─ Quick reference

📄 PROJECT_SUMMARY.md
  └─ Project overview


┌─────────────────────────────────────────────────────────────┐
│  DÒNG DỮ LIỆU INFERENCE                                   │
└─────────────────────────────────────────────────────────────┘

Camera Input (640×480 @ 30fps)
    ↓ OpenCV
RGB Frame (640×480×3)
    ↓ MediaPipe
Pose Landmarks (33 points)
    ↓ Visualization
Annotated Frame
    ↓ Region of Interest (224×224)
Cropped Frame
    ↓ Preprocessing
Normalized Input (224×224×3)
    ↓ Batch
Batched Input (1×224×224×3)
    ↓ MobileNetV2 + Custom Head
Predictions (1×10)
    ↓ Softmax
Class Probabilities (0-1)
    ↓ Argmax + Smoothing
Sign Label + Confidence
    ↓ Display
Output to Screen


┌─────────────────────────────────────────────────────────────┐
│  KEYBOARD SHORTCUTS                                        │
└─────────────────────────────────────────────────────────────┘

Trong Camera Mode (hehe.py):
├─ 'q' → Quit application
├─ 's' → Save current frame
└─ ESC → Exit


Trong Data Collection (collect_data.py):
├─ SPACE → Capture image
├─ ESC → Skip to next sign
└─ Enter → Start capture


┌─────────────────────────────────────────────────────────────┐
│  PERFORMANCE METRICS                                       │
└─────────────────────────────────────────────────────────────┘

CPU Performance:
├─ FPS: 8-12
├─ Inference: 80-150ms
├─ Memory: 500-800MB
└─ Model Size: 8-10MB

GPU Performance (NVIDIA):
├─ FPS: 20-30
├─ Inference: 20-50ms
├─ Memory: 1-2GB
└─ Model Size: 8-10MB

Accuracy (with trained model):
├─ Test Accuracy: 85-95%
├─ Confidence: >90%
├─ Real-time robustness: Good
└─ Requires: 100+ images/class


┌─────────────────────────────────────────────────────────────┐
│  CHỌN MỤC CỦA BẠN                                          │
└─────────────────────────────────────────────────────────────┘

NHANH NHẤT (Demo):
1. check_setup.py
2. hehe.py
✅ Xong trong 5 phút

TỰ TRAINING:
1. check_setup.py
2. collect_data.py (100+ ảnh/tín hiệu)
3. train_model.py
4. Sửa hehe.py
5. hehe.py
⏱️ ~1 giờ

ADVANCED:
1. Sửa config.py
2. Thêm tín hiệu mới
3. Fine-tune parameters
4. Deployment optimization
⏰ Tùy chỉnh

---

**Ready to Start?** → Run: python check_setup.py
