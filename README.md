# 🤖 Nhận Diện Cử Chỉ Tay - Dịch Ngôn Ngữ Kí Hiệu

Phát hiện và nhận diện cử chỉ tay thực thời sử dụng **MediaPipe + Transfer Learning (MobileNetV2)**.

Nhận diện cử chỉ A-Z và chuyển đổi chuỗi cử chỉ thành từ tiếng Anh.

## 📋 Tính Năng

- ✅ **Phát hiện tay thực thời** (MediaPipe Hands - 21 điểm trên mỗi tay)
- ✅ **Transfer Learning** với MobileNetV2 (pre-trained trên ImageNet)
- ✅ **26 lớp cử chỉ** (A-Z)
- ✅ **Điểm độ tin cậy** (0-100%)
- ✅ **Bộ đếm FPS** thực thời
- ✅ **Lưu frame** (nhấn 's')
- ✅ **Nhận diện từ** từ chuỗi cử chỉ
- ✅ **Công cụ thu thập dữ liệu** tùy chỉnh
- ✅ **Pipeline huấn luyện** dễ dàng

## 🎯 Cử Chỉ Được Hỗ Trợ

**26 Lớp Chữ (A-Z)** từ thư mục sign_language_data:

```
a b c d e f g h i j k l m n o p q r s t u v w x y z
```

Cộng thêm: **Nhận diện từ** từ chuỗi cử chỉ sử dụng difflib

## 🚀 KHỞI ĐỘNG NHANH (Windows)

### Cách Dễ Nhất - Tự Động Setup
```bash
# 1. Nhấp đúp: setup.bat  (chờ 2-3 phút)
# 2. Nhấp đúp: run.bat
# Xong! ✓
```

### Setup Thủ Công
```bash
# 1. Tạo virtual environment
python -m venv venv

# 2. Kích hoạt (Windows)
venv\Scripts\activate

# 3. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 4. Chạy ứng dụng
python gesture_recognition.py
```

## ⌨️ Phím Tắt

| Phím | Hành Động |
|------|-----------|
| `q` | Thoát ứng dụng |
| `s` | Lưu frame hiện tại |
| `SPACE` | (word_recognition.py) Thêm cử chỉ vào chuỗi |
| `ENTER` | (word_recognition.py) Dự đoán từ |
| `c` | (word_recognition.py) Xóa chuỗi |

## 📦 Các File Trong Dự Án

```
projectai/
├── gesture_recognition.py    ← Ứng dụng chính (chạy cái này!)
├── collect_data.py           ← Thu thập dữ liệu huấn luyện
├── train.py                  ← Huấn luyện mô hình mới
├── word_recognition.py       ← Chuyển cử chỉ → từ
├── config.py                 ← Cấu hình
├── check_setup.py            ← Xác minh cài đặt
├── setup.bat                 ← Auto setup (Windows)
├── run.bat                   ← Chạy nhanh (Windows)
│
├── sign_language_data/       ← Dữ liệu huấn luyện (a-z)
├── sign_language_model.h5    ← Mô hình đã huấn luyện
├── requirements.txt          ← Gói Python
│
├── README.md                 ← Tài liệu đầy đủ (file này)
├── README.txt                ← Hướng dẫn khởi động nhanh
└── QUICKSTART.md             ← Tham khảo nhanh
```

## 🔧 Tuning & Cải Thiện

### Tăng Độ Chính Xác:
1. **Thu thập dữ liệu**: ~100-200 ảnh mỗi tín hiệu
2. **Fine-tuning**: Huấn luyện lại các layer cuối
3. **Data augmentation**: Xoay, scaling, brightness adjustment
4. **Threshold adjustment**: Thay đổi `confidence > 0.5`

## ⚙️ Các Tham Số Có Thể Điều Chỉnh

Trong `hehe.py`:
```python
SIGN_LABELS = {...}        # Thêm/bớt tín hiệu
NUM_CLASSES = len(...)     # Số lớp
confidence > 0.5           # Ngưỡng độ tin cậy
## 🧠 Kiến Trúc Mô Hình

### Pipeline Transfer Learning
```
MobileNetV2 (pre-trained trên ImageNet)
         ↓
GlobalAveragePooling2D
         ↓
Dense(256, relu) + Dropout(0.3)
         ↓
Dense(128, relu) + Dropout(0.2)
         ↓
Dense(26, softmax) ← Output (A-Z)
```

**Tại sao Transfer Learning?**
- Sử dụng trọng số được huấn luyện trên 1.2 triệu ảnh
- Huấn luyện nhanh với dữ liệu hạn chế (~12,600 ảnh)
- Độ chính xác cao hơn (85-95%)
- Tiết kiệm thời gian huấn luyện (so với training từ đầu)

### MediaPipe Hands
Phát hiện **21 điểm trên mỗi tay:**
- Cổ tay, lòng bàn tay, 5 ngón tay
- Thực thời, hiệu quả, hoạt động trên CPU

## 📊 Hiệu Năng

| Chỉ Số | CPU | GPU (NVIDIA) |
|--------|-----|------|
| FPS | 8-15 | 20-30 |
| Inference | 80-150ms | 20-50ms |
| Bộ Nhớ | 500-800MB | 1-2GB |
| Kích Thước | 10MB | 10MB |

## 🛠️ Ví Dụ Sử Dụng

### 1. Nhận Diện Cử Chỉ Thực Thời
```bash
python gesture_recognition.py
```
Sử dụng webcam để phát hiện cử chỉ tay thực thời.

### 2. Thu Thập Dữ Liệu Huấn Luyện
```bash
python collect_data.py
```
Công cụ tương tác để chụp ảnh cho huấn luyện tùy chỉnh.

### 3. Huấn Luyện Mô Hình Tùy Chỉnh
```bash
python train.py
```
Huấn luyện mô hình mới trên sign_language_data/ với batch loading.

### 4. Nhận Diện Từ
```bash
python word_recognition.py
```
Kết hợp chuỗi cử chỉ (A-Z) thành từ tiếng Anh.

## 🐛 Xử Lý Sự Cố

| Vấn Đề | Giải Pháp |
|--------|-----------|
| Camera không được phát hiện | Kiểm tra kết nối, cấp quyền trong Settings |
| Lỗi "Module not found" | Chạy: `pip install -r requirements.txt` |
| FPS chậm (8-15) | Bình thường trên CPU; dùng GPU để nhanh hơn |
| Mô hình không tìm thấy | Đặt `sign_language_model.h5` trong thư mục dự án |
| Độ chính xác thấp | Thu thập thêm dữ liệu, huấn luyện lại: `python train.py` |

## 📚 Công Nghệ Sử Dụng

- **Deep Learning:** TensorFlow 2.14 + Keras
- **Computer Vision:** OpenCV 4.8, MediaPipe 0.10
- **Mô Hình Cơ Sở:** MobileNetV2 (1.4M tham số)
- **Phát Hiện:** MediaPipe Hands (thực thời)
- **Ngôn Ngữ:** Python 3.9+

## 📋 Yêu Cầu Hệ Thống

- **Hệ Điều Hành:** Windows 10+, macOS, Linux
- **Python:** 3.9 hoặc cao hơn
- **RAM:** Tối thiểu 100MB, khuyến khích 500MB
- **Ổ Cứng:** 2GB (Python + TensorFlow + Model)
- **Webcam:** Có (bắt buộc)
- **GPU:** Tùy chọn (để suy luận nhanh hơn)

## 🚀 Sử Dụng Nâng Cao

### Sử Dụng Mô Hình Tùy Chỉnh
Thay thế `sign_language_model.h5` bằng mô hình của bạn đã huấn luyện.

### Điều Chỉnh Ngưỡng Độ Tin Cậy
Chỉnh sửa trong `gesture_recognition.py`:
```python
if confidence > 0.7:  # Thay 0.7 bằng ngưỡng của bạn
    # Hiển thị kết quả
```

### Thu Thập Cử Chỉ Tùy Chỉnh
```bash
python collect_data.py
# Lưu ảnh vào: sign_language_data/<cử chỉ>/
```

## 📖 Thống Kê Dự Án

- **Tổng Code:** ~1,500 dòng
- **Tổng Tài Liệu:** ~3,000 dòng  
- **Dữ Liệu Huấn Luyện:** 12,637 ảnh (a-z)
- **Kích Thước Mô Hình:** 10MB
- **Thời Gian Setup:** 5-10 phút
- **Thời Gian Huấn Luyện:** 30-60 phút (CPU)

## 🎓 Tài Liệu Học Tập

- [Transfer Learning Cơ Bản](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Bài Báo MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Tài Liệu MediaPipe](https://developers.google.com/mediapipe)
- [Hướng Dẫn OpenCV](https://docs.opencv.org/4.8.0/)

## 📝 Ghi Chú

- Mô hình tải lần đầu mất 30-60 giây (khởi động TensorFlow)
- Hoạt động trên CPU nhưng nhanh hơn với GPU
- Cử chỉ nên rõ ràng và riêng biệt để có độ chính xác tốt
- Hiện hỗ trợ chữ a-z; dễ dàng mở rộng cho từ

## 📧 Hỗ Trợ

Để báo cáo lỗi, câu hỏi hoặc cải thiện - kiểm tra README.txt để có câu trả lời nhanh.

---

**Phiên Bản:** 1.0  
**Cập Nhật Lần Cuối:** 2026-01-31  
**Trạng Thái:** ✅ Sẵn Sàng Sử Dụng  
**Giấy Phép:** MIT
