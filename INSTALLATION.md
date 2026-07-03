# 🚀 HƯỚNG DẪN CÀI ĐẶT VÀ SỬ DỤNG

## 📋 Yêu Cầu Hệ Thống

- **OS**: Windows, macOS, hoặc Linux
- **Python**: 3.8 hoặc cao hơn
- **RAM**: Tối thiểu 4GB (8GB khuyến nghị)
- **Camera**: Webcam hoặc camera tích hợp
- **GPU** (optional): NVIDIA GPU để tăng tốc độ

## 1️⃣ CÀI ĐẶT PYTHON

### Windows
- Tải từ: https://www.python.org/downloads/
- Chọn phiên bản **3.10** hoặc **3.11**
- ✅ Tick "Add Python to PATH" khi cài
- Kiểm tra: `python --version`

### macOS
```bash
brew install python@3.11
```

### Linux
```bash
sudo apt-get install python3.11 python3.11-venv
```

## 2️⃣ CÀI ĐẶT DEPENDENCIES

### Bước 1: Mở Terminal/PowerShell

**Windows**: Nhấn `Win + R`, gõ `cmd` hoặc `powershell`

**macOS/Linux**: Mở Terminal

### Bước 2: Di chuyển tới thư mục dự án

```bash
cd c:\Users\Admin\Desktop\projectai
```

### Bước 3: Tạo Virtual Environment (khuyến nghị)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Bước 4: Cài đặt packages

```bash
pip install -r requirements.txt
```

Nếu cài từng gói:
```bash
pip install tensorflow opencv-python mediapipe numpy scipy
```

**Lưu ý**: Lần đầu cài TensorFlow mất 5-10 phút và tải ~500MB

## 3️⃣ CHẠY CAMERA DỊCH TÍN HIỆU

### Chạy trực tiếp (Demo Mode)
```bash
python hehe.py
```

**Hướng dẫn sử dụng:**
- Ứng dụng sẽ khởi động camera
- Đứng trước camera
- Thực hiện tín hiệu ngôn ngữ kí hiệu
- Kết quả sẽ hiển thị lên màn hình
- **'q'** - Thoát
- **'s'** - Lưu frame hiện tại

## 4️⃣ HUẤN LUYỆN MỤC CÓ RIÊNG

### Bước 1: Thu thập dữ liệu

```bash
python collect_data.py
```

**Hướng dẫn:**
1. Ứng dụng sẽ yêu cầu bạn thực hiện từng tín hiệu
2. Nhấn **SPACE** để bắt đầu/ghi ảnh
3. Thu thập 100-200 ảnh mỗi tín hiệu
4. Thay đổi vị trí, góc độ, ánh sáng

**Cấu trúc thư mục sau khi thu thập:**
```
sign_language_data/
├── 0/ (Hello)
│   ├── Hello_0.jpg
│   ├── Hello_1.jpg
│   └── ...
├── 1/ (Thank You)
├── 2/ (Yes)
└── ...
```

### Bước 2: Huấn luyện model

```bash
python train_model.py
```

**Quá trình:**
1. Tải dữ liệu
2. Tạo model Transfer Learning
3. Huấn luyện 2 giai đoạn
4. Lưu model thành `sign_language_model.h5`

**Thời gian**: 10-30 phút (tùy CPU/GPU)

### Bước 3: Sử dụng model đã huấn luyện

Chỉnh sửa trong `hehe.py`:
```python
# Thay vào line 205 khoảng
model.load_weights("sign_language_model.h5")
# hoặc
model = tf.keras.models.load_model("sign_language_model.h5")
```

## 🎯 QUICK START (5 phút)

1. **Cài Python 3.11**
2. **Mở Terminal, chạy:**
```bash
cd c:\Users\Admin\Desktop\projectai
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python hehe.py
```
3. **Đứng trước camera và thực hiện tín hiệu!**

## ⚡ TĂNG TỐCĐỘ

### Dùng GPU (NVIDIA)
```bash
pip install tensorflow[and-cuda]
```

### Dùng ONNX Runtime (nhanh hơn)
```bash
pip install onnxruntime onnx
```

## 🔍 KIỂM TRA CÀI ĐẶT

Chạy script kiểm tra:
```python
python
>>> import tensorflow as tf
>>> import cv2
>>> import mediapipe as mp
>>> print("TensorFlow:", tf.__version__)
>>> print("OpenCV:", cv2.__version__)
>>> print("MediaPipe: OK")
>>> exit()
```

Nếu không lỗi = **Cài đặt OK!**

## 🐛 XỬ LỶ LỖI

| Lỗi | Giải pháp |
|-----|----------|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow` |
| `cv2 not found` | `pip install opencv-python` |
| Camera không hoạt động | Kiểm tra quyền camera, thử `cv2.VideoCapture(1)` |
| Tốc độ chậm | Sử dụng GPU hoặc giảm độ phân giải |
| RAM không đủ | Đóng các ứng dụng khác, giảm batch size |

## 📚 TÀI LIỆU THÊM

- Đọc [README.md](README.md) để hiểu kiến trúc model
- Tham khảo [TensorFlow docs](https://www.tensorflow.org)
- MediaPipe tutorials: https://google.github.io/mediapipe/

## ✅ HOÀN TẤT

Nếu camera chạy được = **Thành công!** 🎉

Vui lòng báo cáo lỗi nếu gặp vấn đề.

---

**Được tạo**: 2026-01-27  
**Phiên bản**: 1.0
