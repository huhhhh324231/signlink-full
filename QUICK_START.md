# 📱 CAMERA DỊCH NGÔN NGỮ KÍ HIỆU - HƯỚNG DẪN ĐẦYĐỦ

## 🎯 Tổng Quan Dự Án

Ứng dụng **nhận diện và dịch ngôn ngữ kí hiệu thời gian thực** sử dụng:
- ✅ **Transfer Learning** (MobileNetV2)
- ✅ **MediaPipe Pose** (phát hiện tư thế)
- ✅ **TensorFlow/Keras** (Deep Learning)
- ✅ **OpenCV** (xử lý video)

---

## 📦 Cấu Trúc Dự Án

```
projectai/
├── hehe.py                  ⭐ CHƯƠNG TRÌNH CHÍNH
├── requirements.txt         📋 Dependencies
├── config.py               ⚙️  Cấu hình nâng cao
├── collect_data.py         🎥 Thu thập dữ liệu
├── train_model.py          🧠 Huấn luyện model
├── check_setup.py          ✅ Kiểm tra cài đặt
├── README.md               📖 Tài liệu chi tiết
├── INSTALLATION.md         🚀 Hướng dẫn cài đặt
├── QUICK_START.md          ⚡ Bắt đầu nhanh
└── sign_language_data/     📂 Dữ liệu (tự tạo)
    ├── 0/ (Hello)
    ├── 1/ (Thank You)
    └── ...
```

---

## 🚀 QUICK START (5 phút)

### 1️⃣ Cài Đặt Python & Dependencies

```bash
# Mở Terminal/PowerShell

# Vào thư mục dự án
cd c:\Users\Admin\Desktop\projectai

# Tạo Virtual Environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Cài packages
pip install -r requirements.txt
```

### 2️⃣ Kiểm Tra Cài Đặt

```bash
python check_setup.py
```

Nếu tất cả ✅ = **Sẵn sàng!**

### 3️⃣ Chạy Camera

```bash
python hehe.py
```

**Hướng dẫn:**
- Đứng trước camera
- Thực hiện tín hiệu ngôn ngữ kí hiệu
- Xem kết quả hiển thị
- **'q'** - Thoát | **'s'** - Lưu ảnh

---

## 📚 HƯỚNG DẪN CHI TIẾT

### Để Huấn Luyện Model Riêng

#### Bước 1: Thu Thập Dữ Liệu
```bash
python collect_data.py
```
- Thu thập 100-200 ảnh mỗi tín hiệu
- Thay đổi vị trí, góc độ, ánh sáng
- Dữ liệu sẽ lưu vào `sign_language_data/`

#### Bước 2: Huấn Luyện Model
```bash
python train_model.py
```
- Tạo model Transfer Learning
- Huấn luyện 2 giai đoạn
- Lưu model thành `sign_language_model.h5`

#### Bước 3: Sử Dụng Model Tùy Chỉnh
Chỉnh `hehe.py`:
```python
# Thêm vào main() sau khi tạo model
model = tf.keras.models.load_model("sign_language_model.h5")
```

---

## 🎓 KHÁI NIỆM TRANSFER LEARNING

### Tại Sao Sử Dụng Transfer Learning?

```
ImageNet (1.2 triệu ảnh)
        ↓
MobileNetV2 (pre-trained)
        ↓
Custom Model (Sign Language)
```

**Lợi ích:**
- ✅ Học nhanh hơn (20x)
- ✅ Cần ít dữ liệu hơn (100 ảnh vs 10,000)
- ✅ Độ chính xác cao hơn
- ✅ Tiết kiệm tài nguyên

### Kiến Trúc Model

```python
Input Image (224×224×3)
    ↓
MobileNetV2 (Frozen)        # Giữ nguyên đặc trưng
    ↓
Global Average Pooling       # Giảm chiều
    ↓
Dense(256, ReLU) + Dropout   # Tùy chỉnh
    ↓
Dense(128, ReLU) + Dropout
    ↓
Dense(10, Softmax)           # Output (10 tín hiệu)
```

---

## 🔧 CẤU HÌNH & TUNING

Chỉnh sửa `config.py`:

### Tăng Độ Chính Xác
```python
MODEL_CONFIG = {
    "dropout_rate": 0.4,      # ↑ Giảm overfitting
    "batch_size": 16,         # ↓ Batch nhỏ hơn
    "epochs": 40,             # ↑ Huấn luyện lâu hơn
}
```

### Tăng Tốc Độ
```python
MODEL_CONFIG = {
    "base_model": "MobileNetV2",  # ✅ Nhanh nhất
    "batch_size": 64,             # ↑ Batch lớn
}

INFERENCE_CONFIG = {
    "buffer_size": 1,  # ↓ Xử lý nhanh
}
```

### Dùng GPU
```bash
pip install tensorflow[and-cuda]
```

---

## 📊 THÔNG SỐ MODEL

| Thông Số | Giá Trị |
|---------|--------|
| **Base Model** | MobileNetV2 |
| **Parameters** | ~2.25M |
| **Input Size** | 224×224×3 |
| **Output Classes** | 10 |
| **Inference Time** | ~50-100ms (CPU) |
| **Model Size** | ~8-10 MB |

---

## 🎯 TIN HIỆU ĐƯỢC HỖ TRỢ

| STT | Tín Hiệu | Mô Tả |
|-----|---------|------|
| 1 | 👋 **Hello** | Xin chào |
| 2 | 🙏 **Thank You** | Cảm ơn |
| 3 | 👍 **Yes** | Có |
| 4 | 👎 **No** | Không |
| 5 | ❤️ **I Love You** | Tôi yêu bạn |
| 6 | 🤝 **Please** | Làm ơn |
| 7 | ✅ **Good** | Tốt |
| 8 | ❌ **Bad** | Xấu |
| 9 | 🆘 **Help** | Giúp đỡ |
| 10 | 👋 **Goodbye** | Tạm biệt |

---

## 🐛 TROUBLESHOOTING

### ❌ Lỗi: "Module not found"
```bash
pip install -r requirements.txt
```

### ❌ Camera không hoạt động
```python
# Thay đổi camera ID trong hehe.py
cap = cv2.VideoCapture(1)  # Thay 0 thành 1, 2, ...
```

### ❌ Dự đoán không chính xác
- Thu thập dữ liệu tốt hơn (100+ ảnh/tín hiệu)
- Huấn luyện lâu hơn (50+ epochs)
- Tăng confidence threshold

### ❌ Tốc độ chậm
- Sử dụng GPU
- Giảm độ phân giải camera
- Giảm model complexity

---

## 📖 TÀI LIỆU THÊM

- **Transfer Learning**: https://www.tensorflow.org/tutorials/images/transfer_learning
- **MobileNetV2**: https://arxiv.org/abs/1801.04381
- **MediaPipe**: https://google.github.io/mediapipe/
- **OpenCV**: https://docs.opencv.org/

---

## ✅ CHECKLIST

- [ ] Python 3.8+ cài đặt
- [ ] Virtual environment tạo & kích hoạt
- [ ] `pip install -r requirements.txt` chạy
- [ ] `python check_setup.py` ✅ pass
- [ ] Camera test OK
- [ ] `python hehe.py` chạy
- [ ] Tín hiệu được nhận diện
- [ ] (Optional) Dữ liệu riêng thu thập
- [ ] (Optional) Model tùy chỉnh huấn luyện

---

## 🎉 HOÀN TẤT!

Ứng dụng của bạn đã sẵn sàng sử dụng!

### Bước Tiếp Theo:
1. ✅ Chạy camera: `python hehe.py`
2. 🎥 Thu thập dữ liệu: `python collect_data.py`
3. 🧠 Huấn luyện: `python train_model.py`
4. 🚀 Triển khai: Lưu model & tích hợp ứng dụng

---

**Tạo ngày**: 2026-01-27  
**Phiên bản**: 1.0  
**Kiến trúc**: Transfer Learning + MediaPipe + TensorFlow

Chúc bạn thành công! 🎊
