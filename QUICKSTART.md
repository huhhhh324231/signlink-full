# 🤖 Nhận Diện Cử Chỉ Tay - Dịch Ngôn Ngữ Kí Hiệu

Phát hiện cử chỉ tay thực thời sử dụng MediaPipe + Transfer Learning.

## 🚀 Khởi Động Nhanh (Windows)

### Setup Lần Đầu
1. **Nhấp đúp** `setup.bat` → Chờ 2-3 phút ⏳
2. **Xong!** Virtual environment + tất cả gói cài đặt

### Chạy Ứng Dụng
- **Nhấp đúp** `run.bat`
- Hoặc: `python gesture_recognition.py`

## 📂 Các File
- `gesture_recognition.py` - Phát hiện thực thời (webcam)
- `train.py` - Huấn luyện mô hình mới
- `collect_data.py` - Thu thập dữ liệu huấn luyện
- `word_recognition.py` - Chuyển cử chỉ → từ
- `sign_language_model.h5` - Mô hình được huấn luyện (26 lớp: a-z)
- `sign_language_data/` - Dữ liệu huấn luyện (thư mục a-z)

## ⚙️ Yêu Cầu Hệ Thống
- Python 3.9+
- Webcam
- Windows/Mac/Linux
- ~2GB dung lượng ổ cứng

## 🎯 Tính Năng
- ✅ Phát hiện tay thực thời
- ✅ 26 lớp cử chỉ (a-z)
- ✅ Tính điểm độ tin cậy
- ✅ Dự đoán từ từ chuỗi cử chỉ
- ✅ Công cụ thu thập dữ liệu
- ✅ Huấn luyện tùy chỉnh

## 📊 Thông Tin Mô Hình
- **Kiến Trúc:** MobileNetV2 (transfer learning)
- **Đầu Vào:** 224×224 ảnh RGB
- **Đầu Ra:** 26 lớp (a-z)
- **Kích Thước:** ~10MB

## 🔧 Khắc Phục Sự Cố

**Camera không hoạt động?**
- Kiểm tra quyền → Khởi động lại → Thử lại

**Tệp mô hình bị thiếu?**
- Chạy `python train.py` để huấn luyện mô hình mới

**FPS thấp?**
- Bình thường trên CPU (8-15 FPS)
- Thêm GPU để 20+ FPS

## 📝 Giấy Phép
MIT

---

**Câu hỏi?** Xem README.md để tìm tài liệu đầy đủ.
