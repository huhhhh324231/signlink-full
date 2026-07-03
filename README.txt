================================================================================
 NHẬN DIỆN CỬ CHỈ TAY - HƯỚNG DẪN KHỞI ĐỘNG NHANH
================================================================================

SETUP LẦN ĐẦU (Windows)
================================================================================

1. GIẢI NÉN ZIP
   - Giải nén projectai.zip vào bất kỳ thư mục nào

2. CHẠY SETUP (Tự Động)
   - Nhấp đúp: setup.bat
   - Chờ 2-3 phút để cài đặt các thư viện phụ thuộc
   - Bạn sẽ thấy thông báo [✓] Setup complete!

3. CHẠY ỨNG DỤNG
   - Nhấp đúp: run.bat
   - Hoặc: python gesture_recognition.py

================================================================================

SETUP THỦ CÔNG (Nếu setup.bat không hoạt động)
================================================================================

1. Mở PowerShell trong thư mục projectai

2. Tạo virtual environment:
   python -m venv venv

3. Kích hoạt virtual environment:
   venv\Scripts\activate

4. Cài đặt các thư viện phụ thuộc:
   pip install -r requirements.txt

5. Chạy ứng dụng:
   python gesture_recognition.py

================================================================================

ỨNG DỤNG LÀM GÌ
================================================================================

✓ Sử dụng webcam để phát hiện cử chỉ tay
✓ Nhận diện A-Z (26 cử chỉ chữ cái)
✓ Hiển thị phát hiện thực thời với điểm độ tin cậy
✓ Có thể lưu dữ liệu huấn luyện
✓ Có thể huấn luyện mô hình tùy chỉnh

PHÍM TẮT:
  q  - Thoát ứng dụng
  s  - Lưu frame hiện tại

================================================================================

KHẮC PHỤC SỰ CỐ
================================================================================

VẤN ĐỀ: setup.bat không hoạt động
GIẢI PHÁP: Sử dụng setup thủ công ở trên (dòng lệnh Python)

VẤN ĐỀ: Camera không được phát hiện
GIẢI PHÁP:
  - Kiểm tra webcam có kết nối không
  - Kiểm tra ứng dụng có quyền truy cập camera (Settings → Privacy)
  - Thử khởi động lại ứng dụng

VẤN ĐỀ: Lỗi "Module not found"
GIẢI PHÁP: Đảm bảo setup hoàn tất, chạy: pip install -r requirements.txt

VẤN ĐỀ: FPS chậm/thấp (8-15)
GIẢI PHÁP: Bình thường trên CPU. Để tốc độ nhanh hơn, sử dụng GPU (NVIDIA CUDA)

================================================================================

MÔ TẢ CÁC FILE
================================================================================

gesture_recognition.py   - Ứng dụng chính (phát hiện thực thời)
collect_data.py          - Thu thập hình ảnh huấn luyện
train.py                 - Huấn luyện mô hình mới
word_recognition.py      - Chuyển cử chỉ → từ
config.py                - Cấu hình
check_setup.py           - Xác minh cài đặt
setup.bat                - Auto setup (chạy một lần)
run.bat                  - Chạy ứng dụng (nhấp đúp)

sign_language_data/      - Dữ liệu huấn luyện (thư mục a-z)
sign_language_model.h5   - Mô hình đã được huấn luyện trước

requirements.txt         - Các gói Python cần thiết
README.md               - Tài liệu đầy đủ

================================================================================

TÍNH NĂNG
================================================================================

✓ Phát hiện cử chỉ tay thực thời (MediaPipe)
✓ 26 lớp cử chỉ (a-z)
✓ Transfer Learning (MobileNetV2 - pre-trained)
✓ Điểm số độ tin cậy
✓ Bộ đếm FPS
✓ Công cụ thu thập dữ liệu
✓ Huấn luyện mô hình tùy chỉnh
✓ Dự đoán từ từ chuỗi cử chỉ

================================================================================

YÊU CẦU HỆ THỐNG
================================================================================

✓ Windows 10+ / Mac / Linux
✓ Python 3.9+
✓ Webcam
✓ 2GB dung lượng ổ cứng
✓ 100MB RAM (tối thiểu)

================================================================================

BƯỚC TIẾP THEO
================================================================================

1. Chạy gesture_recognition.py để xem phát hiện thực thời

2. Để thu thập dữ liệu tùy chỉnh:
   python collect_data.py

3. Để huấn luyện với dữ liệu của bạn:
   python train.py

4. Để nhận diện từ:
   python word_recognition.py

================================================================================

CÓ CÂU HỎI?
================================================================================

Xem README.md để có tài liệu đầy đủ
Kiểm tra requirements.txt để xem tất cả các thư viện phụ thuộc

Hỗ Trợ: Liên hệ với nhà phát triển

================================================================================
Tạo: 2026-01-31
Trạng Thái: Sẵn Sàng Sản Xuất ✓
================================================================================
