import sys
import subprocess

def check_python_version():
    """Kiểm tra phiên bản Python"""
    print("=" * 60)
    print("KIỂM TRA CÀI ĐẶT - NGÔN NGỮ KÍ HIỆU")
    print("=" * 60)
    
    version = sys.version_info
    print(f"\n[✓] Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or version.minor < 8:
        print("    [!] Cần Python 3.8 trở lên!")
        return False
    
    return True

def check_package(package_name, import_name=None):
    """Kiểm tra xem package đã cài hay chưa"""
    if import_name is None:
        import_name = package_name.split('-')[0].replace('_', '-')
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"[✓] {package_name}: {version}")
        return True
    except ImportError:
        print(f"[✗] {package_name}: NOT INSTALLED")
        return False

def check_all_packages():
    """Kiểm tra tất cả packages"""
    print("\n[*] Kiểm tra packages...")
    
    packages = [
        ("TensorFlow", "tensorflow"),
        ("OpenCV", "cv2"),
        ("MediaPipe", "mediapipe"),
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
    ]
    
    all_ok = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_ok = False
    
    return all_ok

def check_camera():
    """Kiểm tra camera"""
    print("\n[*] Kiểm tra camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                print("[✓] Camera: OK")
                return True
            else:
                print("[!] Camera: Không thể đọc frame")
                return False
        else:
            print("[!] Camera: Không thể mở (thử cv2.VideoCapture(1))")
            return False
    except Exception as e:
        print(f"[!] Camera: Lỗi - {e}")
        return False

def check_gpu():
    """Kiểm tra GPU"""
    print("\n[*] Kiểm tra GPU...")
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[✓] GPU: Phát hiện {len(gpus)} GPU")
            for gpu in gpus:
                print(f"  - {gpu}")
            return True
        else:
            print("[i] GPU: Không phát hiện (sử dụng CPU)")
            return False
    except Exception as e:
        print(f"[!] GPU: Lỗi - {e}")
        return False

def main():
    """Hàm chính"""
    all_ok = True
    
    # Kiểm tra Python
    if not check_python_version():
        all_ok = False
    
    # Kiểm tra packages
    if not check_all_packages():
        all_ok = False
    
    # Kiểm tra camera
    check_camera()
    
    # Kiểm tra GPU
    check_gpu()
    
    # Kết quả
    print("\n" + "=" * 60)
    
    if all_ok:
        print("[✓] ĐỦ ĐIỀU KIỆN - SẴN SÀNG CHẠY!")
        print("\nGõ lệnh: python hehe.py")
    else:
        print("[!] THIẾU DEPENDENCIES")
        print("\nGõ lệnh: pip install -r requirements.txt")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
