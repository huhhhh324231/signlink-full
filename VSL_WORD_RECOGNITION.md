# Nhan dien tu VSL bang video

Tinh nang nay su dung model PyTorch `best_vsl_transformer.pth` da duoc tach thanh:

- `vsl_transformer_state.pth`: trong so model Transformer.
- `vsl_word_labels.json`: danh sach 400 nhan tu tieng Viet.

## Chay che do quay video

```powershell
venv\Scripts\python.exe vsl_word_video_recognition.py --camera
```

Hoac bam:

```powershell
run_vsl_words.bat
```

Phim dieu khien:

- `R`: bat dau/dung quay.
- `Q`: ket thuc, sau do chuong trinh se nhan dien tu trong clip vua quay.

Video vua quay duoc luu vao thu muc `recordings/`.

## Chay che do upload/chon file video

Mo hop chon file:

```powershell
venv\Scripts\python.exe vsl_word_video_recognition.py --upload
```

Hoac bam:

```powershell
run_vsl_upload.bat
```

Truyen duong dan truc tiep:

```powershell
venv\Scripts\python.exe vsl_word_video_recognition.py --video "C:\path\to\video.mp4"
```

Neu video duoc quay bang camera truoc va bi nguoc trai/phai, co the them:

```powershell
venv\Scripts\python.exe vsl_word_video_recognition.py --video "C:\path\to\video.mp4" --mirror-video
```

## Pipeline xu ly

1. Doc frame tu webcam hoac file video.
2. MediaPipe Hands trich xuat toi da 2 ban tay.
3. Moi frame duoc chuyen thanh vector 126 chieu: `2 tay * 21 diem * (x, y, z)`.
4. Chuoi duoc cat/downsample hoac zero-pad ve 500 frame.
5. Transformer PyTorch du doan 1 trong 400 tu tieng Viet.
6. Chuong trinh in ra ket qua tot nhat va top 5 du doan.

## Nhan dien cau don gian

Backend web ho tro che do `sentence`:

1. Doc toan bo video cau.
2. Phat hien cac khoang nghi bang do mat tay hoac chuyen dong tay rat nho trong nhieu frame lien tiep.
3. Cat video thanh cac doan ung voi tung tu.
4. Chay model 472 tu tren tung doan.
5. Ghep cac nhan co do tin cay cao nhat thanh mot cau.

De demo on dinh, hay ra dau tung tu ro rang va nghi khoang 0.5-1 giay giua hai tu. Vi du:

```text
[Toi] ... nghi ... [Di] ... nghi ... [Hoc]
```

Ket qua tra ve dang:

```text
Toi Di Hoc
```

## Luu y

- Can cai `torch` trong `venv`; `requirements.txt` da duoc cap nhat.
- Model nay nhan dien tu theo chuoi hanh dong, nen nen quay tron ven mot tu/hanh dong trong mot clip.
- Nen dung nen sang, tay ro, han che vat the che ban tay.
- Neu web bao chua ket noi API sau khi khoi dong lai may, chay `repair_venv.bat` mot lan de tao lai moi truong Python, sau do chay `run_web_local_vsl.bat`.
