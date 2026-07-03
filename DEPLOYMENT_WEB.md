# SignLink Deploy Guide

## Kien truc deploy

Frontend React/Vite chay tren Vercel.

Python API chay tren Render/Railway/VPS rieng vi can Torch, MediaPipe, OpenCV va file model `.pth`.

```text
Vercel frontend
  -> VITE_API_BASE
  -> Python API backend
```

## 1. Deploy Python API

Nen dung Render cho ban demo.

Backend can cac file:

- `vsl_web_api.py`
- `vsl_word_video_recognition.py`
- `vsl_transformer_model.py`
- `vsl_word_labels.json`
- `vsl_transformer_state.pth`
- `requirements-deploy.txt`

Len Render:

1. Tao Web Service moi tu repository.
2. Root directory: thu muc project goc.
3. Build command:

```bash
pip install -r requirements-deploy.txt
```

4. Start command:

```bash
python vsl_web_api.py
```

5. Environment variables:

```env
PYTHON_VERSION=3.10.13
CORS_ALLOW_ORIGIN=*
```

Sau khi deploy, kiem tra:

```text
https://your-signlink-api.onrender.com/api/vsl/health
```

Neu tra ve `ok: true`, backend da san sang.

## 2. Deploy frontend Vercel

Thu muc frontend:

```text
signlink-ai
```

Len Vercel:

1. Import repository.
2. Chon Root Directory la `signlink-ai`.
3. Framework Preset: Vite.
4. Build command:

```bash
npm run build
```

5. Output directory:

```text
dist
```

6. Them Environment Variable:

```env
VITE_API_BASE=https://your-signlink-api.onrender.com
```

Sau khi them env, redeploy frontend.

## 3. Local van chay nhu cu

Chay:

```bat
run_web_local_vsl.bat
```

Neu khong co `VITE_API_BASE`, frontend se tu thu:

- `http://127.0.0.1:8008`
- `http://localhost:8008`

## 4. Luu y

- Khong deploy Python AI API len Vercel Functions.
- Backend co the khoi dong cham lan dau vi phai load Torch/model.
- Neu muon khoa CORS sau khi co domain Vercel, doi:

```env
CORS_ALLOW_ORIGIN=https://your-frontend.vercel.app
```
