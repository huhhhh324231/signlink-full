# -*- coding: utf-8 -*-
"""Local HTTP API for Vietnamese sign-language word recognition.

Run:
  venv\Scripts\python.exe vsl_web_api.py

Endpoint:
  POST http://localhost:8008/api/vsl/predict-video
  multipart/form-data field: video
  GET http://localhost:8008/api/dictionary/search?text=hoc
  GET http://localhost:8008/api/text-to-sign?text=toi%20di%20hoc
"""

from __future__ import annotations

import cgi
import difflib
import html
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / "vsl_transformer_state.pth"
LABELS_PATH = BASE_DIR / "vsl_word_labels.json"


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8008"))
CORS_ALLOW_ORIGIN = os.environ.get("CORS_ALLOW_ORIGIN", "*")
PREDICT_MAX_VIDEO_FRAMES = int(os.environ.get("PREDICT_MAX_VIDEO_FRAMES", "48"))
PREDICT_MAX_FRAME_WIDTH = int(os.environ.get("PREDICT_MAX_FRAME_WIDTH", "480"))
PREDICT_MEDIAPIPE_COMPLEXITY = int(os.environ.get("PREDICT_MEDIAPIPE_COMPLEXITY", "0"))
QIPEDC_BASE_URL = "https://qipedc.moet.gov.vn"
TDNNKH_BASE_URL = "https://www.tudienngonngukyhieu.com"

_labels: list[str] | None = None
_model = None
_device = None
_dictionary_cache: dict[str, Any] = {}
_merged_dictionary_cache: dict[str, list[dict[str, Any]]] = {}
_text_to_sign_cache: dict[str, list[dict[str, Any]]] = {}
_segmented_text_cache: dict[str, list[dict[str, Any]]] = {}
_runtime_lock = threading.Lock()
_predict_lock = threading.Lock()
NETWORK_TIMEOUT_SECONDS = 8


def log_event(message: str) -> None:
    print(f"[SignLink API] {message}", flush=True)


def get_runtime():
    global _labels, _model, _device
    started = time.perf_counter()
    with _runtime_lock:
        if _labels is not None and _model is not None and _device is not None:
            return _model, _labels, _device

        log_event("runtime: importing torch/model helpers")
        import torch

        from vsl_transformer_model import load_vsl_transformer
        from vsl_word_video_recognition import load_labels

        try:
            torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
            torch.set_num_interop_threads(max(1, int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1"))))
        except RuntimeError:
            pass

        if _labels is None:
            log_event("runtime: loading labels")
            _labels = load_labels()
        if _device is None:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if _model is None:
            log_event(f"runtime: loading transformer model on {_device}")
            _model = load_vsl_transformer(STATE_PATH, num_classes=len(_labels), device=_device)
        log_event(f"runtime: ready in {time.perf_counter() - started:.2f}s")
        return _model, _labels, _device


def json_bytes(payload: dict[str, Any], status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(payload, ensure_ascii=False).encode("utf-8")


class VSLRequestHandler(BaseHTTPRequestHandler):
    server_version = "SignLinkVSL/1.0"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", CORS_ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_HEAD(self) -> None:
        self.send_response(200)
        self.end_headers()

    def do_GET(self) -> None:
        parsed_path = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"

        if parsed_path == "/":
            model_ready = STATE_PATH.exists() and LABELS_PATH.exists()
            self.send_json(
                {
                    "ok": True,
                    "service": "SignLink VSL API",
                    "health": "/api/vsl/health",
                    "predictVideo": "/api/vsl/predict-video",
                    "dictionarySearch": "/api/dictionary/search?text=hoc",
                    "textToSign": "/api/text-to-sign?text=toi%20di%20hoc",
                    "modelReady": model_ready,
                }
            )
            return

        if parsed_path == "/api/vsl/health" or parsed_path.endswith("/api/vsl/health") or "/api/vsl/health" in self.path:
            model_ready = STATE_PATH.exists() and LABELS_PATH.exists()
            self.send_json(
                {
                    "ok": model_ready,
                    "apiVersion": "best_model_2026_06_28_kaggle_keypoint_pipeline",
                    "model": STATE_PATH.name,
                    "labels": LABELS_PATH.name,
                    "device": "lazy",
                    "path": parsed_path,
                }
            )
            return

        if parsed_path == "/api/dictionary/search":
            try:
                status, payload = self.handle_dictionary_search()
            except Exception as exc:
                status, payload = json_bytes({"ok": False, "error": str(exc)}, status=500)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if parsed_path == "/api/text-to-sign":
            try:
                status, payload = self.handle_text_to_sign()
            except Exception as exc:
                status, payload = json_bytes({"ok": False, "error": str(exc)}, status=500)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_json({"ok": False, "error": "Not found", "path": parsed_path}, status=404)

    def do_POST(self) -> None:
        if self.path != "/api/vsl/predict-video":
            self.send_json({"ok": False, "error": "Not found"}, status=404)
            return

        if not _predict_lock.acquire(blocking=False):
            self.send_json(
                {
                    "ok": False,
                    "error": "Server dang xu ly mot video khac. Hay thu lai sau vai giay.",
                },
                status=429,
            )
            return

        try:
            status, payload = self.handle_predict_video()
        except Exception as exc:
            status, payload = json_bytes({"ok": False, "error": str(exc)}, status=500)
        finally:
            _predict_lock.release()

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def handle_predict_video(self) -> tuple[int, bytes]:
        request_started = time.perf_counter()
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            return json_bytes({"ok": False, "error": "Expected multipart/form-data"}, status=400)

        log_event(
            "predict: request started "
            f"content_length={self.headers.get('Content-Length', 'unknown')}"
        )
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )

        if "video" not in form:
            return json_bytes({"ok": False, "error": "Missing form field: video"}, status=400)

        item = form["video"]
        filename = Path(getattr(item, "filename", "") or "uploaded.webm").name
        suffix = Path(filename).suffix or ".webm"

        saved_bytes = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = item.file.read(1024 * 1024)
                if not chunk:
                    break
                saved_bytes += len(chunk)
                temp_file.write(chunk)

        try:
            log_event(f"predict: saved upload filename={filename} bytes={saved_bytes}")
            model, labels, device = get_runtime()
            log_event("predict: importing video recognition pipeline")
            from vsl_word_video_recognition import (
                extract_feature_timeline_from_video,
                predict_features,
                predict_sentence_features,
            )

            mode = "sentence"
            if "mode" in form:
                mode_item = form["mode"]
                mode = str(getattr(mode_item, "value", "sentence") or "sentence")

            log_event(
                "predict: extracting timeline "
                f"mode={mode} max_frames={PREDICT_MAX_VIDEO_FRAMES} "
                f"max_width={PREDICT_MAX_FRAME_WIDTH} complexity={PREDICT_MEDIAPIPE_COMPLEXITY}"
            )
            timeline = extract_feature_timeline_from_video(
                temp_path,
                mirror=False,
                show_preview=False,
                max_frames=PREDICT_MAX_VIDEO_FRAMES,
                max_frame_width=PREDICT_MAX_FRAME_WIDTH,
                model_complexity=PREDICT_MEDIAPIPE_COMPLEXITY,
            )
            log_event(f"predict: extracted frames={len(timeline)}")
            if not timeline:
                return json_bytes(
                    {
                        "ok": False,
                        "error": "Khong doc duoc frame nao tu video.",
                        "validFrames": 0,
                    },
                    status=422,
                )
            raw_frame_shape = list(timeline[0].shape)
            pipeline = {
                "rawFrameShape": raw_frame_shape,
                "rawSequenceShape": [len(timeline), *raw_frame_shape],
                "normalizedShape": [64, 228],
                "layout": "34 normalized body landmarks + 21 normalized left hand + 21 normalized right hand, flattened per landmark as x,y,z",
            }

            if mode == "sentence":
                log_event("predict: running sentence prediction")
                sentence, words = predict_sentence_features(model, labels, timeline, device)
                if not words:
                    return json_bytes(
                        {
                            "ok": False,
                            "mode": "sentence",
                            "error": "Khong tach duoc tu nao. Hay nghi ro hon giua cac dau.",
                            "validFrames": len(timeline),
                        },
                        status=422,
                    )
                return json_bytes(
                    {
                        "ok": True,
                        "mode": "sentence",
                        "filename": filename,
                        "validFrames": len(timeline),
                        "pipeline": pipeline,
                        "sentence": sentence,
                        "words": words,
                        "predictions": words[0]["predictions"] if words else [],
                    }
                )

            log_event("predict: running word prediction")
            predictions = predict_features(model, labels, timeline, device)
            status_payload = json_bytes(
                {
                    "ok": True,
                    "mode": "word",
                    "filename": filename,
                    "validFrames": len(timeline),
                    "pipeline": pipeline,
                    "predictions": [
                        {"label": label, "confidence": confidence}
                        for label, confidence in predictions
                    ],
                }
            )
            log_event(f"predict: completed in {time.perf_counter() - request_started:.2f}s")
            return status_payload
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def handle_dictionary_search(self) -> tuple[int, bytes]:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        text = query.get("text", [""])[0].strip()
        limit_text = query.get("limit", ["40"])[0]
        try:
            limit = max(1, min(200, int(limit_text)))
        except ValueError:
            limit = 40

        merged_items = search_merged_dictionary(text, limit)
        topics = sorted(
            {
                item["topic"]
                for item in merged_items
                if item["topic"]
            }
        )
        return json_bytes(
            {
                "ok": True,
                "source": f"{QIPEDC_BASE_URL}/dictionary + {TDNNKH_BASE_URL}/ngon-ngu-ky-hieu-theo-tu",
                "query": text,
                "total": len(merged_items),
                "items": merged_items,
                "topics": topics[:100],
                "sources": [
                    {"name": "QIPEDC", "count": len([item for item in merged_items if item["sourceName"] == "QIPEDC"])},
                    {"name": "TuDienNgonNguKyHieu", "count": len([item for item in merged_items if item["sourceName"] == "TuDienNgonNguKyHieu"])},
                ],
            }
        )

    def handle_text_to_sign(self) -> tuple[int, bytes]:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        text = query.get("text", [""])[0].strip()
        if not text:
            return json_bytes({"ok": False, "error": "Missing text"}, status=400)

        tokens = split_text_words(text)
        words = segment_text_to_sign(tokens)
        matched_count = len([word for word in words if word["matched"]])

        return json_bytes(
            {
                "ok": True,
                "source": f"{QIPEDC_BASE_URL}/dictionary + {TDNNKH_BASE_URL}/ngon-ngu-ky-hieu-theo-tu",
                "text": text,
                "tokens": tokens,
                "matchedCount": matched_count,
                "missing": [word["token"] for word in words if not word["matched"]],
                "words": words,
            }
        )

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        status_code, body = json_bytes(payload, status=status)
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def fetch_qipedc_dictionary(text: str) -> list[dict[str, Any]]:
    cache_key = text.lower()
    if cache_key in _dictionary_cache:
        return _dictionary_cache[cache_key]

    body = urllib.parse.urlencode({"group": "20", "text": text}).encode("utf-8")
    request = urllib.request.Request(
        f"{QIPEDC_BASE_URL}/dictionary/getAll",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "SignLinkLocal/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read().decode("utf-8"))

    data = payload.get("data", [])
    if not isinstance(data, list):
        data = []
    _dictionary_cache[cache_key] = data
    return data


def search_merged_dictionary(text: str, limit: int = 40) -> list[dict[str, Any]]:
    cache_key = f"merged:{normalize_text_key(text)}:{limit}"
    if cache_key in _merged_dictionary_cache:
        return _merged_dictionary_cache[cache_key]

    with ThreadPoolExecutor(max_workers=2) as executor:
        qipedc_future = executor.submit(fetch_qipedc_dictionary, text)
        tdnnkh_future = executor.submit(fetch_tdnnkh_dictionary, text, limit)
        qipedc_data = qipedc_future.result()
        tdnnkh_items = tdnnkh_future.result()

    qipedc_items = [normalize_dictionary_item(item) for item in qipedc_data[:limit]]
    existing_keys = {item["normalizedKey"] for item in qipedc_items if item["normalizedKey"]}
    merged = merge_dictionary_sources(qipedc_items, tdnnkh_items, existing_keys, limit)
    _merged_dictionary_cache[cache_key] = merged
    return merged


def search_text_to_sign_dictionary(text: str, limit: int = 20) -> list[dict[str, Any]]:
    cache_key = f"textsign:{normalize_text_key(text)}:{limit}"
    if cache_key in _text_to_sign_cache:
        return _text_to_sign_cache[cache_key]

    with ThreadPoolExecutor(max_workers=2) as executor:
        qipedc_future = executor.submit(fetch_qipedc_dictionary, text)
        tdnnkh_future = executor.submit(fetch_tdnnkh_dictionary, text, limit)
        qipedc_data = qipedc_future.result()
        tdnnkh_items = tdnnkh_future.result()

    qipedc_items = [normalize_dictionary_item(item) for item in qipedc_data[:limit]]
    existing_keys = {item["normalizedKey"] for item in qipedc_items if item["normalizedKey"]}
    merged = merge_dictionary_sources(qipedc_items, tdnnkh_items, existing_keys, limit * 2)
    token_key = normalize_text_key(text)
    exact_items = [item for item in merged if item.get("normalizedKey") == token_key]
    fuzzy_items = [item for item in merged if item.get("normalizedKey") != token_key]
    result = (exact_items + fuzzy_items)[:limit]
    _text_to_sign_cache[cache_key] = result
    return result


def segment_text_to_sign(
    tokens: list[str],
    max_phrase_words: int = 4,
    fuzzy_threshold: float = 0.72,
) -> list[dict[str, Any]]:
    cache_key = f"segment:{' '.join(normalize_text_key(token) for token in tokens)}:{max_phrase_words}:{fuzzy_threshold}"
    if cache_key in _segmented_text_cache:
        return _segmented_text_cache[cache_key]

    words: list[dict[str, Any]] = []
    token_index = 0
    output_index = 1

    while token_index < len(tokens):
        best = find_best_phrase_match(
            tokens,
            token_index,
            max_phrase_words=max_phrase_words,
            fuzzy_threshold=fuzzy_threshold,
        )
        words.append(
            {
                "index": output_index,
                "token": best["phrase"],
                "matched": best["item"] is not None,
                "matchType": best["matchType"],
                "score": best["score"],
                "item": best["item"],
                "suggestions": best["suggestions"],
            }
        )
        token_index += int(best["size"])
        output_index += 1

    _segmented_text_cache[cache_key] = words
    return words


def find_best_phrase_match(
    tokens: list[str],
    start_index: int,
    max_phrase_words: int = 4,
    fuzzy_threshold: float = 0.72,
) -> dict[str, Any]:
    max_size = min(max_phrase_words, len(tokens) - start_index)
    best_fuzzy: dict[str, Any] | None = None

    for size in range(max_size, 0, -1):
        phrase = " ".join(tokens[start_index : start_index + size])
        matches = search_text_to_sign_dictionary(phrase, limit=20)
        exact_match = choose_dictionary_match(phrase, matches, allow_fallback=False, fuzzy=False)
        if exact_match:
            return {
                "phrase": phrase,
                "size": size,
                "item": exact_match,
                "matchType": "exact",
                "score": 1.0,
                "suggestions": rank_dictionary_matches(phrase, matches)[:5],
            }

        fuzzy_match, score = choose_fuzzy_dictionary_match(phrase, matches)
        if fuzzy_match and score >= fuzzy_threshold:
            candidate = {
                "phrase": phrase,
                "size": size,
                "item": fuzzy_match,
                "matchType": "fuzzy",
                "score": score,
                "suggestions": rank_dictionary_matches(phrase, matches)[:5],
            }
            if best_fuzzy is None:
                best_fuzzy = candidate
            else:
                best_size = int(best_fuzzy["size"])
                best_score = float(best_fuzzy["score"])
                if size > best_size or (size == best_size and score > best_score):
                    best_fuzzy = candidate

    if best_fuzzy:
        return best_fuzzy

    token = tokens[start_index]
    matches = search_text_to_sign_dictionary(token, limit=20)
    return {
        "phrase": token,
        "size": 1,
        "item": None,
        "matchType": "missing",
        "score": 0.0,
        "suggestions": rank_dictionary_matches(token, matches)[:5],
    }


def choose_fuzzy_dictionary_match(token: str, matches: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    ranked = rank_dictionary_matches(token, matches)
    if not ranked:
        return None, 0.0
    best = ranked[0]
    return best, float(best.get("matchScore", 0.0))


def rank_dictionary_matches(token: str, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    token_key = normalize_text_key(token)
    ranked: list[dict[str, Any]] = []
    for item in matches:
        score = dictionary_similarity(token_key, item)
        enriched = dict(item)
        enriched["matchScore"] = score
        ranked.append(enriched)
    ranked.sort(
        key=lambda item: (
            float(item.get("matchScore", 0.0)),
            1 if item.get("sourceName") == "QIPEDC" else 0,
            len(str(item.get("normalizedKey", ""))),
        ),
        reverse=True,
    )
    return ranked


def dictionary_similarity(token_key: str, item: dict[str, Any]) -> float:
    if not token_key:
        return 0.0

    candidates = [
        str(item.get("normalizedKey", "")),
        str(item.get("normalizedWord", "")),
        str(item.get("_word", "")),
        str(item.get("word", "")),
    ]
    scores = []
    for candidate in candidates:
        candidate_key = normalize_text_key(candidate)
        if not candidate_key:
            continue
        if token_key == candidate_key:
            scores.append(1.0)
        elif token_key in candidate_key or candidate_key in token_key:
            shorter = min(len(token_key), len(candidate_key))
            longer = max(len(token_key), len(candidate_key))
            scores.append(0.82 + (0.18 * shorter / max(1, longer)))
        else:
            scores.append(difflib.SequenceMatcher(None, token_key, candidate_key).ratio())

    return max(scores) if scores else 0.0


def fetch_tdnnkh_dictionary(text: str, limit: int = 40) -> list[dict[str, Any]]:
    if limit <= 0 or not text.strip():
        return []

    cache_key = f"tdnnkh:{text.lower()}:{limit}"
    if cache_key in _dictionary_cache:
        return _dictionary_cache[cache_key]

    params = urllib.parse.urlencode({"keyword": text})
    request = urllib.request.Request(
        f"{TDNNKH_BASE_URL}/tu-ngu?{params}",
        headers={"User-Agent": "SignLinkLocal/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=NETWORK_TIMEOUT_SECONDS) as response:
            page = response.read().decode("utf-8", errors="ignore")
    except Exception:
        _dictionary_cache[cache_key] = []
        return []

    items = parse_tdnnkh_items(page, limit)
    _dictionary_cache[cache_key] = items
    return items


def parse_tdnnkh_items(page: str, limit: int) -> list[dict[str, Any]]:
    card_pattern = re.compile(
        r'(?is)<div class="flex flex-col items-center overflow-hidden rounded-lg border md:flex-row">(.*?)</div>\s*</div>\s*</div>'
    )
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for card_match in card_pattern.finditer(page):
        card = card_match.group(1)
        href_match = re.search(r'<a href="([^"]+)"', card)
        image_match = re.search(r'<img src="([^"]+)"', card)
        title_match = re.search(r'(?is)<h3.*?<a[^>]*>\s*(.*?)\s*</a>', card)
        if not href_match or not title_match:
            continue

        source_path = html.unescape(href_match.group(1))
        source_url = source_path if source_path.startswith("http") else f"{TDNNKH_BASE_URL}{source_path}"
        word = clean_html_text(title_match.group(1))
        normalized_key = normalize_text_key(word)
        if not normalized_key:
            continue

        tags = [
            clean_html_text(match.group(1))
            for match in re.finditer(
                r'(?is)<a href="/(?:phan-loai|vung-mien)/[^"]+"[^>]*>.*?</i>\s*(.*?)\s*</a>',
                card,
            )
        ]
        topic = tags[0] if tags else ""
        region = tags[-1] if tags else ""
        thumb_url = html.unescape(image_match.group(1)) if image_match else ""
        vimeo_match = re.search(r"vumbnail\.com/(\d+)\.jpg", thumb_url)
        embed_url = f"https://player.vimeo.com/video/{vimeo_match.group(1)}" if vimeo_match else ""
        item_id = source_path.rstrip("/").split("/")[-1]
        dedupe_key = f"{normalized_key}:{region.lower()}:{item_id}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        items.append(
            {
                "id": f"tdnnkh-{item_id}",
                "word": word,
                "normalizedWord": normalized_key,
                "normalizedKey": normalized_key,
                "description": f"Nguon TuDienNgonNguKyHieu. Vung mien: {region or 'Khong ro'}.",
                "lexicalType": topic,
                "topic": topic,
                "region": region,
                "hasImage": bool(thumb_url),
                "thumbUrl": thumb_url,
                "videoUrl": embed_url,
                "embedUrl": embed_url,
                "imageUrl": "",
                "sourceUrl": source_url,
                "sourceName": "TuDienNgonNguKyHieu",
            }
        )
        if len(items) >= limit:
            break

    return items


def clean_html_text(value: str) -> str:
    text = re.sub(r"(?is)<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def merge_dictionary_sources(
    primary_items: list[dict[str, Any]],
    secondary_items: list[dict[str, Any]],
    primary_keys: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    merged = list(primary_items)
    seen_secondary: set[str] = set()

    for item in secondary_items:
        normalized_key = item.get("normalizedKey", "")
        if not normalized_key or normalized_key in primary_keys:
            continue
        variant_key = f"{normalized_key}:{item.get('region', '')}:{item.get('sourceName', '')}"
        if variant_key in seen_secondary:
            continue
        seen_secondary.add(variant_key)
        merged.append(item)
        if len(merged) >= limit:
            break

    return merged


def infer_topic(word: str, lexical_type: str) -> str:
    if "(" in word and ")" in word:
        start = word.rfind("(")
        end = word.rfind(")")
        if start < end:
            return word[start + 1 : end].strip()
    return lexical_type.strip()


def normalize_text_key(value: str) -> str:
    text = unicodedata.normalize("NFD", value.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_text_words(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\sÀ-ỹà-ỹĐđ]", " ", text, flags=re.UNICODE)
    return [word.strip() for word in cleaned.split() if word.strip()]


def choose_dictionary_match(
    token: str,
    matches: list[dict[str, Any]],
    allow_fallback: bool = True,
    fuzzy: bool = True,
) -> dict[str, Any] | None:
    token_key = normalize_text_key(token)
    if not token_key:
        return None

    normalized_matches = []
    for item in matches:
        word = str(item.get("word", ""))
        normalized_word = str(item.get("_word", "") or item.get("normalizedWord", ""))
        normalized_key = str(item.get("normalizedKey", ""))
        word_key = normalize_text_key(word)
        alias_key = normalize_text_key(normalized_word)
        item_key = normalize_text_key(normalized_key)
        normalized_matches.append((item, word_key, alias_key))
        if token_key == word_key or token_key == alias_key or token_key == item_key:
            return item

    if fuzzy:
        for item, word_key, alias_key in normalized_matches:
            if word_key.startswith(token_key) or alias_key.startswith(token_key):
                return item

    return matches[0] if allow_fallback and matches else None


def normalize_text_key(value: str) -> str:
    text = unicodedata.normalize("NFD", value.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "d").replace("Ä‘", "d").replace("Ä", "d")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_text_words(text: str) -> list[str]:
    return re.findall(r"[\wĐđÀ-ỹà-ỹ]+", text, flags=re.UNICODE)


def normalize_dictionary_item(item: dict[str, Any]) -> dict[str, Any]:
    item_id = str(item.get("_id", ""))
    word = str(item.get("word", ""))
    normalized_key = normalize_text_key(word)
    lexical_type = str(item.get("tl", "") or "")
    topic = infer_topic(word, lexical_type)
    return {
        "id": item_id,
        "word": word,
        "normalizedWord": str(item.get("_word", "")),
        "normalizedKey": normalized_key,
        "description": str(item.get("description", "") or ""),
        "lexicalType": lexical_type,
        "topic": topic,
        "region": "",
        "hasImage": bool(item.get("i", False)),
        "thumbUrl": f"{QIPEDC_BASE_URL}/thumbs/{item_id}.png" if item_id else "",
        "videoUrl": f"{QIPEDC_BASE_URL}/videos/{item_id}.mp4" if item_id else "",
        "embedUrl": "",
        "imageUrl": f"{QIPEDC_BASE_URL}/Anh/{item_id[:5]}.png" if item_id and item.get("i", False) else "",
        "sourceUrl": f"{QIPEDC_BASE_URL}/dictionary",
        "sourceName": "QIPEDC",
    }


def main() -> None:
    print(f"[*] SignLink VSL API: http://{HOST}:{PORT}")
    print("[*] Health check: /api/vsl/health")
    if os.environ.get("PRELOAD_MODEL", "1") == "1":
        try:
            log_event("startup: preloading transformer runtime")
            get_runtime()
            log_event("startup: transformer runtime preloaded")
        except Exception as exc:
            log_event(f"startup: preload failed: {exc}")
    print("[*] Press Ctrl+C to stop")
    server = ThreadingHTTPServer((HOST, PORT), VSLRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping API server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
