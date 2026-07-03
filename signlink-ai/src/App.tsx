import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  BookOpen,
  Camera,
  CheckCircle2,
  FileVideo,
  HeartHandshake,
  Play,
  RefreshCw,
  Search,
  Sparkles,
  Square,
  Type,
  Upload,
  Video,
  XCircle,
} from 'lucide-react';
import { motion } from 'motion/react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

type Prediction = {
  label: string;
  confidence: number;
};

type ApiResponse = {
  ok: boolean;
  mode?: 'word' | 'sentence';
  filename?: string;
  validFrames?: number;
  sentence?: string;
  words?: Array<{
    index: number;
    label: string;
    confidence: number;
    frames: number;
    startFrame?: number;
    endFrame?: number;
    predictions?: Prediction[];
  }>;
  predictions?: Prediction[];
  error?: string;
};

type DictionaryItem = {
  id: string;
  word: string;
  normalizedWord: string;
  normalizedKey?: string;
  description: string;
  lexicalType: string;
  topic: string;
  region?: string;
  hasImage: boolean;
  thumbUrl: string;
  videoUrl: string;
  embedUrl?: string;
  imageUrl: string;
  sourceUrl: string;
  sourceName?: string;
  matchScore?: number;
};

type DictionaryResponse = {
  ok: boolean;
  source?: string;
  query?: string;
  total?: number;
  items?: DictionaryItem[];
  topics?: string[];
  error?: string;
};

type TextToSignWord = {
  index: number;
  token: string;
  matched: boolean;
  matchType?: 'exact' | 'fuzzy' | 'missing';
  score?: number;
  item: DictionaryItem | null;
  suggestions: DictionaryItem[];
};

type TextToSignResponse = {
  ok: boolean;
  source?: string;
  text?: string;
  tokens?: string[];
  matchedCount?: number;
  missing?: string[];
  words?: TextToSignWord[];
  error?: string;
};

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
] as const;

const ENV_API_BASE = import.meta.env.VITE_API_BASE?.trim();
const API_BASES = [
  ...(ENV_API_BASE ? [ENV_API_BASE.replace(/\/$/, '')] : []),
  'https://signlink-full.onrender.com',
  'http://127.0.0.1:8008',
  'http://localhost:8008',
];
const TEXT_TO_SIGN_SUGGESTIONS = ['bạn đi học', 'tôi yêu gia đình', 'xin chào', 'cảm ơn bạn'];
const DICTIONARY_QUICK_SEARCHES = ['học', 'gia đình', 'bạn', 'cảm ơn', 'xin chào'];

export default function App() {
  const [isBackendReady, setIsBackendReady] = useState(false);
  const [backendText, setBackendText] = useState('Dang kiem tra API...');
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [status, setStatus] = useState('Sẵn sàng nhận diện từ bằng video.');
  const [recordedUrl, setRecordedUrl] = useState<string | null>(null);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [apiBase, setApiBase] = useState(API_BASES[0]);
  const [recognitionMode, setRecognitionMode] = useState<'word' | 'sentence'>('sentence');
  const [dictionaryQuery, setDictionaryQuery] = useState('');
  const [dictionaryTopic, setDictionaryTopic] = useState('');
  const [dictionaryItems, setDictionaryItems] = useState<DictionaryItem[]>([]);
  const [dictionaryTopics, setDictionaryTopics] = useState<string[]>([]);
  const [dictionaryStatus, setDictionaryStatus] = useState('Nhập từ khóa để tra cứu từ điển.');
  const [selectedDictionaryItem, setSelectedDictionaryItem] = useState<DictionaryItem | null>(null);
  const [isDictionaryLoading, setIsDictionaryLoading] = useState(false);
  const [textToSignInput, setTextToSignInput] = useState('bạn đi học');
  const [textToSignWords, setTextToSignWords] = useState<TextToSignWord[]>([]);
  const [textToSignStatus, setTextToSignStatus] = useState('Nhập câu tiếng Việt để phát chuỗi video ký hiệu.');
  const [isTextToSignLoading, setIsTextToSignLoading] = useState(false);
  const [currentSignIndex, setCurrentSignIndex] = useState(0);
  const [handTrackText, setHandTrackText] = useState('Handtrack chua bat.');
  const [headerVisibility, setHeaderVisibility] = useState(1);

  const videoRef = useRef<HTMLVideoElement>(null);
  const handCanvasRef = useRef<HTMLCanvasElement>(null);
  const textSignVideoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const handTrackFrameRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const lastScrollYRef = useRef(0);
  const dictionaryResultCacheRef = useRef(new Map<string, DictionaryResponse>());
  const textToSignResultCacheRef = useRef(new Map<string, TextToSignResponse>());

  const bestPrediction = result?.predictions?.[0] ?? null;
  const sentenceText = result?.sentence || result?.words?.map((word) => word.label).join(' ') || '';
  const recordingLabel = isRecording ? 'Đang quay.' : isCameraActive ? 'Camera sẵn sàng.' : 'Camera tắt.';

  const confidenceText = useMemo(() => {
    if (result?.mode === 'sentence') return `${result.words?.length ?? 0} tu`;
    if (!bestPrediction) return '--';
    return `${Math.round(bestPrediction.confidence * 100)}%`;
  }, [bestPrediction, result]);

  useEffect(() => {
    void checkBackend();
    const timer = window.setInterval(() => {
      void checkBackend(false);
    }, 10000);
    return () => {
      window.clearInterval(timer);
      stopCamera();
      handLandmarkerRef.current?.close();
      if (recordedUrl) URL.revokeObjectURL(recordedUrl);
    };
  }, []);

  useEffect(() => {
    lastScrollYRef.current = window.scrollY;

    function handleScroll() {
      const currentScrollY = window.scrollY;
      const scrollDelta = currentScrollY - lastScrollYRef.current;

      if (currentScrollY < 24) {
        setHeaderVisibility(1);
      } else if (Math.abs(scrollDelta) > 1) {
        setHeaderVisibility((visibility) => {
          const nextVisibility = visibility - scrollDelta / 180;
          return Math.min(1, Math.max(0, nextVisibility));
        });
      }

      lastScrollYRef.current = currentScrollY;
    }

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  async function checkBackend(showFailure = true) {
    for (const base of API_BASES) {
      try {
        const response = await fetch(`${base}/api/vsl/health`, { cache: 'no-store' });
        const data = await response.json();
        setApiBase(base);
        setIsBackendReady(Boolean(data.ok));
        setBackendText(data.ok ? `API san sang (${data.device}).` : 'API chua san sang model.');
        return;
      } catch {
        // Try the next localhost variant.
      }
    }

    setIsBackendReady(false);
    if (showFailure) {
      setBackendText('Chua ket noi duoc API.');
    }
  }

  async function startCamera() {
    setResult(null);
    setStatus('Đang xin quyền camera...');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        videoRef.current.playsInline = true;
        await videoRef.current.play();
      }
      setIsCameraActive(true);
      void startHandTracking();
      setStatus('Camera đã bật. Bấm “Bắt đầu quay” để ghi một từ hoặc một câu ký hiệu.');
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Không mở được camera.');
    }
  }

  function stopCamera() {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    stopHandTracking();
    setIsCameraActive(false);
    setIsRecording(false);
  }

  async function getHandLandmarker() {
    if (handLandmarkerRef.current) return handLandmarkerRef.current;

    setHandTrackText('Dang tai handtrack...');
    const vision = await FilesetResolver.forVisionTasks('/wasm');
    const landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: '/hand_landmarker.task',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.45,
      minHandPresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });
    handLandmarkerRef.current = landmarker;
    return landmarker;
  }

  async function startHandTracking() {
    try {
      const landmarker = await getHandLandmarker();
      lastVideoTimeRef.current = -1;
      const track = () => {
        const video = videoRef.current;
        const canvas = handCanvasRef.current;
        if (!video || !canvas || !streamRef.current || video.readyState < 2) {
          handTrackFrameRef.current = window.requestAnimationFrame(track);
          return;
        }

        syncHandCanvasSize(video, canvas);
        if (video.currentTime !== lastVideoTimeRef.current) {
          lastVideoTimeRef.current = video.currentTime;
          const result = landmarker.detectForVideo(video, performance.now());
          drawHandTrack(canvas, result.landmarks ?? []);
          setHandTrackText(result.landmarks?.length ? `Đang bắt ${result.landmarks.length} bàn tay.` : 'Chưa thấy bàn tay.');
        }

        handTrackFrameRef.current = window.requestAnimationFrame(track);
      };

      if (handTrackFrameRef.current !== null) {
        window.cancelAnimationFrame(handTrackFrameRef.current);
      }
      handTrackFrameRef.current = window.requestAnimationFrame(track);
    } catch (error) {
      setHandTrackText(error instanceof Error ? error.message : 'Không bật được handtrack.');
    }
  }

  function stopHandTracking() {
    if (handTrackFrameRef.current !== null) {
      window.cancelAnimationFrame(handTrackFrameRef.current);
      handTrackFrameRef.current = null;
    }
    lastVideoTimeRef.current = -1;
    const canvas = handCanvasRef.current;
    if (canvas) {
      const context = canvas.getContext('2d');
      context?.clearRect(0, 0, canvas.width, canvas.height);
    }
    setHandTrackText('Handtrack da tat.');
  }

  function startRecording() {
    const stream = streamRef.current;
    if (!stream) {
      setStatus('Hãy bật camera trước khi quay.');
      return;
    }

    setResult(null);
    chunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp8')
      ? 'video/webm;codecs=vp8'
      : 'video/webm';
    const recorder = new MediaRecorder(stream, { mimeType });
    recorderRef.current = recorder;

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeType });
      if (recordedUrl) URL.revokeObjectURL(recordedUrl);
      setRecordedUrl(URL.createObjectURL(blob));
      setIsRecording(false);
      void analyzeBlob(blob, 'recorded-word.webm');
    };

    recorder.start();
    setIsRecording(true);
    setStatus('Đang quay. Hãy thực hiện trọn vẹn ký hiệu, rồi bấm “Dừng quay”.');
  }

  function stopRecording() {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state === 'inactive') return;
    setStatus('Đang dừng quay và gửi video sang model...');
    recorder.stop();
  }

  async function analyzeBlob(blob: Blob, filename: string) {
    if (isAnalyzing) {
      setStatus('Đang xử lý video trước đó, vui lòng đợi hoàn tất.');
      return;
    }

    setIsAnalyzing(true);
    setStatus('Đang trích xuất landmarks và nhận diện...');
    try {
      const formData = new FormData();
      formData.append('video', blob, filename);
      formData.append('mode', recognitionMode);
      const response = await fetch(`${apiBase}/api/vsl/predict-video`, {
        method: 'POST',
        body: formData,
      });
      const data = (await response.json()) as ApiResponse;
      setResult(data);
      setStatus(data.ok ? `Đã phân tích ${data.validFrames ?? 0} frame hợp lệ.` : data.error ?? 'Nhận diện thất bại.');
    } catch (error) {
      setResult({ ok: false, error: 'Không gọi được Python API.' });
      setStatus(error instanceof Error ? error.message : 'Không gọi được Python API.');
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl);
      setRecordedUrl(null);
    }
    setResult(null);
    await analyzeBlob(file, file.name);
    event.target.value = '';
  }

  async function searchDictionary(query = dictionaryQuery) {
    const text = query.trim();
    setIsDictionaryLoading(true);
    setDictionaryTopic('');
    setDictionaryItems([]);
    setDictionaryTopics([]);
    setSelectedDictionaryItem(null);
    setDictionaryStatus(text ? 'Đang tra cứu từ điển...' : 'Hãy nhập từ khóa để tra cứu.');
    if (!text) {
      setIsDictionaryLoading(false);
      return;
    }
    const cacheKey = text.toLocaleLowerCase('vi');
    const cached = dictionaryResultCacheRef.current.get(cacheKey);
    if (cached) {
      setDictionaryItems(cached.items ?? []);
      setDictionaryTopics(cached.topics ?? []);
      setSelectedDictionaryItem(cached.items?.[0] ?? null);
      setDictionaryStatus(`Tìm thấy ${cached.total ?? 0} kết quả từ bộ nhớ đệm.`);
      setIsDictionaryLoading(false);
      return;
    }
    try {
      const params = new URLSearchParams({
        text,
        limit: '80',
      });
      const response = await fetch(`${apiBase}/api/dictionary/search?${params.toString()}`);
      const data = (await response.json()) as DictionaryResponse;
      if (!data.ok) {
        throw new Error(data.error || 'Tra cuu that bai.');
      }
      dictionaryResultCacheRef.current.set(cacheKey, data);
      setDictionaryItems(data.items ?? []);
      setDictionaryTopics(data.topics ?? []);
      setSelectedDictionaryItem(data.items?.[0] ?? null);
      setDictionaryStatus(`Tìm thấy ${data.total ?? 0} kết quả. Các từ trùng với QIPEDC đã được bỏ qua.`);
    } catch (error) {
      setDictionaryItems([]);
      setSelectedDictionaryItem(null);
      setDictionaryStatus(error instanceof Error ? error.message : 'Không tra cứu được từ điển.');
    } finally {
      setIsDictionaryLoading(false);
    }
  }

  const visibleDictionaryItems = dictionaryTopic
    ? dictionaryItems.filter((item) => item.topic === dictionaryTopic)
    : dictionaryItems;
  const playableTextToSignWords = textToSignWords.filter((word) => word.matched && (word.item?.videoUrl || word.item?.embedUrl));
  const currentSignWord = playableTextToSignWords[currentSignIndex] ?? null;

  async function buildTextToSign() {
    const text = textToSignInput.trim();
    if (!text) {
      setTextToSignWords([]);
      setCurrentSignIndex(0);
      setTextToSignStatus('Hãy nhập một câu ngắn để tạo video ký hiệu.');
      return;
    }

    setIsTextToSignLoading(true);
    setTextToSignWords([]);
    setCurrentSignIndex(0);
    setTextToSignStatus('Đang tách từ và tìm video từ các nguồn từ điển...');
    const cacheKey = text.toLocaleLowerCase('vi');
    const cached = textToSignResultCacheRef.current.get(cacheKey);
    if (cached) {
      const words = cached.words ?? [];
      setTextToSignWords(words);
      setCurrentSignIndex(0);
      const missing = cached.missing?.length ? ` Thiếu: ${cached.missing.join(', ')}.` : '';
      setTextToSignStatus(`Ghép được ${cached.matchedCount ?? 0}/${words.length} từ từ bộ nhớ đệm.${missing}`);
      setIsTextToSignLoading(false);
      return;
    }
    try {
      const params = new URLSearchParams({ text });
      const response = await fetch(`${apiBase}/api/text-to-sign?${params.toString()}`);
      const data = (await response.json()) as TextToSignResponse;
      if (!data.ok) {
        throw new Error(data.error || 'Không tạo được chuỗi ký hiệu.');
      }

      textToSignResultCacheRef.current.set(cacheKey, data);
      const words = data.words ?? [];
      setTextToSignWords(words);
      setCurrentSignIndex(0);
      const missing = data.missing?.length ? ` Thiếu: ${data.missing.join(', ')}.` : '';
      setTextToSignStatus(`Ghép được ${data.matchedCount ?? 0}/${words.length} từ thành video ký hiệu.${missing}`);
    } catch (error) {
      setTextToSignWords([]);
      setCurrentSignIndex(0);
      setTextToSignStatus(error instanceof Error ? error.message : 'Không tạo được chuỗi ký hiệu.');
    } finally {
      setIsTextToSignLoading(false);
    }
  }

  function playNextSign() {
    setCurrentSignIndex((index) => {
      if (!playableTextToSignWords.length) return 0;
      return (index + 1) % playableTextToSignWords.length;
    });
  }

  function syncHandCanvasSize(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  function drawHandTrack(canvas: HTMLCanvasElement, hands: Array<Array<{ x: number; y: number; z?: number }>>) {
    const context = canvas.getContext('2d');
    if (!context) return;

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.lineWidth = Math.max(2, canvas.width / 420);
    context.lineCap = 'round';
    context.lineJoin = 'round';

    hands.forEach((landmarks, handIndex) => {
      const color = handIndex === 0 ? '#22d3ee' : '#34d399';
      context.strokeStyle = color;
      context.fillStyle = color;

      HAND_CONNECTIONS.forEach(([start, end]) => {
        const from = landmarks[start];
        const to = landmarks[end];
        if (!from || !to) return;
        context.beginPath();
        context.moveTo(from.x * canvas.width, from.y * canvas.height);
        context.lineTo(to.x * canvas.width, to.y * canvas.height);
        context.stroke();
      });

      landmarks.forEach((point, index) => {
        const radius = index === 0 ? 5 : 3.5;
        context.beginPath();
        context.arc(point.x * canvas.width, point.y * canvas.height, radius, 0, Math.PI * 2);
        context.fill();
      });
    });
  }

  return (
    <div className="min-h-screen bg-white text-slate-900">
      <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.16),transparent_34%),radial-gradient(circle_at_top_right,rgba(124,58,237,0.14),transparent_32%),linear-gradient(180deg,#ffffff_0%,#eff6ff_48%,#ffffff_100%)]" />
      <div className="mx-auto flex min-h-screen max-w-7xl flex-col px-5 py-5">
        <header
          className="sticky top-4 z-40 flex flex-col gap-4 rounded-3xl border border-white/70 bg-white/85 px-5 py-4 shadow-xl shadow-blue-900/5 backdrop-blur-xl transition-transform duration-150 ease-out lg:flex-row lg:items-center lg:justify-between"
          style={{
            opacity: headerVisibility,
            transform: `translateY(${(1 - headerVisibility) * -112}px)`,
            pointerEvents: headerVisibility < 0.2 ? 'none' : 'auto',
          }}
        >
          <div className="flex items-center gap-3">
            <div className="flex h-14 w-14 items-center justify-center overflow-hidden rounded-2xl bg-white shadow-lg shadow-blue-900/10 ring-1 ring-blue-100">
              <img
                src="/brand/signlink-icon.png"
                alt="SignLink Language"
                className="h-12 w-12 object-contain"
              />
            </div>
            <div>
              <h1 className="bg-gradient-to-r from-blue-600 to-cyan-500 bg-clip-text text-2xl font-black text-transparent">SignLink Language</h1>
              <p className="text-sm text-slate-500">Nhận diện 472 từ, ghép câu và phát video ký hiệu từ từ điển.</p>
            </div>
          </div>

          <nav className="flex flex-wrap items-center gap-2 text-sm font-bold text-slate-600">
            <a href="#home" className="rounded-2xl px-3 py-2 transition hover:bg-blue-50 hover:text-blue-700">Trang chủ</a>
            <a href="#nhan-dien" className="rounded-2xl px-3 py-2 transition hover:bg-blue-50 hover:text-blue-700">Nhận diện</a>
            <a href="#text-to-sign" className="rounded-2xl px-3 py-2 transition hover:bg-blue-50 hover:text-blue-700">Text sang ký hiệu</a>
            <a href="#tu-dien" className="rounded-2xl px-3 py-2 transition hover:bg-violet-50 hover:text-violet-700">Từ điển</a>
          </nav>

          <button
            type="button"
            onClick={() => void checkBackend()}
            className="inline-flex items-center justify-center gap-2 rounded-2xl border border-blue-100 bg-white px-4 py-2.5 text-sm font-bold text-slate-700 shadow-sm transition hover:border-blue-200 hover:bg-blue-50"
          >
            {isBackendReady ? <CheckCircle2 className="h-4 w-4 text-emerald-600" /> : <XCircle className="h-4 w-4 text-red-600" />}
            {backendText}
          </button>
        </header>

        <section id="home" className="relative overflow-hidden py-14 lg:py-20">
          <div className="absolute right-6 top-10 h-72 w-72 rounded-full bg-violet-200/40 blur-3xl" />
          <div className="absolute bottom-0 left-0 h-64 w-64 rounded-full bg-blue-200/50 blur-3xl" />
          <div className="relative grid items-center gap-10 lg:grid-cols-[1.05fr_0.95fr]">
            <div>
              <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-4 py-2 text-sm font-bold text-blue-700">
                <HeartHandshake className="h-4 w-4" />
                Vì cộng đồng người điếc và khiếm thính Việt Nam.
              </div>
              <h2 className="max-w-3xl text-4xl font-black leading-tight tracking-tight text-slate-950 md:text-5xl lg:text-6xl">
                Cầu nối AI cho{' '}
                <span className="bg-gradient-to-r from-blue-600 via-violet-600 to-indigo-600 bg-clip-text text-transparent">
                  ngôn ngữ ký hiệu Việt Nam
                </span>
              </h2>
              <p className="mt-6 max-w-2xl text-lg leading-relaxed text-slate-600">
                Một nền tảng web để nhận diện từ/câu ký hiệu, tra cứu video minh họa và chuyển văn bản thành chuỗi video ký hiệu từ các nguồn từ điển.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <a href="#nhan-dien" className="inline-flex items-center gap-2 rounded-2xl bg-gradient-to-r from-blue-500 to-violet-500 px-6 py-3 text-sm font-black text-white shadow-xl shadow-blue-400/25 transition hover:scale-[1.02]">
                  <Camera className="h-4 w-4" />
                  Bắt đầu nhận diện
                </a>
                <a href="#tu-dien" className="inline-flex items-center gap-2 rounded-2xl border border-violet-100 bg-white px-6 py-3 text-sm font-black text-violet-700 shadow-sm transition hover:bg-violet-50">
                  <BookOpen className="h-4 w-4" />
                  Mở từ điển
                </a>
              </div>
            </div>

            <div className="relative">
              <div className="absolute inset-0 rounded-[2.5rem] bg-gradient-to-br from-blue-300 to-violet-400 opacity-30 blur-2xl" />
              <div className="relative rounded-[2rem] border border-white/70 bg-white p-6 shadow-2xl shadow-violet-900/10">
                <div className="mb-5 flex items-center justify-between">
                  <div>
                    <p className="text-xs font-black uppercase tracking-widest text-blue-600">AI Sign System</p>
                    <img
                      src="/brand/signlink-full-logo.png"
                      alt="SignLink Language"
                      className="mt-2 h-20 w-auto max-w-[min(100%,28rem)] object-contain"
                    />
                  </div>
                  <div className="rounded-2xl bg-emerald-50 px-3 py-2 text-xs font-black text-emerald-700">
                    {isBackendReady ? 'API sẵn sàng.' : 'Đang đợi API.'}
                  </div>
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  {[
                    { label: 'Nhận diện', value: '472 từ', icon: Camera },
                    { label: 'Chế độ câu', value: 'Cắt & ghép', icon: Sparkles },
                    { label: 'Từ điển', value: '2 nguồn', icon: BookOpen },
                    { label: 'Text-to-Sign', value: 'Longest phrase', icon: Type },
                  ].map((item) => {
                    const Icon = item.icon;
                    return (
                      <div key={item.label} className="rounded-3xl border border-slate-100 bg-gradient-to-br from-slate-50 to-white p-4">
                        <Icon className="mb-3 h-5 w-5 text-blue-600" />
                        <p className="text-xl font-black text-slate-950">{item.value}</p>
                        <p className="mt-1 text-xs font-bold text-slate-500">{item.label}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </section>


        <main id="nhan-dien" className="grid flex-1 scroll-mt-28 gap-8 py-8 lg:grid-cols-[1.35fr_0.8fr]">
          <section className="flex flex-col gap-6">
            <div className="overflow-hidden rounded-[2rem] border border-blue-100 bg-slate-950 shadow-2xl shadow-blue-900/10">
              <div className="relative aspect-video">
                <video ref={videoRef} autoPlay playsInline muted className="h-full w-full scale-x-[-1] object-cover" />
                <canvas ref={handCanvasRef} className="pointer-events-none absolute inset-0 h-full w-full scale-x-[-1] object-cover" />
                {!isCameraActive && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-br from-slate-900 via-slate-950 to-blue-950 px-6 text-center text-white">
                    <Camera className="h-16 w-16 text-cyan-300 opacity-80" />
                    <p className="mt-4 text-xl font-bold">Bật camera để quay ký hiệu.</p>
                    <p className="mt-2 max-w-lg text-sm text-slate-300">Model word-level cần một chuỗi hành động rõ ràng, vì vậy hãy quay trọn vẹn một từ hoặc một câu ngắn.</p>
                  </div>
                )}
                <div className="absolute left-4 top-4 rounded-2xl bg-black/65 px-3 py-2 text-sm font-bold text-white backdrop-blur">
                  {recordingLabel}
                </div>
                <div className="absolute bottom-4 left-4 rounded-2xl bg-black/65 px-3 py-2 text-sm font-bold text-cyan-100 backdrop-blur">
                  {handTrackText}
                </div>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <ActionButton icon={<Camera className="h-5 w-5" />} label="Bật camera" disabled={isCameraActive} onClick={startCamera} />
              <ActionButton icon={<Play className="h-5 w-5" />} label="Bắt đầu quay" disabled={!isCameraActive || isRecording} onClick={startRecording} />
              <ActionButton icon={<Square className="h-5 w-5" />} label="Dừng quay" disabled={!isRecording} onClick={stopRecording} />
              <ActionButton icon={<XCircle className="h-5 w-5" />} label="Tắt camera" disabled={!isCameraActive} onClick={stopCamera} />
            </div>

            <div className="rounded-3xl border border-blue-100 bg-white p-5 shadow-xl shadow-blue-900/5">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-lg font-black text-slate-900">Chế độ nhận diện.</h2>
                  <p className="text-sm text-slate-600">Chế độ câu sẽ tách video theo khoảng nghỉ, nhận diện từng từ và ghép thành câu.</p>
                </div>
                <div className="inline-flex rounded-2xl border border-blue-100 bg-blue-50 p-1">
                  <button
                    type="button"
                    onClick={() => setRecognitionMode('word')}
                    className={`rounded-xl px-4 py-2 text-sm font-bold transition ${recognitionMode === 'word' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-blue-700'}`}
                  >
                    Từ đơn
                  </button>
                  <button
                    type="button"
                    onClick={() => setRecognitionMode('sentence')}
                    className={`rounded-xl px-4 py-2 text-sm font-bold transition ${recognitionMode === 'sentence' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-500 hover:text-blue-700'}`}
                  >
                    Câu
                  </button>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-blue-100 bg-white p-5 shadow-xl shadow-blue-900/5">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <h2 className="text-lg font-black text-slate-900">Upload video có sẵn.</h2>
                  <p className="text-sm text-slate-600">Chọn file MP4, AVI, MOV, MKV hoặc WEBM để API xử lý.</p>
                </div>
                <div className="flex gap-2">
                  <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={handleUpload} />
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-blue-500 to-cyan-500 px-4 py-2.5 text-sm font-bold text-white shadow-lg shadow-blue-400/20 transition hover:scale-[1.02]"
                  >
                    <Upload className="h-4 w-4" />
                    Chọn video
                  </button>
                </div>
              </div>
            </div>

            {recordedUrl && (
              <div className="rounded-3xl border border-blue-100 bg-white p-5 shadow-xl shadow-blue-900/5">
                <div className="mb-3 flex items-center gap-2 text-sm font-bold text-slate-700">
                  <FileVideo className="h-4 w-4" />
                  Clip vua quay
                </div>
                <video src={recordedUrl} controls className="w-full rounded-2xl border border-slate-200" />
              </div>
            )}
          </section>

            <div id="text-to-sign" className="scroll-mt-28 rounded-[2rem] border border-blue-100 bg-white p-6 shadow-xl shadow-blue-900/5 lg:col-span-2 lg:p-8">
              <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
                <div className="flex-1">
                  <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50 px-3 py-1.5 text-xs font-black text-blue-700">
                    <Type className="h-3.5 w-3.5" />
                    Text to Sign
                  </div>
                  <h2 className="text-2xl font-black text-slate-900">Text sang video ký hiệu.</h2>
                  <p className="text-sm text-slate-600">Nhập một câu ngắn, hệ thống sẽ tách từ và phát lần lượt video từ các nguồn từ điển.</p>
                  <div className="mt-5 flex flex-col gap-3 lg:flex-row">
                    <input
                      value={textToSignInput}
                      onChange={(event) => setTextToSignInput(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') void buildTextToSign();
                      }}
                      className="min-w-0 flex-1 rounded-2xl border border-blue-100 bg-blue-50/50 px-5 py-4 text-base font-semibold outline-none transition focus:border-blue-400 focus:bg-white focus:ring-4 focus:ring-blue-100"
                      placeholder="Vi du: bạn đi học"
                    />
                    <button
                      type="button"
                      onClick={() => void buildTextToSign()}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-blue-500 to-cyan-500 px-6 py-4 text-sm font-black text-white shadow-lg shadow-blue-400/20 transition hover:scale-[1.01]"
                    >
                      {isTextToSignLoading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                      Tạo video
                    </button>
                  </div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    {TEXT_TO_SIGN_SUGGESTIONS.map((sample) => (
                      <button
                        type="button"
                        key={sample}
                        onClick={() => setTextToSignInput(sample)}
                        className="rounded-full border border-blue-100 bg-blue-50 px-4 py-2 text-xs font-black text-blue-700 transition hover:border-blue-200 hover:bg-white"
                      >
                        {sample}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="mt-4 rounded-2xl border border-blue-100 bg-blue-50 px-4 py-3 text-sm font-medium text-slate-700">{textToSignStatus}</div>

              <div className="mt-6 grid gap-6 xl:grid-cols-[1.45fr_0.95fr]">
                <div className="rounded-3xl border border-blue-100 bg-gradient-to-br from-blue-50 via-sky-50 to-cyan-50 p-4 lg:p-5">
                  {currentSignWord?.item ? (
                    <div>
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <div>
                          <p className="text-sm font-semibold text-slate-500">Đang phát từ {currentSignIndex + 1}/{playableTextToSignWords.length}.</p>
                          <h3 className="text-2xl font-bold">{currentSignWord.item.word}</h3>
                        </div>
                        <button
                          type="button"
                          onClick={playNextSign}
                          className="rounded-2xl border border-blue-100 bg-white px-3 py-2 text-sm font-bold text-blue-700 shadow-sm"
                        >
                          Từ tiếp
                        </button>
                      </div>
                      {currentSignWord.item.embedUrl ? (
                        <iframe
                          key={currentSignWord.item.embedUrl}
                          src={currentSignWord.item.embedUrl}
                          title={currentSignWord.item.word}
                          allow="autoplay; fullscreen; picture-in-picture"
                          allowFullScreen
                          className="aspect-video w-full rounded-2xl border border-blue-100 bg-black shadow-lg shadow-blue-900/10"
                        />
                      ) : (
                        <video
                          ref={textSignVideoRef}
                          key={currentSignWord.item.videoUrl}
                          src={currentSignWord.item.videoUrl}
                          poster={currentSignWord.item.thumbUrl}
                          controls
                          autoPlay
                          muted
                          onEnded={playNextSign}
                          className="aspect-video w-full rounded-2xl border border-blue-100 bg-black object-contain shadow-lg shadow-blue-900/10"
                        />
                      )}
                    </div>
                  ) : (
                    <div className="flex min-h-[26rem] items-center justify-center rounded-2xl bg-white/75 px-6 text-center text-base font-medium text-slate-500">
                      Chưa có chuỗi video. Hãy nhập câu và bấm “Tạo video”.
                    </div>
                  )}
                </div>

                <div className="max-h-[640px] space-y-3 overflow-auto rounded-3xl border border-blue-100 bg-blue-50/40 p-3">
                  {textToSignWords.length ? (
                    textToSignWords.map((word) => {
                      const playableIndex = playableTextToSignWords.findIndex((item) => item.index === word.index);
                      return (
                        <button
                          type="button"
                          key={`${word.index}-${word.token}`}
                          onClick={() => {
                            if (playableIndex >= 0) setCurrentSignIndex(playableIndex);
                          }}
                          className={`w-full rounded-2xl border p-4 text-left transition ${playableIndex === currentSignIndex ? 'border-blue-300 bg-white shadow-md shadow-blue-900/5' : 'border-blue-100 bg-white/80 hover:border-blue-200 hover:bg-white'}`}
                        >
                          <span className="flex items-center justify-between gap-3">
                            <span className="font-bold">{word.index}. {word.token}</span>
                            <span className={`text-xs font-bold ${word.matched ? 'text-emerald-700' : 'text-red-700'}`}>
                              {word.matched ? (word.matchType === 'fuzzy' ? 'Gần đúng' : 'Có video') : 'Chưa có'}
                            </span>
                          </span>
                          <span className="mt-1 block text-sm text-slate-600">
                            {word.item?.word || (word.suggestions.length ? `Gợi ý: ${word.suggestions.map((item) => item.word).join(', ')}` : 'Không tìm thấy trong từ điển.')}
                          </span>
                          {word.matchType === 'fuzzy' && (
                            <span className="mt-1 block text-xs font-bold text-amber-700">
                              Độ giống nhau: {Math.round((word.score ?? word.item?.matchScore ?? 0) * 100)}%
                            </span>
                          )}
                          {word.item?.sourceName && (
                            <span className="mt-1 block text-xs font-bold text-slate-500">
                              {word.item.sourceName}{word.item.region ? ` · ${word.item.region}` : ''}
                            </span>
                          )}
                        </button>
                      );
                    })
                  ) : (
                    <p className="rounded-2xl bg-blue-50 px-4 py-4 text-sm text-slate-600">Chưa có danh sách từ để phát.</p>
                  )}
                </div>
              </div>
            </div>

            <div id="tu-dien" className="scroll-mt-28 rounded-[2rem] border border-violet-100 bg-white p-6 shadow-xl shadow-violet-900/5 lg:col-span-2 lg:p-8">
              <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
                <div className="flex-1">
                  <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-violet-200 bg-violet-50 px-3 py-1.5 text-xs font-black text-violet-700">
                    <BookOpen className="h-3.5 w-3.5" />
                    Từ điển
                  </div>
                  <h2 className="text-2xl font-black text-slate-900">Từ điển tra cứu.</h2>
                  <p className="text-sm text-slate-600">Tra theo từ khóa, ưu tiên QIPEDC và bỏ qua các từ trùng từ nguồn mở rộng.</p>
                  <div className="mt-5 flex flex-col gap-3 lg:flex-row">
                    <input
                      value={dictionaryQuery}
                      onChange={(event) => setDictionaryQuery(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') void searchDictionary();
                      }}
                      className="min-w-0 flex-1 rounded-2xl border border-violet-100 bg-violet-50/50 px-5 py-4 text-base font-semibold outline-none transition focus:border-violet-400 focus:bg-white focus:ring-4 focus:ring-violet-100"
                      placeholder="Nhập từ cần tìm, ví dụ: học, toán, gia đình..."
                    />
                    <button
                      type="button"
                      onClick={() => void searchDictionary()}
                      className="inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-violet-500 to-indigo-500 px-6 py-4 text-sm font-black text-white shadow-lg shadow-violet-400/20 transition hover:scale-[1.01]"
                    >
                      {isDictionaryLoading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                      Tìm
                    </button>
                  </div>
                </div>

                <select
                  value={dictionaryTopic}
                  onChange={(event) => setDictionaryTopic(event.target.value)}
                  className="rounded-2xl border border-violet-100 bg-white px-5 py-4 text-sm font-bold text-slate-700 shadow-sm"
                >
                  <option value="">Tất cả chủ đề</option>
                  {dictionaryTopics.map((topic) => (
                    <option key={topic} value={topic}>
                      {topic}
                    </option>
                  ))}
                </select>
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                {DICTIONARY_QUICK_SEARCHES.map((query) => (
                  <button
                    type="button"
                    key={query}
                    onClick={() => {
                      setDictionaryQuery(query);
                      void searchDictionary(query);
                    }}
                    className="rounded-full border border-violet-100 bg-violet-50 px-4 py-2 text-xs font-black text-violet-700 transition hover:border-violet-200 hover:bg-white"
                  >
                    {query}
                  </button>
                ))}
              </div>

              <div className="mt-4 rounded-2xl border border-violet-100 bg-violet-50 px-4 py-3 text-sm font-medium text-slate-700">{dictionaryStatus}</div>

              <div className="mt-6 grid gap-6 xl:grid-cols-[0.85fr_1.35fr]">
                <div className="max-h-[720px] space-y-3 overflow-auto rounded-3xl border border-violet-100 bg-violet-50/40 p-3">
                  {visibleDictionaryItems.length ? (
                    visibleDictionaryItems.map((item) => (
                      <button
                        type="button"
                        key={item.id}
                        onClick={() => setSelectedDictionaryItem(item)}
                        className={`flex w-full gap-4 rounded-2xl border p-4 text-left transition ${selectedDictionaryItem?.id === item.id ? 'border-violet-300 bg-white shadow-md shadow-violet-900/5' : 'border-violet-100 bg-white/80 hover:border-violet-200 hover:bg-white'}`}
                      >
                        {item.thumbUrl ? (
                          <img src={item.thumbUrl} alt={item.word} className="h-20 w-28 rounded-xl object-cover" />
                        ) : (
                          <span className="flex h-20 w-28 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-xs font-bold text-violet-600">
                            Video
                          </span>
                        )}
                        <span className="min-w-0">
                          <span className="block text-lg font-black text-slate-900">{item.word}</span>
                          <span className="mt-1 block text-xs font-semibold text-violet-700">{item.topic || item.lexicalType || 'Khác'}</span>
                          <span className="mt-1 block text-xs font-bold text-slate-500">{item.sourceName || 'QIPEDC'}{item.region ? ` · ${item.region}` : ''}</span>
                          <span className="mt-1 line-clamp-2 block text-xs text-slate-600">{item.description || 'Chưa có giải nghĩa.'}</span>
                        </span>
                      </button>
                    ))
                  ) : (
                    <p className="rounded-2xl bg-violet-50 px-4 py-4 text-sm text-slate-600">Chưa có kết quả từ điển.</p>
                  )}
                </div>

                <div className="rounded-3xl border border-violet-100 bg-gradient-to-br from-violet-50 to-indigo-50 p-4 lg:p-5">
                  {selectedDictionaryItem ? (
                    <div>
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <h3 className="text-4xl font-black text-slate-950">{selectedDictionaryItem.word}</h3>
                          <p className="mt-1 text-sm font-semibold text-violet-700">{selectedDictionaryItem.topic || selectedDictionaryItem.lexicalType || 'Khác'}</p>
                        </div>
                        <a href={selectedDictionaryItem.sourceUrl} target="_blank" rel="noreferrer" className="rounded-2xl border border-violet-100 bg-white px-3 py-2 text-xs font-bold text-violet-700 shadow-sm">
                          {selectedDictionaryItem.sourceName || 'QIPEDC'}
                        </a>
                      </div>
                      <p className="mt-3 rounded-2xl bg-white px-4 py-3 text-sm leading-relaxed text-slate-700">
                        {selectedDictionaryItem.description || 'Mục từ này chưa có phần giải nghĩa.'}
                      </p>
                      {selectedDictionaryItem.embedUrl ? (
                        <iframe
                          key={selectedDictionaryItem.embedUrl}
                          src={selectedDictionaryItem.embedUrl}
                          title={selectedDictionaryItem.word}
                          allow="autoplay; fullscreen; picture-in-picture"
                          allowFullScreen
                          className="mt-4 aspect-video w-full rounded-2xl border border-violet-100 bg-black shadow-lg shadow-violet-900/10"
                        />
                      ) : (
                        <video key={selectedDictionaryItem.videoUrl} controls className="mt-4 aspect-video w-full rounded-2xl border border-violet-100 bg-black object-contain shadow-lg shadow-violet-900/10" poster={selectedDictionaryItem.thumbUrl}>
                          <source src={selectedDictionaryItem.videoUrl} type="video/mp4" />
                        </video>
                      )}
                      {selectedDictionaryItem.imageUrl && (
                        <img src={selectedDictionaryItem.imageUrl} alt={`${selectedDictionaryItem.word} minh hoa`} className="mt-4 max-h-80 rounded-2xl border border-violet-100 bg-white object-contain" />
                      )}
                    </div>
                  ) : (
                    <div className="flex min-h-[30rem] items-center justify-center px-6 text-center text-base font-medium text-slate-500">
                      Chọn một mục từ để xem video ký hiệu.
                    </div>
                  )}
                </div>
              </div>
            </div>

          <aside className="flex flex-col gap-6 lg:col-start-2 lg:row-start-1">
            <div className="rounded-3xl border border-emerald-100 bg-white p-6 shadow-xl shadow-emerald-900/5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-bold uppercase tracking-wider text-emerald-600">{result?.mode === 'sentence' ? 'Câu nhận diện' : 'Kết quả tốt nhất'}</p>
                  <h2 className="mt-2 text-4xl font-black text-slate-900">{result?.mode === 'sentence' ? sentenceText || '--' : bestPrediction?.label ?? '--'}</h2>
                </div>
                <div className="rounded-2xl border border-emerald-100 bg-emerald-50 px-4 py-3 text-right">
                  <p className="text-xs font-bold text-emerald-700">{result?.mode === 'sentence' ? 'Số từ' : 'Độ tin cậy'}</p>
                  <p className="text-2xl font-bold text-slate-900">{confidenceText}</p>
                </div>
              </div>

              <div className="mt-5 rounded-2xl border border-emerald-100 bg-emerald-50 px-4 py-3 text-sm font-medium text-slate-700">
                <div className="flex items-center gap-2">
                  {isAnalyzing ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Activity className="h-4 w-4" />}
                  <span>{status}</span>
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-blue-100 bg-white p-6 shadow-xl shadow-blue-900/5">
              <div className="mb-4 flex items-center gap-2">
                <Video className="h-5 w-5 text-blue-600" />
                <h3 className="text-lg font-black">{result?.mode === 'sentence' ? 'Các từ đã tách' : 'Top dự đoán'}</h3>
              </div>

              <div className="space-y-3">
                {result?.mode === 'sentence' && result.words?.length ? (
                  result.words.map((word) => (
                    <motion.div
                      key={`${word.index}-${word.label}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="rounded-2xl border border-blue-100 bg-blue-50/70 p-3"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-bold">{word.index}. {word.label}</span>
                        <span className="text-sm font-semibold text-slate-600">{Math.round(word.confidence * 100)}% · {word.frames} frame</span>
                      </div>
                      <div className="mt-1 text-xs font-semibold text-slate-500">
                        Frame {word.startFrame ?? '--'} - {word.endFrame ?? '--'}
                      </div>
                      <div className="mt-2 h-2 rounded-full bg-blue-100">
                        <div className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-cyan-400" style={{ width: `${Math.max(2, word.confidence * 100)}%` }} />
                      </div>
                    </motion.div>
                  ))
                ) : result?.predictions?.length ? (
                  result.predictions.map((prediction, index) => (
                    <motion.div
                      key={`${prediction.label}-${index}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="rounded-2xl border border-blue-100 bg-blue-50/70 p-3"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-bold">{index + 1}. {prediction.label}</span>
                        <span className="text-sm font-semibold text-slate-600">{Math.round(prediction.confidence * 100)}%</span>
                      </div>
                      <div className="mt-2 h-2 rounded-full bg-blue-100">
                        <div className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-cyan-400" style={{ width: `${Math.max(2, prediction.confidence * 100)}%` }} />
                      </div>
                    </motion.div>
                  ))
                ) : (
                  <p className="rounded-2xl bg-blue-50 px-4 py-4 text-sm text-slate-600">
                    Chưa có kết quả. Hãy quay một clip hoặc upload video để nhận diện.
                  </p>
                )}
              </div>
            </div>

          </aside>
        </main>
      </div>
    </div>
  );
}

function ActionButton({
  icon,
  label,
  disabled,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="inline-flex items-center justify-center gap-2 rounded-2xl border border-blue-100 bg-white px-4 py-3 text-sm font-bold text-slate-800 shadow-sm shadow-blue-900/5 transition hover:-translate-y-0.5 hover:border-blue-200 hover:bg-blue-50 disabled:cursor-not-allowed disabled:translate-y-0 disabled:opacity-45"
    >
      {icon}
      {label}
    </button>
  );
}
