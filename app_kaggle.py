# ════════════════════════════════════════════════════════════════════════════
# KAGGLE SETUP
# ════════════════════════════════════════════════════════════════════════════
import subprocess, sys, os

_KAGGLE_PKGS = [
    "gradio>=4.0",
    "qwen-vl-utils",
    "rouge-score",
    "bert-score",
    "sentence-transformers",
    "openai",
    "anthropic",
]

# Packages cần BUỘC upgrade (Kaggle có sẵn nhưng phiên bản quá cũ)
_KAGGLE_PKGS_FORCE = [
    "bitsandbytes>=0.46.1",   # 4-bit NF4 quantization cho B2
    "peft>=0.11.0",           # LoRA adapter
    "transformers>=4.45.0",   # Qwen2-VL support
    "accelerate>=0.34.0",     # device_map dispatcher
]

def _install(pkg: str, upgrade: bool = True):
    args = [sys.executable, "-m", "pip", "install", "-q"]
    if upgrade:
        args.append("--upgrade")
    args.append(pkg)
    subprocess.check_call(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _version_ok(pkg_spec: str) -> bool:
    """Check installed version >= required."""
    import importlib.metadata as md
    try:
        name, _, req = pkg_spec.partition(">=")
        if not req:
            return True
        installed = md.version(name.strip())
        from packaging.version import Version
        return Version(installed) >= Version(req.strip())
    except Exception:
        return False

print("[setup] Checking / installing packages…")
# Cài các package thiếu (skip nếu đã có)
for _pkg in _KAGGLE_PKGS:
    _mod = _pkg.split(">=")[0].split("==")[0].replace("-", "_")
    try:
        __import__(_mod)
    except ImportError:
        print(f"  pip install {_pkg}…", end=" ", flush=True)
        _install(_pkg)
        print("done")

# Force upgrade các package có version cũ
for _pkg in _KAGGLE_PKGS_FORCE:
    if _version_ok(_pkg):
        continue
    print(f"  pip install -U {_pkg}…", end=" ", flush=True)
    try:
        _install(_pkg)
        print("done")
    except subprocess.CalledProcessError as e:
        print(f"FAILED ({e})")

print("[setup] Packages OK\n")
print("[setup] NOTE: nếu vừa upgrade transformers/bitsandbytes/peft, "
      "bạn cần RESTART KERNEL để load module mới!\n")

# NLTK data (METEOR cần wordnet)
import nltk
for _corpus in ["wordnet", "punkt", "averaged_perceptron_tagger", "omw-1.4"]:
    try:
        nltk.download(_corpus, quiet=True)
    except Exception:
        pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# ── Standard library ────────────────────────────────────────────────────────
import re, gc, json, math, time, warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tv_models
import gradio as gr
from PIL import Image
from transformers import AutoTokenizer, AutoModel

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — tự động phát hiện Kaggle vs local
# ════════════════════════════════════════════════════════════════════════════

# ─── Tuỳ chỉnh slug dataset Kaggle của bạn ───────────────────────────────
KAGGLE_DATASET_SLUG = "vqa-food-project"   # tên folder trong /kaggle/input/
# ─────────────────────────────────────────────────────────────────────────

IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    KAGGLE_INPUT  = Path("/kaggle/input")
    WORKING_DIR   = Path("/kaggle/working")
    WORKING_DIR.mkdir(exist_ok=True)

    # Tự động tìm data/ và checkpoints/ trong toàn bộ /kaggle/input/
    # Hỗ trợ cả mount path nested kiểu /kaggle/input/datasets/<user>/<slug>/
    def _find_dir(name: str) -> Optional[Path]:
        # 1. Ưu tiên theo slug đã cấu hình
        slug_path = KAGGLE_INPUT / KAGGLE_DATASET_SLUG / name
        if slug_path.exists() and slug_path.is_dir():
            return slug_path
        # 2. Scan 1 cấp (chuẩn Kaggle: /kaggle/input/<slug>/<name>)
        for p in sorted(KAGGLE_INPUT.glob(f"*/{name}")):
            if p.is_dir():
                return p
        # 3. Scan đệ quy — bắt mọi cấu trúc nested
        for p in sorted(KAGGLE_INPUT.rglob(name)):
            if p.is_dir() and ".ipynb_checkpoints" not in str(p):
                return p
        return None

    _data_dir  = _find_dir("data")
    _ckpt_dir  = _find_dir("checkpoints")
    _res_dir   = _find_dir("results")

    # Debug: in cấu trúc /kaggle/input để dễ chẩn đoán
    print("[app] /kaggle/input/ contents:")
    try:
        for top in sorted(KAGGLE_INPUT.iterdir())[:5]:
            print(f"        ├── {top.name}/")
            if top.is_dir():
                for sub in sorted(top.iterdir())[:8]:
                    marker = "/" if sub.is_dir() else ""
                    print(f"        │   ├── {sub.name}{marker}")
    except Exception as e:
        print(f"        (lỗi liệt kê: {e})")

    DATA_DIR    = _data_dir  or WORKING_DIR / "data"
    CKPT_DIR    = _ckpt_dir  or WORKING_DIR / "checkpoints"
    RESULTS_DIR = _res_dir   or WORKING_DIR / "results"
    ROOT        = WORKING_DIR
else:
    # Local / Windows
    try:
        ROOT = Path(__file__).resolve().parent
    except NameError:
        ROOT = Path.cwd()
    DATA_DIR    = ROOT / "data"
    CKPT_DIR    = ROOT / "checkpoints"
    RESULTS_DIR = ROOT / "results"

IMG_DIR    = DATA_DIR / "images"
TRAIN_JSON = DATA_DIR / "annotations" / "train.json"
TEST_JSON  = DATA_DIR / "annotations" / "test.json"

CKPT_A1      = CKPT_DIR / "best_model_A1.pth"
CKPT_A2      = CKPT_DIR / "best_model_A2.pth"
CKPT_B2_LORA = CKPT_DIR / "qwen2vl_lora_b2" / "adapter_best"

MODEL_B1_HF = "Salesforce/blip2-opt-2.7b"
MODEL_B2_HF = "Qwen/Qwen2-VL-2B-Instruct"
MT_VI_EN_HF = "Helsinki-NLP/opus-mt-vi-en"
MT_EN_VI_HF = "Helsinki-NLP/opus-mt-en-vi"

DECODE_MAX, DECODE_MIN, DECODE_BEAMS, DECODE_ALPHA, DECODE_NGR = 10, 3, 5, 0.7, 2

# ── Device setup ─────────────────────────────────────────────────────────────
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS   = torch.cuda.device_count() if torch.cuda.is_available() else 0

print(f"[app] Device : {DEVICE}")
print(f"[app] GPUs   : {N_GPUS}")
if DEVICE.type == "cuda":
    for gi in range(N_GPUS):
        p = torch.cuda.get_device_properties(gi)
        print(f"[app]   GPU{gi}: {p.name}  {p.total_memory/1024**3:.1f} GB")
print(f"[app] Paths  : DATA={DATA_DIR}  CKPT={CKPT_DIR}")
print(f"[app] CKPT_A1   : {'✓' if CKPT_A1.exists()      else '✗ MISSING'}")
print(f"[app] CKPT_A2   : {'✓' if CKPT_A2.exists()      else '✗ MISSING'}")
print(f"[app] CKPT_B2   : {'✓' if CKPT_B2_LORA.exists() else '✗ MISSING'}")

# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def vram_str() -> str:
    if DEVICE.type != "cuda":
        return "CPU"
    used  = torch.cuda.memory_allocated()  / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{used:.1f} / {total:.1f} GB"


def free_vram():
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def tokenize_vi(text: str) -> list:
    return re.sub(r"[^\w\s]", "", text.lower().strip()).split()


def _load_pil(x) -> Image.Image:
    if isinstance(x, np.ndarray):
        return Image.fromarray(x).convert("RGB")
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    try:
        return Image.open(x).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224))


# ════════════════════════════════════════════════════════════════════════════
# VOCAB
# ════════════════════════════════════════════════════════════════════════════
def build_vocab(data, min_freq=2):
    counter = Counter()
    for d in data:
        counter.update(tokenize_vi(d["answer"]))
    vocab = ["<pad>", "<bos>", "<eos>", "<unk>"]
    for w, c in counter.most_common():
        if c >= min_freq:
            vocab.append(w)
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for w, i in w2i.items()}
    return w2i, i2w


if TRAIN_JSON.exists():
    print(f"[app] Loading vocab from {TRAIN_JSON}…")
    with open(TRAIN_JSON, encoding="utf-8") as f:
        _train_data = json.load(f)
    w2i, i2w = build_vocab(_train_data)
    PAD, BOS, EOS, UNK = w2i["<pad>"], w2i["<bos>"], w2i["<eos>"], w2i["<unk>"]
    VOCAB_SIZE = len(w2i)
    print(f"[app] Vocab size: {VOCAB_SIZE}")
else:
    print(f"[app] WARNING: {TRAIN_JSON} not found — A1/A2 unavailable.")
    w2i, i2w = {}, {}
    PAD = BOS = EOS = UNK = 0
    VOCAB_SIZE = 100


# ════════════════════════════════════════════════════════════════════════════
# MODEL A — Architecture (phải khớp với notebook training)
# ════════════════════════════════════════════════════════════════════════════
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.proj = nn.Sequential(
            nn.Conv2d(2048, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.proj(self.backbone(x))
        B, C, H, W = x.shape
        return x.view(B, C, H * W).permute(0, 2, 1)


class BahdanauAttention(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        self.U = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, 1, bias=False)

    def forward(self, features, hidden):
        e = torch.tanh(self.W(features) + self.U(hidden).unsqueeze(1))
        a = torch.softmax(self.v(e).squeeze(-1), dim=1)
        return (features * a.unsqueeze(-1)).sum(1), a


class PositionalEncoding(nn.Module):
    def __init__(self, d=512, maxlen=128, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(0, maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TrfDecoder(nn.Module):
    def __init__(self, vocab_size, d=512, nhead=8, layers=3, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = PositionalEncoding(d, dropout=dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 4,
            dropout=dropout, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=layers)
        self.fc  = nn.Linear(d, vocab_size)

    def forward(self, memory, captions):
        tgt  = self.pos(self.emb(captions))
        sz   = tgt.size(1)
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=tgt.device), 1)
        return self.fc(self.dec(tgt=tgt, memory=memory, tgt_mask=mask))


class MultimodalFusion(nn.Module):
    def __init__(self, d=512, dropout=0.1):
        super().__init__()
        self.fc   = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, img, txt):
        return self.norm(self.drop(self.fc(img + txt.unsqueeze(1))))


class VQAModelA1(nn.Module):
    def __init__(self, vocab_size, d=512, emb_dim=256):
        super().__init__()
        self.img_enc  = ImageEncoder(d)
        self.txt_enc  = AutoModel.from_pretrained("vinai/phobert-base",
                                                   use_safetensors=True)
        for p in self.txt_enc.parameters():
            p.requires_grad = False
        self.txt_proj = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d),
                                      nn.Dropout(0.1))
        self.fusion   = nn.Sequential(nn.Linear(d * 2, d), nn.Tanh())
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.attn     = BahdanauAttention(d)
        self.lstm     = nn.LSTMCell(emb_dim + d, d)
        self.dropout  = nn.Dropout(0.3)
        self.fc       = nn.Linear(d, vocab_size)

    def encode(self, image, input_ids, attn_mask):
        img = self.img_enc(image)
        with torch.no_grad():
            out = self.txt_enc(input_ids=input_ids, attention_mask=attn_mask)
        txt = self.txt_proj(out.last_hidden_state[:, 0])
        return img, txt


class VQAModelA2(nn.Module):
    def __init__(self, vocab_size, d=512):
        super().__init__()
        self.img_enc  = ImageEncoder(d)
        self.txt_enc  = AutoModel.from_pretrained("vinai/phobert-base",
                                                   use_safetensors=True)
        for p in self.txt_enc.parameters():
            p.requires_grad = False
        self.txt_proj = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d),
                                      nn.Dropout(0.1))
        self.fusion   = MultimodalFusion(d)
        self.decoder  = TrfDecoder(vocab_size, d)

    def encode(self, image, input_ids, attn_mask):
        img = self.img_enc(image)
        with torch.no_grad():
            out = self.txt_enc(input_ids=input_ids, attention_mask=attn_mask)
        txt = self.txt_proj(out.last_hidden_state[:, 0])
        return self.fusion(img, txt)


# ════════════════════════════════════════════════════════════════════════════
# BEAM SEARCH
# ════════════════════════════════════════════════════════════════════════════
def _safe_token(t):
    return max(0, min(int(t), VOCAB_SIZE - 1))


def _block_ngrams(toks, logits, n):
    if n <= 0 or len(toks) < n - 1:
        return logits
    suf = tuple(toks[-(n - 1):])
    for s in range(len(toks) - (n - 1)):
        if tuple(toks[s:s + n - 1]) == suf:
            nxt = toks[s + n - 1]
            if 0 <= nxt < logits.size(0):
                logits[nxt] = float("-inf")
    return logits


def _beam(logits_fn, init_state):
    beams = [(0.0, [], init_state)]
    done  = []
    for _ in range(DECODE_MAX):
        cands = []
        for lp, toks, state in beams:
            logits, ns = logits_fn(toks, state)
            logits = logits.view(-1)[:VOCAB_SIZE]
            if len(toks) < DECODE_MIN and 0 <= EOS < logits.size(0):
                logits[EOS] = float("-inf")
            logits = _block_ngrams(toks, logits, DECODE_NGR)
            lprob  = torch.log_softmax(logits, dim=-1)
            topv, topi = torch.topk(lprob, DECODE_BEAMS)
            for tid, tlp in zip(topi.tolist(), topv.tolist()):
                tid   = _safe_token(tid)
                score = lp + tlp
                if tid == EOS:
                    n = max(len(toks), 1)
                    done.append((score / (n ** DECODE_ALPHA), toks))
                else:
                    cands.append((score, toks + [tid], ns))
        if not cands:
            break
        cands.sort(key=lambda x: x[0], reverse=True)
        beams = cands[:DECODE_BEAMS]
    if not done:
        for lp, toks, _ in beams:
            done.append((lp / (max(len(toks), 1) ** DECODE_ALPHA), toks))
    done.sort(key=lambda x: x[0], reverse=True)
    return done[0][1] if done else []


def _decode(toks):
    words = []
    for t in toks:
        t = _safe_token(t)
        if t == EOS:
            break
        words.append(i2w.get(t, "<unk>"))
    return " ".join(words)


# ════════════════════════════════════════════════════════════════════════════
# LAZY MODEL LOADERS
# ════════════════════════════════════════════════════════════════════════════
_TF_EVAL = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_loaded: Dict[str, object] = {}
_phobert_tok = None


def get_phobert_tok():
    global _phobert_tok
    if _phobert_tok is None:
        _phobert_tok = AutoTokenizer.from_pretrained("vinai/phobert-base")
    return _phobert_tok


def get_model_a1():
    if "A1" in _loaded:
        return _loaded["A1"]
    if not CKPT_A1.exists():
        _loaded["A1"] = None
        return None
    print(f"[app] Loading A1 from {CKPT_A1}…")
    m = VQAModelA1(VOCAB_SIZE).to(DEVICE)
    m.load_state_dict(torch.load(CKPT_A1, map_location=DEVICE, weights_only=True))
    m.eval()
    _loaded["A1"] = m
    print(f"[app] ✓ A1  |  VRAM: {vram_str()}")
    return m


def get_model_a2():
    if "A2" in _loaded:
        return _loaded["A2"]
    if not CKPT_A2.exists():
        _loaded["A2"] = None
        return None
    print(f"[app] Loading A2 from {CKPT_A2}…")
    m = VQAModelA2(VOCAB_SIZE).to(DEVICE)
    m.load_state_dict(torch.load(CKPT_A2, map_location=DEVICE, weights_only=True))
    m.eval()
    _loaded["A2"] = m
    print(f"[app] ✓ A2  |  VRAM: {vram_str()}")
    return m


def get_b1():
    """B1 = BLIP-2 OPT-2.7B + MarianMT.
    Trên Kaggle T4x2: device_map='auto' sẽ chia BLIP-2 qua 2 GPU tự động.
    """
    if "B1" in _loaded:
        return _loaded["B1"]
    try:
        from transformers import (Blip2Processor, Blip2ForConditionalGeneration,
                                  MarianMTModel, MarianTokenizer)
    except ImportError as e:
        print(f"[app] B1 import failed: {e}")
        _loaded["B1"] = None
        return None
    print("[app] Loading B1 (BLIP-2 + MarianMT)…")
    try:
        proc  = Blip2Processor.from_pretrained(MODEL_B1_HF)

        # device_map="auto" phân tải tự động qua T4x2 nếu có
        blip_kwargs = dict(torch_dtype=torch.float16)
        if N_GPUS >= 2:
            blip_kwargs["device_map"] = "auto"   # chia ViT + OPT qua 2 GPU
        else:
            blip_kwargs["device_map"] = {"": 0}

        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_B1_HF, **blip_kwargs).eval()

        tok_vi_en = MarianTokenizer.from_pretrained(MT_VI_EN_HF)
        m_vi_en   = MarianMTModel.from_pretrained(
            MT_VI_EN_HF, torch_dtype=torch.float16).to(DEVICE).eval()
        tok_en_vi = MarianTokenizer.from_pretrained(MT_EN_VI_HF)
        m_en_vi   = MarianMTModel.from_pretrained(
            MT_EN_VI_HF, torch_dtype=torch.float16).to(DEVICE).eval()

        _loaded["B1"] = (proc, model, tok_vi_en, m_vi_en, tok_en_vi, m_en_vi)
        print(f"[app] ✓ B1  |  VRAM: {vram_str()}")
    except Exception as e:
        print(f"[app] B1 load failed: {e}")
        _loaded["B1"] = None
    return _loaded["B1"]


def get_b2():
    """B2 = Qwen2-VL-2B + LoRA (4-bit NF4).
    QUAN TRỌNG trên Kaggle T4x2:
      - device_map={"": 0}  → giữ toàn bộ model trên GPU 0 (tránh crash bitsandbytes)
      - Không dùng device_map="auto" với 4-bit quantization
    """
    if "B2" in _loaded:
        return _loaded["B2"]
    if not CKPT_B2_LORA.exists():
        print(f"[app] B2 skipped — adapter không tìm thấy: {CKPT_B2_LORA}")
        _loaded["B2"] = None
        return None
    try:
        from transformers import (Qwen2VLForConditionalGeneration, AutoProcessor,
                                  BitsAndBytesConfig)
        from peft import PeftModel
    except ImportError as e:
        print(f"[app] B2 import failed: {e}")
        _loaded["B2"] = None
        return None
    print(f"[app] Loading B2 (Qwen2-VL 4-bit + LoRA {CKPT_B2_LORA})…")
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)

        proc = AutoProcessor.from_pretrained(
            MODEL_B2_HF, min_pixels=256 * 28 * 28, max_pixels=384 * 28 * 28)

        base = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_B2_HF,
            quantization_config=bnb,
            device_map={"": 0},        # ← cố định GPU 0, không dùng "auto"
            torch_dtype=torch.float16,
            attn_implementation="sdpa")

        model = PeftModel.from_pretrained(base, str(CKPT_B2_LORA))
        model.config.use_cache = True
        model.eval()

        _loaded["B2"] = (proc, model)
        print(f"[app] ✓ B2  |  VRAM: {vram_str()}")
    except Exception as e:
        print(f"[app] B2 load failed: {e}")
        _loaded["B2"] = None
    return _loaded["B2"]


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def infer_a1(pil_img: Image.Image, question: str) -> str:
    m = get_model_a1()
    if m is None:
        return "⚠️ A1 chưa load (thiếu checkpoint)"
    img = _TF_EVAL(pil_img).unsqueeze(0).to(DEVICE)
    tok = get_phobert_tok()
    enc = tok(question, return_tensors="pt", truncation=True, max_length=64)
    iid = enc["input_ids"].long().to(DEVICE)
    am  = enc["attention_mask"].long().to(DEVICE)
    img_f, txt_f = m.encode(img, iid, am)
    h0 = m.fusion(torch.cat([img_f.mean(1), txt_f], dim=1))
    c0 = torch.zeros_like(h0)

    def step(toks, state):
        h, c   = state
        last   = BOS if not toks else _safe_token(toks[-1])
        w      = torch.tensor([last], dtype=torch.long, device=DEVICE)
        emb    = m.embedding(w)
        ctx, _ = m.attn(img_f, h)
        h, c   = m.lstm(torch.cat([emb, ctx], dim=1), (h, c))
        return m.fc(h).squeeze(0), (h, c)

    return _decode(_beam(step, (h0, c0)))


@torch.no_grad()
def infer_a2(pil_img: Image.Image, question: str) -> str:
    m = get_model_a2()
    if m is None:
        return "⚠️ A2 chưa load (thiếu checkpoint)"
    img = _TF_EVAL(pil_img).unsqueeze(0).to(DEVICE)
    tok = get_phobert_tok()
    enc = tok(question, return_tensors="pt", truncation=True, max_length=64)
    iid = enc["input_ids"].long().to(DEVICE)
    am  = enc["attention_mask"].long().to(DEVICE)
    mem = m.encode(img, iid, am)

    def step(toks, _s):
        seq = [BOS] + [_safe_token(t) for t in toks]
        inp = torch.tensor(seq, dtype=torch.long, device=DEVICE).unsqueeze(0)
        return m.decoder(mem, inp)[0, -1], None

    return _decode(_beam(step, None))


@torch.no_grad()
def infer_b1(pil_img: Image.Image, question: str) -> str:
    bundle = get_b1()
    if bundle is None:
        return "⚠️ B1 không khả dụng"
    proc, model, tok_vi_en, m_vi_en, tok_en_vi, m_en_vi = bundle

    # Vi → En
    enc   = tok_vi_en([question], return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(DEVICE)
    out   = m_vi_en.generate(**enc, max_length=96, num_beams=2)
    q_en  = tok_vi_en.batch_decode(out, skip_special_tokens=True)[0]

    # BLIP-2 — detect device of model (có thể trải qua nhiều GPU)
    _dev  = next(iter(model.parameters())).device
    prompt = f"Question: {q_en.strip().rstrip('?')}? Answer:"
    inp   = proc(images=pil_img, text=prompt, return_tensors="pt", padding=True)
    inp   = {k: (v.to(_dev, torch.float16) if torch.is_floating_point(v)
                 else v.to(_dev)) for k, v in inp.items()}
    ids   = model.generate(**inp, max_new_tokens=20, num_beams=3,
                           min_new_tokens=1, length_penalty=0.8,
                           no_repeat_ngram_size=2, early_stopping=True)
    txt   = proc.batch_decode(ids, skip_special_tokens=True)[0]
    ans_en = txt.split("Answer:")[-1].strip() or "unknown"

    # En → Vi
    enc2  = tok_en_vi([ans_en], return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(DEVICE)
    out2  = m_en_vi.generate(**enc2, max_length=96, num_beams=2)
    return tok_en_vi.batch_decode(out2, skip_special_tokens=True)[0].strip().lower()


_QWEN_SYSTEM = ("Bạn là trợ lý VQA tiếng Việt chuyên về món ăn Việt. "
                "Trả lời câu hỏi NGẮN GỌN (1–6 từ), bằng tiếng Việt, "
                "không thêm giải thích.")


@torch.no_grad()
def infer_b2(pil_img: Image.Image, question: str) -> str:
    bundle = get_b2()
    if bundle is None:
        return "⚠️ B2 không khả dụng"
    proc, model = bundle
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return "⚠️ thiếu qwen-vl-utils"

    # Lưu ảnh tạm vào /kaggle/working/.cache_demo/
    tmp_dir = ROOT / ".cache_demo"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / "current.jpg"
    pil_img.save(tmp_path, format="JPEG")

    msgs = [
        {"role": "system", "content": _QWEN_SYSTEM},
        {"role": "user",   "content": [
            {"type": "image", "image": str(tmp_path)},
            {"type": "text",  "text": question},
        ]},
    ]
    text  = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)
    inp   = proc(text=text, images=image_inputs, padding=True,
                 return_tensors="pt").to(DEVICE)
    gen   = model.generate(**inp, max_new_tokens=32, num_beams=3,
                           do_sample=False,
                           pad_token_id=proc.tokenizer.pad_token_id)
    trimmed = gen[0, inp.input_ids.shape[1]:]
    return proc.tokenizer.decode(trimmed, skip_special_tokens=True).strip().lower()


# ════════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════════
def _token_f1(p: str, g: str) -> float:
    pc, gc_ = Counter(tokenize_vi(p)), Counter(tokenize_vi(g))
    com = sum((pc & gc_).values())
    if not com:
        return 0.0
    pr = com / sum(pc.values())
    rc = com / sum(gc_.values())
    return 2 * pr * rc / (pr + rc)


def _exact(p: str, g: str) -> float:
    return 1.0 if p.strip().lower() == g.strip().lower() else 0.0


def _bleu1(p: str, g: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref, hyp = tokenize_vi(g), tokenize_vi(p)
        if not hyp:
            return 0.0
        return sentence_bleu([ref], hyp, weights=(1, 0, 0, 0),
                             smoothing_function=SmoothingFunction().method1)
    except Exception:
        return float("nan")


def _rouge_l(p: str, g: str) -> float:
    try:
        from rouge_score import rouge_scorer
        return rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=False
        ).score(g, p)["rougeL"].fmeasure
    except Exception:
        return float("nan")


def compute_metrics_table(preds: dict, gt: str) -> pd.DataFrame:
    rows = []
    for name, pred in preds.items():
        rows.append({
            "Model":    name,
            "Prediction": pred[:40],
            "EM":       f"{_exact(pred, gt):.2f}",
            "Token F1": f"{_token_f1(pred, gt):.3f}",
            "BLEU-1":   f"{_bleu1(pred, gt):.3f}",
            "ROUGE-L":  f"{_rouge_l(pred, gt):.3f}",
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# LLM JUDGE
# ════════════════════════════════════════════════════════════════════════════
def llm_judge(provider: str, api_key: str, question: str,
              preds: dict, gt: str) -> dict:
    if not api_key.strip():
        return {}
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            return {"_err": "openai package chưa cài"}
        return _judge_openai(OpenAI(api_key=api_key.strip()), question, preds, gt)
    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            return {"_err": "anthropic package chưa cài"}
        return _judge_anthropic(anthropic.Anthropic(api_key=api_key.strip()),
                                question, preds, gt)
    return {"_err": f"Unknown provider: {provider}"}


def _judge_openai(client, question, preds, gt) -> dict:
    schema = {"type": "object",
              "properties": {"score":  {"type": "number",  "minimum": 0, "maximum": 10},
                             "reason": {"type": "string"}},
              "required": ["score", "reason"], "additionalProperties": False}
    results = {}
    for name, pred in preds.items():
        prompt = (f"Bạn là giám khảo VQA tiếng Việt.\n"
                  f"Câu hỏi: {question}\nĐáp án đúng: {gt}\n"
                  f"Dự đoán [{name}]: {pred}\n\n"
                  'Chấm 0–10. JSON: {"score":<số>,"reason":"<1 câu>"}.')
        try:
            r    = client.responses.create(
                model="gpt-4o-mini", input=prompt, max_output_tokens=120,
                text={"format": {"type": "json_schema", "name": "vqa_judge",
                                 "schema": schema, "strict": True}})
            data = json.loads(r.output_text.strip())
            results[name] = {"score":  max(0.0, min(10.0, float(data["score"]))),
                             "reason": str(data.get("reason", ""))}
        except Exception as e:
            results[name] = {"score": 0.0, "reason": f"Error: {e}"}
    return results


def _judge_anthropic(client, question, preds, gt) -> dict:
    results = {}
    for name, pred in preds.items():
        prompt = (f"Bạn là giám khảo VQA tiếng Việt.\n"
                  f"Câu hỏi: {question}\nĐáp án đúng: {gt}\n"
                  f"Dự đoán [{name}]: {pred}\n\n"
                  'Chấm 0–10. JSON: {"score":<số>,"reason":"<1 câu tiếng Việt>"}')
        try:
            msg  = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=150,
                messages=[{"role": "user", "content": prompt}])
            raw  = (msg.content[0].text.strip()
                    .replace("```json", "").replace("```", "").strip())
            r    = json.loads(raw)
            results[name] = {"score":  max(0.0, min(10.0, float(r["score"]))),
                             "reason": str(r.get("reason", ""))}
        except Exception as e:
            results[name] = {"score": 0.0, "reason": f"Error: {e}"}
    return results


# ════════════════════════════════════════════════════════════════════════════
# AGGREGATE RESULTS
# ════════════════════════════════════════════════════════════════════════════
def load_aggregate_results():
    res_a = RESULTS_DIR / "results_A.json"
    res_b = RESULTS_DIR / "results_B.json"
    if not (res_a.exists() and res_b.exists()):
        return None
    a = json.loads(res_a.read_text(encoding="utf-8"))
    b = json.loads(res_b.read_text(encoding="utf-8"))
    rows = []
    for mid, src in [("A1", a["A1"]), ("A2", a["A2"]),
                     ("B1", b["B1"]), ("B2", b["B2"])]:
        for split in ["val", "test"]:
            d = src[split]
            rows.append({
                "Model":     mid,
                "Split":     split.title(),
                "EM":        round(d.get("Exact Match",     0) or 0, 4),
                "F1":        round(d.get("Token F1 (Soft)", 0) or 0, 4),
                "BLEU-1":    round(d.get("BLEU-1",          0) or 0, 4),
                "ROUGE-L":   round(d.get("ROUGE-L",         0) or 0, 4),
                "BERTScore": round(d.get("BERTScore F1",    0) or 0, 4),
                "SemSim":    round(d.get("Semantic Sim",    0) or 0, 4),
            })
    return pd.DataFrame(rows)


AGG_DF = load_aggregate_results()


# ════════════════════════════════════════════════════════════════════════════
# SAMPLE EXAMPLES
# ════════════════════════════════════════════════════════════════════════════
def get_sample_examples():
    if not TEST_JSON.exists():
        return []
    test = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    test_only = [d for d in test if d["image"].startswith("test/")
                 and (IMG_DIR / d["image"]).exists()]
    by_cat = {}
    for d in test_only:
        cat = d["image"].split("/")[1]
        by_cat.setdefault(cat, []).append(d)
    samples = []
    for cat in sorted(by_cat.keys())[:6]:
        items = by_cat[cat]
        for q_type in ["recognition", "attribute", "yes_no"]:
            for it in items:
                if it.get("type") == q_type:
                    samples.append([str(IMG_DIR / it["image"]),
                                    it["question"], it["answer"]])
                    break
            if len(samples) > len(samples[:-1]):
                break
    return samples


# ════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ════════════════════════════════════════════════════════════════════════════
MODEL_META = {
    "A1": ("LSTM + Attention",    "#3b82f6"),
    "A2": ("Transformer Decoder", "#10b981"),
    "B1": ("BLIP-2 (zero-shot)",  "#f59e0b"),
    "B2": ("Qwen2-VL + LoRA",     "#8b5cf6"),
}

_CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.gr-button-primary {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  border: 0 !important;
}
#header-grad {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
  border-radius: 14px; padding: 24px 28px; margin-bottom: 14px; color: white;
}
.status-box textarea {
  background: #f9fafb !important; font-family: ui-monospace, monospace !important;
  font-size: 12px !important; color: #4b5563 !important; border: none !important;
}
"""


def _pred_card(name: str, pred: str, elapsed_ms: int) -> str:
    sub, color = MODEL_META[name]
    short = (pred[:90] + "…") if len(pred) > 90 else pred
    return f'''
    <div style="border:1px solid #e5e7eb;border-left:4px solid {color};
                border-radius:10px;padding:14px 16px;background:#fff;
                box-shadow:0 1px 2px rgba(0,0,0,0.04)">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:8px">
        <div>
          <span style="font-size:14px;font-weight:700;color:{color}">{name}</span>
          <span style="font-size:11px;color:#6b7280;margin-left:8px">{sub}</span>
        </div>
        <span style="font-size:10px;color:#9ca3af">{elapsed_ms} ms</span>
      </div>
      <div style="font-size:15px;color:#111827;font-weight:500;
                  line-height:1.5;min-height:24px">{short}</div>
    </div>'''


def run_demo(image, question: str, gt: str,
             judge_provider: str, api_key: str, use_judge: bool):
    if image is None:
        return ("<p style='color:#dc2626'>⚠️ Vui lòng upload ảnh.</p>",
                None, "<p style='color:#9ca3af'>—</p>", "❌ Thiếu ảnh")
    if not question.strip():
        return ("<p style='color:#dc2626'>⚠️ Vui lòng nhập câu hỏi.</p>",
                None, "<p style='color:#9ca3af'>—</p>", "❌ Thiếu câu hỏi")

    pil = _load_pil(image)
    timings, preds = {}, {}
    for name, fn in [("A1", infer_a1), ("A2", infer_a2),
                     ("B1", infer_b1), ("B2", infer_b2)]:
        t = time.time()
        try:
            preds[name] = fn(pil, question)
        except Exception as e:
            preds[name] = f"⚠️ {e}"
        timings[name] = int((time.time() - t) * 1000)

    cards = ("<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:12px'>"
             + "".join(_pred_card(n, preds[n], timings[n]) for n in ["A1","A2","B1","B2"])
             + "</div>")

    metrics_df = compute_metrics_table(preds, gt) if gt.strip() else None

    if use_judge and api_key.strip() and gt.strip():
        try:
            jres = llm_judge(judge_provider, api_key, question, preds, gt)
            if "_err" in jres:
                judge_html = f'<p style="color:#dc2626">⚠️ {jres["_err"]}</p>'
            else:
                cj = ("<div style='display:grid;grid-template-columns:"
                      "repeat(4,1fr);gap:10px'>")
                for name in ["A1", "A2", "B1", "B2"]:
                    r = jres.get(name, {"score": 0, "reason": ""})
                    _, color = MODEL_META[name]
                    s = r["score"]
                    cj += f'''
                    <div style="border:1px solid #e5e7eb;border-radius:10px;
                                padding:14px;background:#fff;text-align:center">
                      <div style="font-size:11px;color:#6b7280;font-weight:600;
                                  margin-bottom:6px">{name}</div>
                      <div style="font-size:32px;font-weight:700;color:{color};
                                  line-height:1">{s:.1f}</div>
                      <div style="font-size:10px;color:#9ca3af;margin-top:2px">/10</div>
                      <div style="height:4px;background:#f3f4f6;border-radius:2px;
                                  margin:10px 0 8px;overflow:hidden">
                        <div style="width:{int(s*10)}%;height:100%;background:{color};
                                    border-radius:2px"></div></div>
                      <div style="font-size:11px;color:#4b5563;line-height:1.4;
                                  text-align:left">{r["reason"][:100]}</div>
                    </div>'''
                judge_html = cj + "</div>"
        except Exception as e:
            judge_html = f'<p style="color:#dc2626">⚠️ LLM Judge lỗi: {e}</p>'
    elif use_judge and not gt.strip():
        judge_html = '<p style="color:#f59e0b">ℹ️ Cần nhập Ground Truth.</p>'
    elif use_judge and not api_key.strip():
        judge_html = '<p style="color:#f59e0b">ℹ️ Cần nhập API key.</p>'
    else:
        judge_html = ('<p style="font-size:12px;color:#9ca3af">'
                      'Bật LLM Judge để xem điểm.</p>')

    total_ms = sum(timings.values())
    status   = (f"✅  A1:{timings['A1']}ms · A2:{timings['A2']}ms · "
                f"B1:{timings['B1']}ms · B2:{timings['B2']}ms  |  "
                f"Total {total_ms}ms  |  VRAM: {vram_str()}")
    return cards, metrics_df, judge_html, status


def build_app():
    samples = get_sample_examples()

    with gr.Blocks(title="Vietnamese VQA Demo", css=_CSS,
                   theme=gr.themes.Soft(primary_hue="indigo")) as demo:

        gr.HTML(f"""
        <div id="header-grad">
          <div style="display:flex;align-items:center;justify-content:space-between;
                      flex-wrap:wrap;gap:12px">
            <div>
              <h1 style="font-size:24px;font-weight:700;margin:0 0 4px 0">
                🍜 Vietnamese Food VQA
              </h1>
              <p style="font-size:13px;opacity:0.92;margin:0">
                So sánh trực tiếp 4 cấu hình · Kaggle T4×{"2" if N_GPUS>=2 else "1"}
              </p>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap">
              {''.join(f'''<span style="font-size:10px;padding:4px 10px;border-radius:12px;
                background:rgba(255,255,255,0.2);font-weight:600">{t}</span>'''
                for t in ["A1·LSTM","A2·Transformer","B1·BLIP-2","B2·Qwen2-VL"])}
            </div>
          </div>
          <div style="margin-top:10px;font-size:11px;opacity:0.85;
                      font-family:ui-monospace,monospace">
            GPUs: {N_GPUS}  ·  VRAM: {vram_str()}  ·
            Vocab: {VOCAB_SIZE}  ·  Samples: {len(samples)}
          </div>
        </div>""")

        with gr.Tabs():

            # ── TAB 1: Demo ───────────────────────────────────────────────
            with gr.Tab("🎯 Demo"):
                with gr.Row():
                    with gr.Column(scale=4):
                        image_in    = gr.Image(label="📷 Ảnh", type="pil", height=300)
                        question_in = gr.Textbox(
                            label="❓ Câu hỏi (tiếng Việt)",
                            placeholder="VD: Đây là món ăn gì?", lines=2)
                        gt_in = gr.Textbox(
                            label="✅ Ground Truth (tuỳ chọn)",
                            placeholder="VD: phở bò", lines=1)
                        with gr.Accordion("⚙️ LLM Judge", open=False):
                            judge_provider = gr.Radio(
                                ["openai", "anthropic"], value="openai",
                                label="Provider")
                            api_key_in = gr.Textbox(
                                label="🔑 API Key", type="password", lines=1,
                                placeholder="sk-... hoặc sk-ant-...")
                            use_judge_chk = gr.Checkbox(
                                label="Dùng LLM-as-a-Judge", value=False)
                        run_btn = gr.Button("▶  Chạy 4 models",
                                            variant="primary", size="lg")

                    with gr.Column(scale=6):
                        gr.HTML('<div style="font-size:13px;font-weight:600;'
                                'color:#374151;margin-bottom:8px">📝 Predictions</div>')
                        preds_out  = gr.HTML(
                            value='<div style="color:#9ca3af;font-size:13px;padding:40px;'
                                  'text-align:center;border:2px dashed #e5e7eb;'
                                  'border-radius:12px">Upload ảnh + nhập câu hỏi</div>')
                        status_out = gr.Textbox(
                            label="", interactive=False, show_label=False,
                            elem_classes="status-box", max_lines=1)

                if samples:
                    gr.HTML('<div style="font-size:13px;font-weight:600;'
                            'color:#374151;margin:14px 0 4px">💡 Ảnh mẫu</div>')
                    gr.Examples(examples=samples,
                                inputs=[image_in, question_in, gt_in],
                                examples_per_page=6, label="")

                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div style="font-size:13px;font-weight:600;'
                                'color:#374151;margin:14px 0 6px">📊 Metrics</div>')
                        metrics_out = gr.Dataframe(
                            headers=["Model","Prediction","EM",
                                     "Token F1","BLEU-1","ROUGE-L"],
                            datatype=["str"]*6, interactive=False)

                gr.HTML('<div style="font-size:13px;font-weight:600;color:#374151;'
                        'margin:14px 0 6px">🤖 LLM-as-a-Judge (0–10)</div>')
                judge_out = gr.HTML(
                    value='<p style="font-size:12px;color:#9ca3af">'
                          'Bật LLM Judge để xem điểm.</p>')

            # ── TAB 2: Aggregate Results ──────────────────────────────────
            with gr.Tab("📈 Kết quả tổng quan"):
                if AGG_DF is None:
                    gr.HTML('<p style="color:#9ca3af;padding:20px">'
                            'Không tìm thấy results/*.json. '
                            'Chạy vqa-model-a.ipynb và vqa-model-b.ipynb trước.</p>')
                else:
                    gr.Dataframe(value=AGG_DF, interactive=False,
                                 datatype=["str","str"]+["number"]*6)
                    best_html = ("<div style='display:grid;"
                                 "grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px'>")
                    for mc in ["F1", "BERTScore", "SemSim"]:
                        tdf = AGG_DF[AGG_DF["Split"] == "Test"]
                        idx = tdf[mc].idxmax()
                        winner = tdf.loc[idx, "Model"]
                        val    = tdf.loc[idx, mc]
                        _, color = MODEL_META.get(winner, ("", "#6b7280"))
                        best_html += f'''
                        <div style="border:1px solid #e5e7eb;border-radius:10px;
                                    padding:14px;background:#fff;text-align:center">
                          <div style="font-size:11px;color:#6b7280;margin-bottom:4px">
                            🏆 Best {mc} (Test)</div>
                          <div style="font-size:18px;font-weight:700;color:{color}">
                            {winner}</div>
                          <div style="font-size:14px;color:#374151;margin-top:4px">
                            {val:.4f}</div>
                        </div>'''
                    gr.HTML(best_html + "</div>")

            # ── TAB 3: About ─────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.HTML(f"""
                <div style="padding:20px 4px;font-size:13px;color:#374151;line-height:1.8">
                  <h3 style="margin-top:0">Vietnamese Food VQA — Kaggle Edition</h3>
                  <p>Môi trường: Kaggle T4×{N_GPUS}  ·  VRAM: {vram_str()}</p>
                  <h4>4 cấu hình</h4>
                  <ul style="line-height:2.2">
                    <li><b style="color:#3b82f6">A1</b> — ResNet50 + PhoBERT + LSTM
                        + Bahdanau Attention. <i>Train from scratch.</i></li>
                    <li><b style="color:#10b981">A2</b> — ResNet50 + PhoBERT
                        + Transformer Decoder (3L, 8H). <i>Train from scratch.</i></li>
                    <li><b style="color:#f59e0b">B1</b> — Salesforce/blip2-opt-2.7b
                        (zero-shot) + MarianMT Vi↔En.
                        <i>device_map=auto → tự chia qua {N_GPUS} GPU.</i></li>
                    <li><b style="color:#8b5cf6">B2</b> — Qwen2-VL-2B-Instruct
                        + LoRA r=16 (4-bit NF4, GPU 0).
                        <i>Fine-tune tiếng Việt.</i></li>
                  </ul>
                  <h4>Paths</h4>
                  <ul style="font-family:monospace;font-size:12px">
                    <li>DATA : {DATA_DIR}</li>
                    <li>CKPT : {CKPT_DIR}</li>
                    <li>A1   : {"✓" if CKPT_A1.exists() else "✗ MISSING"} {CKPT_A1}</li>
                    <li>A2   : {"✓" if CKPT_A2.exists() else "✗ MISSING"} {CKPT_A2}</li>
                    <li>B2   : {"✓" if CKPT_B2_LORA.exists() else "✗ MISSING"} {CKPT_B2_LORA}</li>
                  </ul>
                </div>""")

        run_btn.click(
            fn=run_demo,
            inputs=[image_in, question_in, gt_in,
                    judge_provider, api_key_in, use_judge_chk],
            outputs=[preds_out, metrics_out, judge_out, status_out])
        question_in.submit(
            fn=run_demo,
            inputs=[image_in, question_in, gt_in,
                    judge_provider, api_key_in, use_judge_chk],
            outputs=[preds_out, metrics_out, judge_out, status_out])

    return demo


# ════════════════════════════════════════════════════════════════════════════
# LAUNCH
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  Vietnamese Food VQA — Kaggle Edition")
print("=" * 62)
print(f"  IS_KAGGLE : {IS_KAGGLE}")
print(f"  GPUs      : {N_GPUS}")
print(f"  A1        : {'[OK]' if CKPT_A1.exists()      else '[MISSING]'}")
print(f"  A2        : {'[OK]' if CKPT_A2.exists()      else '[MISSING]'}")
print(f"  B2 LoRA   : {'[OK]' if CKPT_B2_LORA.exists() else '[MISSING]'}")
print(f"  results   : {'[OK]' if AGG_DF is not None    else '[MISSING]'}")
print("=" * 62 + "\n")

# ─── Tìm port trống (Kaggle thường đã chiếm 7860) ──────────────────────────
def _free_port(start: int = 7860, span: int = 50) -> int:
    import socket
    for p in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return 0  # 0 → để OS tự chọn

_PORT = int(os.environ.get("GRADIO_SERVER_PORT", 0)) or _free_port()
print(f"[app] Launching Gradio on port {_PORT or 'auto'} (share=True) …")

demo = build_app()
demo.queue(default_concurrency_limit=1, max_size=20).launch(
    server_name = "0.0.0.0",
    server_port = _PORT or None,   # None → Gradio tự chọn
    share       = True,            # ← BẮT BUỘC trên Kaggle để lấy public URL
    inbrowser   = False,           # ← Không mở browser trên Kaggle
    debug       = False,
    show_error  = True,
    quiet       = False,
    prevent_thread_lock = False,
)
