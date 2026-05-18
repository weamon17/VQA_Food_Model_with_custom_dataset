"""
VQA Food Demo — Vietnamese Visual Question Answering
=====================================================
So sánh trực tiếp 4 cấu hình model trên cùng (ảnh + câu hỏi):

  A1  ResNet50 + PhoBERT + LSTM (Bahdanau attention)   — train from scratch
  A2  ResNet50 + PhoBERT + Transformer Decoder (3L,8H) — train from scratch
  B1  BLIP-2 OPT-2.7B + MarianMT Vi↔En                 — zero-shot
  B2  Qwen2-VL-2B-Instruct + LoRA (4-bit NF4)          — fine-tuned

Chạy:
    pip install gradio>=4.0 openai anthropic qwen-vl-utils peft bitsandbytes \\
                transformers torch torchvision nltk rouge-score pillow pandas
    python app.py

Models lazy-loaded khi gọi lần đầu (tiết kiệm VRAM cho GPU ≤8GB).
"""

# ── Standard library ────────────────────────────────────────────────────────
import os, re, sys, gc, json, math, time, platform, warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

if platform.system() == "Windows":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
ROOT       = Path(__file__).resolve().parent
DATA_DIR   = ROOT / "data"
IMG_DIR    = DATA_DIR / "images"
TRAIN_JSON = DATA_DIR / "annotations" / "train.json"
TEST_JSON  = DATA_DIR / "annotations" / "test.json"
CKPT_DIR   = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

CKPT_A1     = CKPT_DIR / "best_model_A1.pth"
CKPT_A2     = CKPT_DIR / "best_model_A2.pth"
CKPT_B2_LORA = CKPT_DIR / "qwen2vl_lora_b2" / "adapter_best"

MODEL_B1_HF  = "Salesforce/blip2-opt-2.7b"
MODEL_B2_HF  = "Qwen/Qwen2-VL-2B-Instruct"
MT_VI_EN_HF  = "Helsinki-NLP/opus-mt-vi-en"
MT_EN_VI_HF  = "Helsinki-NLP/opus-mt-en-vi"

# Beam search hyperparameters (match training)
DECODE_MAX, DECODE_MIN, DECODE_BEAMS, DECODE_ALPHA, DECODE_NGR = 10, 3, 5, 0.7, 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[app] Device: {DEVICE}")
if DEVICE.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"[app] GPU: {p.name}  |  VRAM: {p.total_memory/1024**3:.1f} GB")


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
# VOCAB cho Model A (rebuild deterministic từ train.json)
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
    print(f"[app] WARNING: {TRAIN_JSON} không tồn tại — A1/A2 sẽ không load.")
    w2i, i2w = {}, {}
    PAD = BOS = EOS = UNK = 0
    VOCAB_SIZE = 100


# ════════════════════════════════════════════════════════════════════════════
# MODEL A — Architecture (must match training notebook)
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
        self.fc = nn.Linear(d, vocab_size)

    def forward(self, memory, captions):
        tgt = self.pos(self.emb(captions))
        sz = tgt.size(1)
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=tgt.device), 1)
        return self.fc(self.dec(tgt=tgt, memory=memory, tgt_mask=mask))


class MultimodalFusion(nn.Module):
    def __init__(self, d=512, dropout=0.1):
        super().__init__()
        self.fc = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, img, txt):
        return self.norm(self.drop(self.fc(img + txt.unsqueeze(1))))


class VQAModelA1(nn.Module):
    def __init__(self, vocab_size, d=512, emb_dim=256):
        super().__init__()
        self.img_enc = ImageEncoder(d)
        self.txt_enc = AutoModel.from_pretrained("vinai/phobert-base", use_safetensors=True)
        for p in self.txt_enc.parameters():
            p.requires_grad = False
        self.txt_proj = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d), nn.Dropout(0.1))
        self.fusion = nn.Sequential(nn.Linear(d * 2, d), nn.Tanh())
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.attn = BahdanauAttention(d)
        self.lstm = nn.LSTMCell(emb_dim + d, d)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(d, vocab_size)

    def encode(self, image, input_ids, attn_mask):
        img = self.img_enc(image)
        with torch.no_grad():
            out = self.txt_enc(input_ids=input_ids, attention_mask=attn_mask)
        txt = self.txt_proj(out.last_hidden_state[:, 0])
        return img, txt


class VQAModelA2(nn.Module):
    def __init__(self, vocab_size, d=512):
        super().__init__()
        self.img_enc = ImageEncoder(d)
        self.txt_enc = AutoModel.from_pretrained("vinai/phobert-base", use_safetensors=True)
        for p in self.txt_enc.parameters():
            p.requires_grad = False
        self.txt_proj = nn.Sequential(nn.Linear(768, d), nn.LayerNorm(d), nn.Dropout(0.1))
        self.fusion = MultimodalFusion(d)
        self.decoder = TrfDecoder(vocab_size, d)

    def encode(self, image, input_ids, attn_mask):
        img = self.img_enc(image)
        with torch.no_grad():
            out = self.txt_enc(input_ids=input_ids, attention_mask=attn_mask)
        txt = self.txt_proj(out.last_hidden_state[:, 0])
        return self.fusion(img, txt)


# ════════════════════════════════════════════════════════════════════════════
# BEAM SEARCH (cho A1, A2)
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
    done = []
    for _ in range(DECODE_MAX):
        cands = []
        for lp, toks, state in beams:
            logits, ns = logits_fn(toks, state)
            logits = logits.view(-1)[:VOCAB_SIZE]
            if len(toks) < DECODE_MIN and 0 <= EOS < logits.size(0):
                logits[EOS] = float("-inf")
            logits = _block_ngrams(toks, logits, DECODE_NGR)
            lprob = torch.log_softmax(logits, dim=-1)
            topv, topi = torch.topk(lprob, DECODE_BEAMS)
            for tid, tlp in zip(topi.tolist(), topv.tolist()):
                tid = _safe_token(tid)
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
# LAZY MODEL LOADERS (load lần đầu, cache về sau)
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
    print(f"[app] [OK] A1 loaded   |  VRAM: {vram_str()}")
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
    print(f"[app] [OK] A2 loaded   |  VRAM: {vram_str()}")
    return m


def get_b1():
    """B1 = BLIP-2 OPT-2.7B + MarianMT Vi↔En (~6 GB VRAM, lazy load)."""
    if "B1" in _loaded:
        return _loaded["B1"]
    try:
        from transformers import (Blip2Processor, Blip2ForConditionalGeneration,
                                  MarianMTModel, MarianTokenizer)
    except ImportError as e:
        print(f"[app] B1 import failed: {e}")
        _loaded["B1"] = None
        return None
    print("[app] Loading B1 (BLIP-2 OPT-2.7B + MarianMT) — lần đầu ~7 GB download…")
    try:
        proc = Blip2Processor.from_pretrained(MODEL_B1_HF)
        model = Blip2ForConditionalGeneration.from_pretrained(
            MODEL_B1_HF, torch_dtype=torch.float16, device_map={"": 0}).eval()
        tok_vi_en = MarianTokenizer.from_pretrained(MT_VI_EN_HF)
        m_vi_en = MarianMTModel.from_pretrained(
            MT_VI_EN_HF, torch_dtype=torch.float16).to(DEVICE).eval()
        tok_en_vi = MarianTokenizer.from_pretrained(MT_EN_VI_HF)
        m_en_vi = MarianMTModel.from_pretrained(
            MT_EN_VI_HF, torch_dtype=torch.float16).to(DEVICE).eval()
        _loaded["B1"] = (proc, model, tok_vi_en, m_vi_en, tok_en_vi, m_en_vi)
        print(f"[app] [OK] B1 loaded   |  VRAM: {vram_str()}")
    except Exception as e:
        print(f"[app] B1 load failed: {e}")
        _loaded["B1"] = None
    return _loaded["B1"]


def get_b2():
    """B2 = Qwen2-VL-2B + LoRA (~2 GB 4-bit, lazy load)."""
    if "B2" in _loaded:
        return _loaded["B2"]
    if not CKPT_B2_LORA.exists():
        print(f"[app] [WARN] B2 skipped ({CKPT_B2_LORA} not found)")
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
    print(f"[app] Loading B2 (Qwen2-VL + LoRA from {CKPT_B2_LORA})…")
    try:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
        proc = AutoProcessor.from_pretrained(
            MODEL_B2_HF, min_pixels=256*28*28, max_pixels=384*28*28)
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_B2_HF, quantization_config=bnb, device_map={"": 0},
            torch_dtype=torch.float16, attn_implementation="sdpa")
        model = PeftModel.from_pretrained(base, str(CKPT_B2_LORA))
        model.config.use_cache = True
        model.eval()
        _loaded["B2"] = (proc, model)
        print(f"[app] [OK] B2 loaded   |  VRAM: {vram_str()}")
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
        h, c = state
        last = BOS if not toks else _safe_token(toks[-1])
        w = torch.tensor([last], dtype=torch.long, device=DEVICE)
        emb = m.embedding(w)
        ctx, _ = m.attn(img_f, h)
        h, c = m.lstm(torch.cat([emb, ctx], dim=1), (h, c))
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
        return "⚠️ B1 không khả dụng (thiếu thư viện hoặc OOM)"
    proc, model, tok_vi_en, m_vi_en, tok_en_vi, m_en_vi = bundle

    # Vi → En
    enc = tok_vi_en([question], return_tensors="pt", padding=True,
                    truncation=True, max_length=96).to(DEVICE)
    out = m_vi_en.generate(**enc, max_length=96, num_beams=2)
    q_en = tok_vi_en.batch_decode(out, skip_special_tokens=True)[0]

    # BLIP-2 inference
    prompt = f"Question: {q_en.strip().rstrip('?')}? Answer:"
    inp = proc(images=pil_img, text=prompt, return_tensors="pt", padding=True)
    inp = {k: (v.to(DEVICE, torch.float16) if torch.is_floating_point(v)
               else v.to(DEVICE)) for k, v in inp.items()}
    ids = model.generate(**inp, max_new_tokens=20, num_beams=3, min_new_tokens=1,
                         length_penalty=0.8, no_repeat_ngram_size=2,
                         early_stopping=True)
    txt = proc.batch_decode(ids, skip_special_tokens=True)[0]
    ans_en = txt.split("Answer:")[-1].strip() or "unknown"

    # En → Vi
    enc2 = tok_en_vi([ans_en], return_tensors="pt", padding=True,
                     truncation=True, max_length=96).to(DEVICE)
    out2 = m_en_vi.generate(**enc2, max_length=96, num_beams=2)
    return tok_en_vi.batch_decode(out2, skip_special_tokens=True)[0].strip().lower()


_QWEN_SYSTEM = ("Bạn là trợ lý VQA tiếng Việt chuyên về món ăn Việt. "
                "Trả lời câu hỏi NGẮN GỌN (1–6 từ), bằng tiếng Việt, không thêm giải thích.")


@torch.no_grad()
def infer_b2(pil_img: Image.Image, question: str) -> str:
    bundle = get_b2()
    if bundle is None:
        return "⚠️ B2 không khả dụng (thiếu adapter hoặc thư viện)"
    proc, model = bundle
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        return "⚠️ thiếu qwen-vl-utils (pip install qwen-vl-utils)"

    # Lưu ảnh tạm để Qwen processor đọc từ path
    tmp_dir = ROOT / ".cache_demo"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / "current.jpg"
    pil_img.save(tmp_path, format="JPEG")

    msgs = [
        {"role": "system", "content": _QWEN_SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": str(tmp_path)},
            {"type": "text",  "text": question},
        ]},
    ]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)
    inp = proc(text=text, images=image_inputs, padding=True,
               return_tensors="pt").to(DEVICE)
    gen = model.generate(**inp, max_new_tokens=32, num_beams=3, do_sample=False,
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
        sf = SmoothingFunction().method1
        return sentence_bleu([ref], hyp, weights=(1, 0, 0, 0),
                             smoothing_function=sf)
    except ImportError:
        return float("nan")


def _rouge_l(p: str, g: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return scorer.score(g, p)["rougeL"].fmeasure
    except ImportError:
        return float("nan")


def compute_metrics_table(preds: dict, gt: str) -> pd.DataFrame:
    rows = []
    for name, pred in preds.items():
        rows.append({
            "Model":   name,
            "Prediction": pred[:40],
            "EM":      f"{_exact(pred, gt):.2f}",
            "Token F1": f"{_token_f1(pred, gt):.3f}",
            "BLEU-1":  f"{_bleu1(pred, gt):.3f}",
            "ROUGE-L": f"{_rouge_l(pred, gt):.3f}",
        })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# LLM JUDGE (OpenAI hoặc Anthropic)
# ════════════════════════════════════════════════════════════════════════════
def llm_judge(provider: str, api_key: str, question: str,
              preds: dict, gt: str) -> dict:
    """Gọi LLM chấm điểm 4 model. provider ∈ {'openai', 'anthropic'}."""
    if not api_key.strip():
        return {}

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            return {"_err": "openai package chưa cài"}
        client = OpenAI(api_key=api_key.strip())
        return _judge_openai(client, question, preds, gt)
    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            return {"_err": "anthropic package chưa cài"}
        client = anthropic.Anthropic(api_key=api_key.strip())
        return _judge_anthropic(client, question, preds, gt)
    else:
        return {"_err": f"Unknown provider: {provider}"}


def _judge_openai(client, question, preds, gt) -> dict:
    schema = {
        "type": "object",
        "properties": {
            "score":  {"type": "number", "minimum": 0, "maximum": 10},
            "reason": {"type": "string"},
        },
        "required": ["score", "reason"],
        "additionalProperties": False,
    }
    results = {}
    for name, pred in preds.items():
        prompt = (
            f"Bạn là giám khảo VQA tiếng Việt.\n"
            f"Câu hỏi: {question}\nĐáp án đúng: {gt}\nDự đoán [{name}]: {pred}\n\n"
            "Chấm 0–10 (0=sai hoàn toàn, 10=đúng tuyệt đối). "
            'JSON: {"score":<số>,"reason":"<1 câu>"}.'
        )
        try:
            r = client.responses.create(
                model="gpt-4o-mini", input=prompt, max_output_tokens=120,
                text={"format": {"type": "json_schema",
                                 "name": "vqa_judge",
                                 "schema": schema, "strict": True}},
            )
            data = json.loads(r.output_text.strip())
            results[name] = {"score": max(0.0, min(10.0, float(data["score"]))),
                             "reason": str(data.get("reason", ""))}
        except Exception as e:
            results[name] = {"score": 0.0, "reason": f"Error: {e}"}
    return results


def _judge_anthropic(client, question, preds, gt) -> dict:
    results = {}
    for name, pred in preds.items():
        prompt = (
            f"Bạn là giám khảo VQA tiếng Việt.\n"
            f"Câu hỏi: {question}\nĐáp án đúng: {gt}\nDự đoán [{name}]: {pred}\n\n"
            "Chấm 0–10. Trả về JSON: "
            '{"score":<số>,"reason":"<1 câu tiếng Việt>"}'
        )
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip().replace("```json", "").replace("```", "").strip()
            r = json.loads(raw)
            results[name] = {"score": max(0.0, min(10.0, float(r["score"]))),
                             "reason": str(r.get("reason", ""))}
        except Exception as e:
            results[name] = {"score": 0.0, "reason": f"Error: {e}"}
    return results


# ════════════════════════════════════════════════════════════════════════════
# AGGREGATE RESULTS từ results JSON (đọc lúc startup)
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
                "Model":   mid,
                "Split":   split.title(),
                "EM":      round(d.get("Exact Match",     0) or 0, 4),
                "F1":      round(d.get("Token F1 (Soft)", 0) or 0, 4),
                "BLEU-1":  round(d.get("BLEU-1",          0) or 0, 4),
                "ROUGE-L": round(d.get("ROUGE-L",         0) or 0, 4),
                "BERTScore": round(d.get("BERTScore F1",  0) or 0, 4),
                "SemSim":  round(d.get("Semantic Sim",    0) or 0, 4),
            })
    return pd.DataFrame(rows)


AGG_DF = load_aggregate_results()


# ════════════════════════════════════════════════════════════════════════════
# GRADIO CALLBACK
# ════════════════════════════════════════════════════════════════════════════
MODEL_META = {
    "A1": ("LSTM + Attention",     "#3b82f6"),
    "A2": ("Transformer Decoder",  "#10b981"),
    "B1": ("BLIP-2 (zero-shot)",   "#f59e0b"),
    "B2": ("Qwen2-VL + LoRA",      "#8b5cf6"),
}


def _pred_card(name: str, pred: str, elapsed_ms: int) -> str:
    sub, color = MODEL_META[name]
    short = (pred[:90] + "…") if len(pred) > 90 else pred
    return f'''
    <div style="border:1px solid #e5e7eb;border-left:4px solid {color};
                border-radius:10px;padding:14px 16px;background:#fff;
                box-shadow:0 1px 2px rgba(0,0,0,0.04);height:100%">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:8px">
        <div>
          <span style="font-size:14px;font-weight:700;color:{color}">{name}</span>
          <span style="font-size:11px;color:#6b7280;margin-left:8px">{sub}</span>
        </div>
        <span style="font-size:10px;color:#9ca3af">{elapsed_ms} ms</span>
      </div>
      <div style="font-size:15px;color:#111827;line-height:1.5;
                  font-weight:500;min-height:24px">{short}</div>
    </div>
    '''


def run_demo(image, question: str, gt: str,
             judge_provider: str, api_key: str, use_judge: bool):
    if image is None:
        return ("<p style='color:#dc2626'>⚠️ Vui lòng upload ảnh.</p>",
                None,
                "<p style='font-size:12px;color:#9ca3af'>—</p>",
                "❌ Thiếu ảnh")
    if not question.strip():
        return ("<p style='color:#dc2626'>⚠️ Vui lòng nhập câu hỏi.</p>",
                None,
                "<p style='font-size:12px;color:#9ca3af'>—</p>",
                "❌ Thiếu câu hỏi")

    pil = _load_pil(image)
    timings, preds = {}, {}

    for name, fn in [("A1", infer_a1), ("A2", infer_a2),
                     ("B1", infer_b1), ("B2", infer_b2)]:
        t = time.time()
        try:
            preds[name] = fn(pil, question)
        except Exception as e:
            preds[name] = f"⚠️ Error: {e}"
        timings[name] = int((time.time() - t) * 1000)

    # ── Prediction cards ──────────────────────────────────────────────────
    cards = "<div style='display:grid;grid-template-columns:repeat(2,1fr);gap:12px'>"
    for name in ["A1", "A2", "B1", "B2"]:
        cards += _pred_card(name, preds[name], timings[name])
    cards += "</div>"

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics_df = compute_metrics_table(preds, gt) if gt.strip() else None

    # ── LLM Judge ─────────────────────────────────────────────────────────
    if use_judge and api_key.strip() and gt.strip():
        try:
            jres = llm_judge(judge_provider, api_key, question, preds, gt)
            if "_err" in jres:
                judge_html = f'<p style="color:#dc2626">⚠️ {jres["_err"]}</p>'
            else:
                cards_j = "<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px'>"
                for name in ["A1", "A2", "B1", "B2"]:
                    r = jres.get(name, {"score": 0, "reason": ""})
                    _, color = MODEL_META[name]
                    score = r["score"]
                    bar_w = int(score * 10)
                    cards_j += f'''
                    <div style="border:1px solid #e5e7eb;border-radius:10px;
                                padding:14px;background:#fff;text-align:center">
                      <div style="font-size:11px;color:#6b7280;font-weight:600;
                                  margin-bottom:6px">{name}</div>
                      <div style="font-size:32px;font-weight:700;color:{color};
                                  line-height:1">{score:.1f}</div>
                      <div style="font-size:10px;color:#9ca3af;margin-top:2px">
                        / 10</div>
                      <div style="height:4px;background:#f3f4f6;border-radius:2px;
                                  margin:10px 0 8px;overflow:hidden">
                        <div style="width:{bar_w}%;height:100%;background:{color};
                                    border-radius:2px"></div>
                      </div>
                      <div style="font-size:11px;color:#4b5563;line-height:1.4;
                                  text-align:left">{r["reason"][:100]}</div>
                    </div>'''
                cards_j += "</div>"
                judge_html = cards_j
        except Exception as e:
            judge_html = f'<p style="color:#dc2626">⚠️ LLM Judge lỗi: {e}</p>'
    elif use_judge and not gt.strip():
        judge_html = '<p style="color:#f59e0b">ℹ️ Cần nhập Ground Truth.</p>'
    elif use_judge and not api_key.strip():
        judge_html = '<p style="color:#f59e0b">ℹ️ Cần API key.</p>'
    else:
        judge_html = ('<p style="font-size:12px;color:#9ca3af">'
                      'Bật LLM Judge trong cài đặt nâng cao để xem điểm.</p>')

    total_ms = sum(timings.values())
    status = (f"✅ Hoàn thành — A1: {timings['A1']}ms · A2: {timings['A2']}ms · "
              f"B1: {timings['B1']}ms · B2: {timings['B2']}ms  |  "
              f"Tổng {total_ms}ms  |  VRAM: {vram_str()}")
    return cards, metrics_df, judge_html, status


# ════════════════════════════════════════════════════════════════════════════
# SAMPLE IMAGES từ test set
# ════════════════════════════════════════════════════════════════════════════
def get_sample_examples():
    """Pick 6 diverse test samples (1 per category, varied question types)."""
    if not TEST_JSON.exists():
        return []
    test = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    # filter chỉ items thuộc split test/ (có image trên đĩa)
    test_only = [d for d in test if d["image"].startswith("test/")
                 and (IMG_DIR / d["image"]).exists()]
    by_cat = {}
    for d in test_only:
        cat = d["image"].split("/")[1]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(d)

    samples = []
    for cat in sorted(by_cat.keys())[:6]:
        items = by_cat[cat]
        # Ưu tiên recognition > attribute > yes_no
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
_CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
.gr-block.gr-box, .gr-box { border-radius: 12px !important; }
.gr-button-primary {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  border: 0 !important;
}
.gr-button-primary:hover {
  filter: brightness(1.1);
}
#header-grad {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
  border-radius: 14px;
  padding: 24px 28px;
  margin-bottom: 14px;
  color: white;
}
.status-box textarea {
  background: #f9fafb !important; font-family: ui-monospace, monospace !important;
  font-size: 12px !important; color: #4b5563 !important; border: none !important;
}
"""


def build_app():
    samples = get_sample_examples()

    with gr.Blocks(title="Vietnamese VQA Demo", css=_CSS,
                   theme=gr.themes.Soft(primary_hue="indigo")) as demo:

        # ── HEADER ─────────────────────────────────────────────────────────
        gr.HTML(f"""
        <div id="header-grad">
          <div style="display:flex;align-items:center;justify-content:space-between;
                      flex-wrap:wrap;gap:12px">
            <div>
              <h1 style="font-size:24px;font-weight:700;margin:0 0 4px 0">
                🍜 Vietnamese Food VQA
              </h1>
              <p style="font-size:13px;opacity:0.92;margin:0">
                So sánh trực tiếp 4 cấu hình model trên cùng (ảnh + câu hỏi tiếng Việt)
              </p>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap">
              <span style="font-size:10px;padding:4px 10px;border-radius:12px;
                background:rgba(255,255,255,0.2);font-weight:600;backdrop-filter:blur(8px)">
                A1 · LSTM</span>
              <span style="font-size:10px;padding:4px 10px;border-radius:12px;
                background:rgba(255,255,255,0.2);font-weight:600;backdrop-filter:blur(8px)">
                A2 · Transformer</span>
              <span style="font-size:10px;padding:4px 10px;border-radius:12px;
                background:rgba(255,255,255,0.2);font-weight:600;backdrop-filter:blur(8px)">
                B1 · BLIP-2 Zero-shot</span>
              <span style="font-size:10px;padding:4px 10px;border-radius:12px;
                background:rgba(255,255,255,0.2);font-weight:600;backdrop-filter:blur(8px)">
                B2 · Qwen2-VL + LoRA</span>
            </div>
          </div>
          <div style="margin-top:10px;font-size:11px;opacity:0.85;
                      font-family:ui-monospace,monospace">
            Device: {DEVICE}  ·  VRAM: {vram_str()}  ·
            Vocab: {VOCAB_SIZE}  ·  Sample images: {len(samples)}
          </div>
        </div>
        """)

        with gr.Tabs():

            # ════════════════════════════════════════════════════════════
            # TAB 1 — Demo
            # ════════════════════════════════════════════════════════════
            with gr.Tab("🎯 Demo"):

                with gr.Row():
                    # LEFT
                    with gr.Column(scale=4):
                        image_in = gr.Image(label="📷 Ảnh", type="pil", height=300)
                        question_in = gr.Textbox(
                            label="❓ Câu hỏi (tiếng Việt)",
                            placeholder="VD: Đây là món ăn gì?",
                            lines=2,
                        )
                        gt_in = gr.Textbox(
                            label="✅ Ground Truth (tuỳ chọn — để tính metrics + LLM Judge)",
                            placeholder="VD: phở bò",
                            lines=1,
                        )

                        with gr.Accordion("⚙️ Cài đặt nâng cao", open=False):
                            judge_provider = gr.Radio(
                                ["openai", "anthropic"], value="openai",
                                label="LLM Judge provider",
                            )
                            api_key_in = gr.Textbox(
                                label="🔑 API Key",
                                placeholder="sk-... (OpenAI) hoặc sk-ant-... (Anthropic)",
                                type="password", lines=1,
                            )
                            use_judge_chk = gr.Checkbox(
                                label="Dùng LLM-as-a-Judge (cần API key + Ground Truth)",
                                value=False,
                            )

                        run_btn = gr.Button("▶  Chạy 4 models",
                                            variant="primary", size="lg")

                    # RIGHT
                    with gr.Column(scale=6):
                        gr.HTML('<div style="font-size:13px;font-weight:600;'
                                'color:#374151;margin-bottom:8px">📝 Predictions</div>')
                        preds_out = gr.HTML(
                            value=('<div style="color:#9ca3af;font-size:13px;'
                                   'padding:40px;text-align:center;border:2px dashed '
                                   '#e5e7eb;border-radius:12px">'
                                   'Upload ảnh + nhập câu hỏi rồi bấm Chạy</div>'))

                        status_out = gr.Textbox(
                            label="", interactive=False, show_label=False,
                            elem_classes="status-box", max_lines=1,
                        )

                # Sample examples
                if samples:
                    gr.HTML('<div style="font-size:13px;font-weight:600;'
                            'color:#374151;margin:14px 0 4px">💡 Ảnh mẫu '
                            '(click để load)</div>')
                    gr.Examples(
                        examples=samples,
                        inputs=[image_in, question_in, gt_in],
                        examples_per_page=6,
                        label="",
                    )

                # Metrics + Judge
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div style="font-size:13px;font-weight:600;'
                                'color:#374151;margin:14px 0 6px">'
                                '📊 Metrics (so với Ground Truth)</div>')
                        metrics_out = gr.Dataframe(
                            headers=["Model", "Prediction", "EM",
                                     "Token F1", "BLEU-1", "ROUGE-L"],
                            datatype=["str"] * 6,
                            interactive=False,
                        )

                gr.HTML('<div style="font-size:13px;font-weight:600;'
                        'color:#374151;margin:14px 0 6px">🤖 LLM-as-a-Judge '
                        '<span style="font-size:10px;color:#9ca3af;'
                        'font-weight:400">(0–10)</span></div>')
                judge_out = gr.HTML(
                    value='<p style="font-size:12px;color:#9ca3af">'
                          'Bật LLM Judge trong cài đặt để xem điểm.</p>')

            # ════════════════════════════════════════════════════════════
            # TAB 2 — Aggregate Results
            # ════════════════════════════════════════════════════════════
            with gr.Tab("📈 Kết quả train tổng quan"):
                if AGG_DF is None:
                    gr.HTML('<p style="color:#9ca3af;padding:20px">'
                            'Không tìm thấy <code>results/results_A.json</code> '
                            'hoặc <code>results/results_B.json</code>. '
                            'Chạy <code>vqa-model-a.ipynb</code> và '
                            '<code>vqa-model-b.ipynb</code> trước.</p>')
                else:
                    gr.HTML(f'''
                    <div style="padding:14px 0">
                      <p style="font-size:13px;color:#4b5563;line-height:1.6">
                        Kết quả từ <code>results/*.json</code> — đã train 4 model
                        trên VQA Food dataset
                        ({len(json.loads(TEST_JSON.read_text(encoding="utf-8")))
                          if TEST_JSON.exists() else "?"} test samples).
                      </p>
                    </div>''')

                    gr.Dataframe(value=AGG_DF, interactive=False,
                                 datatype=["str", "str"] + ["number"] * 6)

                    # Best per metric
                    best_html = "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:14px'>"
                    for metric_col in ["F1", "BERTScore", "SemSim"]:
                        test_only = AGG_DF[AGG_DF["Split"] == "Test"]
                        idx = test_only[metric_col].idxmax()
                        winner = test_only.loc[idx, "Model"]
                        val = test_only.loc[idx, metric_col]
                        _, color = MODEL_META.get(winner, ("", "#6b7280"))
                        best_html += f'''
                        <div style="border:1px solid #e5e7eb;border-radius:10px;
                                    padding:14px;background:#fff;text-align:center">
                          <div style="font-size:11px;color:#6b7280;margin-bottom:4px">
                            🏆 Best {metric_col} (Test)</div>
                          <div style="font-size:18px;font-weight:700;color:{color}">
                            {winner}</div>
                          <div style="font-size:14px;color:#374151;margin-top:4px">
                            {val:.4f}</div>
                        </div>'''
                    best_html += "</div>"
                    gr.HTML(best_html)

            # ════════════════════════════════════════════════════════════
            # TAB 3 — About
            # ════════════════════════════════════════════════════════════
            with gr.Tab("ℹ️ About"):
                gr.HTML(f"""
                <div style="padding:20px 4px;font-size:13px;color:#374151;line-height:1.7">
                  <h3 style="margin-top:0">Project: Vietnamese Food VQA</h3>
                  <p>Bài toán: cho ảnh món ăn Việt + câu hỏi tiếng Việt, model trả lời ngắn.</p>

                  <h4 style="margin-top:20px">4 cấu hình</h4>
                  <ul style="line-height:2">
                    <li><b style="color:#3b82f6">A1</b> — ResNet50 + PhoBERT + LSTM decoder
                        với Bahdanau attention trên 49 spatial regions.
                        <i>Train from scratch.</i></li>
                    <li><b style="color:#10b981">A2</b> — ResNet50 + PhoBERT + Transformer
                        decoder (3 layers, 8 heads) cross-attending fused memory.
                        <i>Train from scratch.</i></li>
                    <li><b style="color:#f59e0b">B1</b> — Salesforce/blip2-opt-2.7b
                        (zero-shot) + Helsinki-NLP MarianMT Vi↔En cho pipeline dịch máy.
                        <i>Không fine-tune.</i></li>
                    <li><b style="color:#8b5cf6">B2</b> — Qwen/Qwen2-VL-2B-Instruct
                        + LoRA (r=16, q/k/v/o_proj) trên 4-bit NF4 quantization.
                        <i>Fine-tune trên dataset tiếng Việt.</i></li>
                  </ul>

                  <h4 style="margin-top:20px">Files</h4>
                  <ul style="font-family:ui-monospace,monospace;font-size:12px">
                    <li><code>{CKPT_A1.name}</code> — A1 weights</li>
                    <li><code>{CKPT_A2.name}</code> — A2 weights</li>
                    <li><code>{CKPT_B2_LORA.relative_to(ROOT)}/</code> — B2 LoRA adapter</li>
                    <li><code>results/results_A.json</code>, <code>results_B.json</code>
                        — eval metrics</li>
                  </ul>

                  <h4 style="margin-top:20px">Yêu cầu GPU</h4>
                  <ul>
                    <li><b>A1 + A2 + B2</b>: ~3-4 GB VRAM (vừa với RTX 3060/4060)</li>
                    <li><b>B1</b>: thêm ~6 GB cho BLIP-2 fp16 → cần ≥10 GB VRAM</li>
                    <li>Models lazy-load — chỉ load khi user gọi inference lần đầu.</li>
                  </ul>
                </div>
                """)

        # Wire up
        run_btn.click(
            fn=run_demo,
            inputs=[image_in, question_in, gt_in,
                    judge_provider, api_key_in, use_judge_chk],
            outputs=[preds_out, metrics_out, judge_out, status_out],
        )
        question_in.submit(
            fn=run_demo,
            inputs=[image_in, question_in, gt_in,
                    judge_provider, api_key_in, use_judge_chk],
            outputs=[preds_out, metrics_out, judge_out, status_out],
        )

    return demo


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Vietnamese Food VQA - Demo App")
    print("=" * 60)
    print(f" Checkpoints in : {CKPT_DIR}")
    def _mark(p):
        return "[OK]     " if p.exists() else "[MISSING]"
    print(f"   A1 ckpt      : {_mark(CKPT_A1)}  {CKPT_A1}")
    print(f"   A2 ckpt      : {_mark(CKPT_A2)}  {CKPT_A2}")
    print(f"   B2 adapter   : {_mark(CKPT_B2_LORA)}  {CKPT_B2_LORA}")
    print(f" Results JSON   : {'[OK] found' if AGG_DF is not None else '[MISSING] not found'}")
    print(f" Models lazy-loaded on first inference call.")
    print("=" * 60 + "\n")

    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
