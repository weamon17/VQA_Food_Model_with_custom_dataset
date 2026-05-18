"""
visualize_dataset.py — Trực quan hóa toàn diện Vietnamese Food VQA dataset
==========================================================================

Sinh ra TẤT CẢ biểu đồ cần thiết cho phần "Đánh giá dữ liệu" trong báo cáo.

Đặt file này TRONG THƯ MỤC `data/`. Chạy:
    pip install matplotlib seaborn wordcloud pillow
    python visualize_dataset.py

Tạo ra các file PNG (300 DPI, sẵn sàng để paste vào Word):
    figures/01_overview_dashboard.png        — 1 hình tổng quát (DÙNG TRONG BÁO CÁO)
    figures/02_split_distribution.png        — phân chia train/val/test
    figures/03_question_type.png             — phân bố loại câu hỏi
    figures/04_category_distribution.png     — phân bố theo món ăn
    figures/05_answer_length.png             — độ dài câu trả lời
    figures/06_question_length.png           — độ dài câu hỏi
    figures/07_top_answers.png               — top câu trả lời phổ biến
    figures/08_wordcloud_questions.png       — wordcloud câu hỏi
    figures/09_wordcloud_answers.png         — wordcloud câu trả lời
    figures/10_sample_images_grid.png        — grid ảnh mẫu từ 10 món
    figures/11_heatmap_type_x_category.png   — heatmap loại câu hỏi × món
"""

import os
import re
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Optional: wordcloud (graceful fallback nếu chưa cài)
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("[INFO] wordcloud chưa được cài. Cài bằng: pip install wordcloud")

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # …/data/scripts/  → …/data/

ANN_DIR    = BASE_DIR / "annotations"     # đổi thành annotations_fixed nếu muốn
IMG_DIR    = BASE_DIR / "images"
OUT_DIR    = BASE_DIR / "figures"

# Đặt font hỗ trợ tiếng Việt — thử theo thứ tự
VN_FONTS = ["DejaVu Sans", "Arial Unicode MS", "Tahoma", "Segoe UI"]
for f in VN_FONTS:
    try:
        plt.rcParams["font.family"] = f
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 100
plt.rcParams["savefig.dpi"]        = 300
plt.rcParams["savefig.bbox"]       = "tight"
plt.rcParams["savefig.facecolor"]  = "white"

# ──────────────────────────────────────────────────────────────────────────
# COLOR PALETTE — chuyên nghiệp, phù hợp báo cáo học thuật
# ──────────────────────────────────────────────────────────────────────────
PALETTE = {
    "train"     : "#2E86AB",
    "val"       : "#A23B72",
    "test"      : "#F18F01",
    "yes_no"    : "#264653",
    "counting"  : "#2A9D8F",
    "recognition": "#E9C46A",
    "attribute" : "#F4A261",
    "spatial"   : "#E76F51",
    "other"     : "#6C757D",
}

CAT_COLORS = ["#264653","#2A9D8F","#8AB17D","#E9C46A","#EFB366",
              "#F4A261","#EE8959","#E76F51","#A23B72","#6A0572","#3D348B"]

os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────
def load_data():
    data = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(ANN_DIR, f"{split}.json")
        with open(path, encoding="utf-8") as f:
            data[split] = json.load(f)
    return data


def extract_cat(image_path: str) -> str:
    parts = image_path.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else "unknown"


def tokenize(text: str) -> list:
    return re.sub(r"[^\w\s]", "", text.lower().strip()).split()


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 1 — OVERVIEW DASHBOARD (HÌNH CHỦ ĐẠO CHO BÁO CÁO)
# ──────────────────────────────────────────────────────────────────────────
def plot_overview(data: dict):
    """Một hình duy nhất tóm tắt toàn bộ dataset — cho báo cáo & slide."""
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4,
                  left=0.06, right=0.97, top=0.92, bottom=0.06)

    all_items = sum(data.values(), [])

    # ── HEADER STATS (top row, full width) ──────────────────────────────
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.axis("off")
    n_total = len(all_items)
    n_imgs  = len(set(item["image"] for item in all_items))
    n_cats  = len(set(extract_cat(item["image"]) for item in all_items))
    avg_qlen = np.mean([len(item["question"].split()) for item in all_items])
    avg_alen = np.mean([len(item["answer"].split())   for item in all_items])
    vocab_q = len(set(t for item in all_items for t in tokenize(item["question"])))
    vocab_a = len(set(t for item in all_items for t in tokenize(item["answer"])))

    stats = [
        (f"{n_total:,}",     "Tổng số mẫu"),
        (f"{n_imgs:,}",      "Ảnh unique"),
        (f"{n_cats}",        "Món ăn"),
        (f"{avg_qlen:.1f}",  "Độ dài Q (TB)"),
        (f"{avg_alen:.1f}",  "Độ dài A (TB)"),
        (f"{vocab_q:,}",     "Vocab Q"),
        (f"{vocab_a}",       "Vocab A"),
    ]
    for i, (num, lbl) in enumerate(stats):
        x = 0.04 + i * (0.94 / len(stats))
        ax_h.text(x, 0.7, num, fontsize=22, fontweight="bold",
                  ha="left", color="#264653", transform=ax_h.transAxes)
        ax_h.text(x, 0.15, lbl, fontsize=10, ha="left",
                  color="#6c757d", transform=ax_h.transAxes)
    ax_h.set_title("Vietnamese Food VQA — Tổng quan Dataset",
                   fontsize=15, fontweight="bold", pad=8, loc="left")

    # ── (Row 1, Col 0–1): Split distribution ─────────────────────────────
    ax1 = fig.add_subplot(gs[1, :2])
    counts = [len(data[s]) for s in ["train", "val", "test"]]
    colors = [PALETTE[s] for s in ["train", "val", "test"]]
    bars = ax1.barh(["Train", "Val", "Test"], counts, color=colors, alpha=0.9)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                 f"{c:,} ({100*c/n_total:.1f}%)", va="center", fontsize=10,
                 fontweight="bold")
    ax1.set_xlim(0, max(counts) * 1.18)
    ax1.set_title("Phân chia Train / Val / Test", fontsize=12, fontweight="bold", pad=6)
    ax1.set_xlabel("Số mẫu", fontsize=9)
    ax1.spines[["top","right"]].set_visible(False)
    ax1.grid(axis="x", alpha=0.3)

    # ── (Row 1, Col 2–3): Question type pie ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 2:])
    type_count = Counter(item.get("type","unknown") for item in all_items)
    types_order = ["yes_no","counting","recognition","attribute","spatial","other"]
    types_order = [t for t in types_order if type_count.get(t, 0) > 0]
    sizes  = [type_count[t] for t in types_order]
    colors_t = [PALETTE[t] for t in types_order]
    wedges, texts, auto = ax2.pie(
        sizes, labels=types_order, colors=colors_t,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9}, pctdistance=0.78,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for a in auto:
        a.set_color("white"); a.set_fontweight("bold"); a.set_fontsize(9)
    ax2.set_title("Phân bố Loại câu hỏi", fontsize=12, fontweight="bold", pad=6)

    # ── (Row 2, Col 0–2): Category distribution ─────────────────────────
    ax3 = fig.add_subplot(gs[2, :3])
    cat_count = Counter(extract_cat(item["image"]) for item in all_items)
    cats   = sorted(cat_count.keys(), key=lambda x: -cat_count[x])
    counts = [cat_count[c] for c in cats]
    # Hiển thị tên món đẹp hơn
    cats_pretty = [c.replace("_viet_nam","").replace("_"," ").title() for c in cats]
    bars = ax3.bar(cats_pretty, counts, color=CAT_COLORS[:len(cats)], alpha=0.9)
    for bar, c in zip(bars, counts):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                 f"{c:,}", ha="center", va="bottom", fontsize=8)
    ax3.set_title("Phân bố theo Món ăn", fontsize=12, fontweight="bold", pad=6)
    ax3.set_ylabel("Số mẫu", fontsize=9)
    ax3.tick_params(axis="x", rotation=30, labelsize=8.5)
    ax3.spines[["top","right"]].set_visible(False)
    ax3.grid(axis="y", alpha=0.3)
    plt.setp(ax3.get_xticklabels(), ha="right")

    # ── (Row 2, Col 3): Answer length distribution ──────────────────────
    ax4 = fig.add_subplot(gs[2, 3])
    ans_lens = [len(item["answer"].split()) for item in all_items]
    bins = np.arange(0.5, max(ans_lens)+1.5, 1)
    ax4.hist(ans_lens, bins=bins, color="#2A9D8F", alpha=0.85, edgecolor="white")
    ax4.set_title("Độ dài Câu trả lời", fontsize=12, fontweight="bold", pad=6)
    ax4.set_xlabel("Số từ", fontsize=9)
    ax4.set_ylabel("Số mẫu", fontsize=9)
    ax4.spines[["top","right"]].set_visible(False)
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_xticks(range(1, max(ans_lens)+1))

    out = os.path.join(OUT_DIR, "01_overview_dashboard.png")
    plt.savefig(out)
    plt.close()
    print(f"  ✅ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 2 — SPLIT DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────
def plot_split(data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Donut
    counts = [len(data[s]) for s in ["train", "val", "test"]]
    labels = ["Train", "Val", "Test"]
    colors = [PALETTE[s] for s in ["train","val","test"]]
    wedges, _, auto = axes[0].pie(
        counts, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops={"width": 0.35, "edgecolor":"white", "linewidth":2},
        textprops={"fontsize": 11},
    )
    for a in auto: a.set_fontweight("bold")
    n_total = sum(counts)
    axes[0].text(0, 0, f"{n_total:,}\nmẫu", ha="center", va="center",
                 fontsize=14, fontweight="bold")
    axes[0].set_title("Tỷ lệ split", fontsize=12, fontweight="bold")

    # Bar w/ counts + unique images
    n_imgs = [len(set(item["image"] for item in data[s])) for s in ["train","val","test"]]
    x = np.arange(3); w = 0.35
    axes[1].bar(x - w/2, counts, w, label="Số mẫu",  color=colors, alpha=0.9)
    axes[1].bar(x + w/2, [v*30 for v in n_imgs], w, label="Số ảnh × 30 (scale)",
                color=colors, alpha=0.45)
    for i, (cnt, img) in enumerate(zip(counts, n_imgs)):
        axes[1].text(i - w/2, cnt + max(counts)*0.01, f"{cnt:,}",
                     ha="center", fontsize=9, fontweight="bold")
        axes[1].text(i + w/2, img*30 + max(counts)*0.01, f"{img}",
                     ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_title("Số mẫu & ảnh unique", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Số lượng")
    axes[1].legend(fontsize=9, frameon=False)
    axes[1].spines[["top","right"]].set_visible(False)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Phân chia Dataset 80/10/10 (stratified)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "02_split_distribution.png")
    plt.savefig(out); plt.close()
    print(f"  ✅ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 3 — QUESTION TYPE BAR + STACKED BY SPLIT
# ──────────────────────────────────────────────────────────────────────────
def plot_question_type(data: dict):
    types_order = ["yes_no","counting","recognition","attribute","spatial","other"]

    # Stacked by split
    stacks = {t: [sum(1 for item in data[s] if item.get("type")==t)
                  for s in ["train","val","test"]] for t in types_order}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: total bar
    totals = [sum(stacks[t]) for t in types_order]
    bars = axes[0].bar(types_order, totals,
                       color=[PALETTE[t] for t in types_order], alpha=0.9)
    total_all = sum(totals)
    for bar, c in zip(bars, totals):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height() + total_all*0.005,
                     f"{c:,}\n({100*c/total_all:.1f}%)",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Phân bố tổng theo loại câu hỏi", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Số mẫu")
    axes[0].tick_params(axis="x", rotation=15, labelsize=10)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[0].grid(axis="y", alpha=0.3)

    # Right: stacked by split
    x = np.arange(3)
    bottom = np.zeros(3)
    for t in types_order:
        vals = np.array(stacks[t])
        axes[1].bar(x, vals, bottom=bottom, label=t, color=PALETTE[t], alpha=0.9)
        bottom += vals
    axes[1].set_xticks(x); axes[1].set_xticklabels(["Train","Val","Test"])
    axes[1].set_title("Phân bố theo Split", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Số mẫu")
    axes[1].legend(loc="upper right", fontsize=9, frameon=False)
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "03_question_type.png")
    plt.savefig(out); plt.close()
    print(f"  ✅ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 4 — CATEGORY × SPLIT
# ──────────────────────────────────────────────────────────────────────────
def plot_category(data: dict):
    all_items = sum(data.values(), [])
    cat_count = Counter(extract_cat(item["image"]) for item in all_items)
    cats = sorted(cat_count.keys(), key=lambda x: -cat_count[x])

    fig, ax = plt.subplots(figsize=(13, 5.5))

    width = 0.27
    x = np.arange(len(cats))
    splits = ["train", "val", "test"]
    for i, split in enumerate(splits):
        counts = [sum(1 for item in data[split] if extract_cat(item["image"])==c) for c in cats]
        ax.bar(x + (i-1)*width, counts, width,
               label=split.capitalize(), color=PALETTE[split], alpha=0.9)

    cats_pretty = [c.replace("_viet_nam","").replace("_"," ").title() for c in cats]
    ax.set_xticks(x); ax.set_xticklabels(cats_pretty, rotation=25, ha="right")
    ax.set_ylabel("Số mẫu")
    ax.set_title("Phân bố theo Món ăn × Split", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "04_category_distribution.png")
    plt.savefig(out); plt.close()
    print(f"  ✅ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 5 — ANSWER LENGTH
# ──────────────────────────────────────────────────────────────────────────
def plot_answer_length(data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Histogram per split
    for split in ["train", "val", "test"]:
        lens = [len(item["answer"].split()) for item in data[split]]
        axes[0].hist(lens, bins=np.arange(0.5, max(lens)+1.5, 1),
                     alpha=0.55, label=split, color=PALETTE[split], edgecolor="white")
    axes[0].set_xlabel("Số từ trong câu trả lời")
    axes[0].set_ylabel("Số mẫu")
    axes[0].set_title("Phân bố độ dài câu trả lời theo split",
                      fontsize=12, fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[0].grid(axis="y", alpha=0.3)

    # Boxplot by question type
    types_order = ["yes_no","counting","recognition","attribute","spatial"]
    all_items = sum(data.values(), [])
    data_box = [[len(item["answer"].split()) for item in all_items if item.get("type")==t]
                for t in types_order]
    data_box = [d for d in data_box if d]
    types_avail = [t for t, d in zip(types_order, data_box) if d]

    bp = axes[1].boxplot(data_box, labels=types_avail, patch_artist=True,
                          showmeans=True, meanprops={"marker":"o","markerfacecolor":"white",
                                                      "markeredgecolor":"black","markersize":6})
    for patch, t in zip(bp["boxes"], types_avail):
        patch.set_facecolor(PALETTE[t]); patch.set_alpha(0.8)
    axes[1].set_ylabel("Số từ")
    axes[1].set_title("Độ dài câu trả lời theo loại câu hỏi",
                      fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].spines[["top","right"]].set_visible(False)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "05_answer_length.png")
    plt.savefig(out); plt.close()
    print(f"  ✅ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 6 — QUESTION LENGTH
# ──────────────────────────────────────────────────────────────────────────
def plot_question_length(data: dict):
    fig, ax = plt.subplots(figsize=(11, 5))

    for split in ["train", "val", "test"]:
        lens = [len(item["question"].split()) for item in data[split]]
        ax.hist(lens, bins=np.arange(min(lens)-0.5, max(lens)+1.5, 1),
                alpha=0.55, label=split, color=PALETTE[split], edgecolor="white")

    ax.set_xlabel("Số từ trong câu hỏi", fontsize=11)
    ax.set_ylabel("Số mẫu", fontsize=11)
    ax.set_title("Phân bố độ dài Câu hỏi", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Stats text
    all_items = sum(data.values(), [])
    qlens = [len(item["question"].split()) for item in all_items]
    stats_txt = (f"min: {min(qlens)}   max: {max(qlens)}\n"
                 f"mean: {np.mean(qlens):.1f}   median: {int(np.median(qlens))}")
    ax.text(0.98, 0.98, stats_txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=10, bbox=dict(boxstyle="round,pad=0.5",
                                    facecolor="white", edgecolor="#ccc"))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "06_question_length.png")
    plt.savefig(out); plt.close()
    print(f"  ✅ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 7 — TOP 20 ANSWERS
# ──────────────────────────────────────────────────────────────────────────
def plot_top_answers(data: dict):
    all_items = sum(data.values(), [])
    ans_count = Counter(item["answer"] for item in all_items)
    top = ans_count.most_common(20)
    labels, counts = zip(*top)

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(range(len(labels)), counts,
                    color=plt.cm.viridis(np.linspace(0.15, 0.85, len(labels))),
                    alpha=0.9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    total = sum(ans_count.values())
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.005, bar.get_y()+bar.get_height()/2,
                f"{c:,} ({100*c/total:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("Số lần xuất hiện", fontsize=11)
    ax.set_title("Top 20 câu trả lời phổ biến", fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(counts)*1.15)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "07_top_answers.png")
    plt.savefig(out); plt.close()
    print(f"  ✅ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 8, 9 — WORDCLOUDS
# ──────────────────────────────────────────────────────────────────────────
def plot_wordclouds(data: dict):
    if not HAS_WORDCLOUD:
        print("  ⚠️  Skipped wordclouds (wordcloud not installed)")
        return

    all_items = sum(data.values(), [])

    # Stopwords tiếng Việt cơ bản
    STOP = {"là","của","và","có","không","đây","này","bạn","tôi","mình",
            "vậy","nhé","nhỉ","cho","hỏi","biết","theo","món","ăn","ở",
            "thì","mà","để","vào","ra","lên","xuống","được","đã","cũng"}

    # Question wordcloud
    qtext = " ".join(item["question"] for item in all_items)
    qtoks = [t for t in tokenize(qtext) if t not in STOP and len(t) > 1]
    qfreq = Counter(qtoks)

    wc_q = WordCloud(width=1200, height=600, background_color="white",
                      colormap="viridis", max_words=100,
                      relative_scaling=0.5).generate_from_frequencies(qfreq)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(wc_q, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Wordcloud — Câu hỏi", fontsize=14, fontweight="bold", pad=10)
    out = os.path.join(OUT_DIR, "08_wordcloud_questions.png")
    plt.savefig(out); plt.close()
    print(f"  ✅ {out}")

    # Answer wordcloud
    atoks = [t for t in tokenize(" ".join(item["answer"] for item in all_items))
             if len(t) > 0]
    afreq = Counter(atoks)
    wc_a = WordCloud(width=1200, height=600, background_color="white",
                      colormap="plasma", max_words=50,
                      relative_scaling=0.5).generate_from_frequencies(afreq)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.imshow(wc_a, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Wordcloud — Câu trả lời", fontsize=14, fontweight="bold", pad=10)
    out = os.path.join(OUT_DIR, "09_wordcloud_answers.png")
    plt.savefig(out); plt.close()
    print(f"  ✅ {out}")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 10 — SAMPLE IMAGES GRID (1 ảnh mẫu/món)
# ──────────────────────────────────────────────────────────────────────────
def plot_sample_grid(data: dict):
    all_items = sum(data.values(), [])
    # Lấy 1 ảnh đại diện mỗi món
    cat_to_img = {}
    for item in all_items:
        cat = extract_cat(item["image"])
        if cat not in cat_to_img:
            cat_to_img[cat] = item["image"]

    cats = sorted(cat_to_img.keys())
    n = len(cats)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    found_any = False
    for ax, cat in zip(axes, cats):
        img_path = cat_to_img[cat]
        # Try multiple paths
        for trial in [
            img_path,
            os.path.join(IMG_DIR, img_path),
            os.path.join(IMG_DIR, os.path.basename(img_path)),
        ]:
            if os.path.exists(trial):
                try:
                    img = Image.open(trial).convert("RGB")
                    ax.imshow(img)
                    found_any = True
                    break
                except Exception:
                    pass
        else:
            ax.text(0.5, 0.5, "(không tìm thấy ảnh)", ha="center", va="center",
                    fontsize=9, transform=ax.transAxes)

        pretty = cat.replace("_viet_nam","").replace("_"," ").title()
        ax.set_title(pretty, fontsize=10, fontweight="bold")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Ảnh mẫu — 1 ảnh đại diện mỗi món",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "10_sample_images_grid.png")
    plt.savefig(out); plt.close()

    if found_any:
        print(f"  ✅ {out}")
    else:
        print(f"  ⚠️  {out} (không load được ảnh — kiểm tra IMG_DIR)")


# ──────────────────────────────────────────────────────────────────────────
# FIGURE 11 — HEATMAP TYPE × CATEGORY
# ──────────────────────────────────────────────────────────────────────────
def plot_heatmap(data: dict):
    all_items = sum(data.values(), [])
    types_order = ["yes_no","counting","recognition","attribute","spatial","other"]
    types_avail = [t for t in types_order
                   if any(item.get("type")==t for item in all_items)]

    cat_count = Counter(extract_cat(item["image"]) for item in all_items)
    cats = sorted(cat_count.keys(), key=lambda x: -cat_count[x])

    matrix = np.zeros((len(types_avail), len(cats)), dtype=int)
    for item in all_items:
        t = item.get("type","unknown")
        c = extract_cat(item["image"])
        if t in types_avail:
            matrix[types_avail.index(t), cats.index(c)] += 1

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    cats_pretty = [c.replace("_viet_nam","").replace("_"," ").title() for c in cats]
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats_pretty, rotation=30, ha="right")
    ax.set_yticks(range(len(types_avail))); ax.set_yticklabels(types_avail)

    for i in range(len(types_avail)):
        for j in range(len(cats)):
            val = matrix[i, j]
            color = "white" if val > matrix.max()*0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Số mẫu", shrink=0.85)
    ax.set_title("Heatmap: Loại câu hỏi × Món ăn",
                 fontsize=13, fontweight="bold", pad=8)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "11_heatmap_type_x_category.png")
    plt.savefig(out); plt.close()
    print(f"  ✅ {out}")


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  VISUALIZE VIETNAMESE FOOD VQA DATASET")
    print("=" * 60)

    print(f"\nLoading data from '{ANN_DIR}/'...")
    data = load_data()
    for s in data:
        print(f"  {s}: {len(data[s]):,} samples")

    print(f"\nGenerating figures into '{OUT_DIR}/'...\n")
    plot_overview(data)
    plot_split(data)
    plot_question_type(data)
    plot_category(data)
    plot_answer_length(data)
    plot_question_length(data)
    plot_top_answers(data)
    plot_wordclouds(data)
    plot_sample_grid(data)
    plot_heatmap(data)

    print("\n" + "=" * 60)
    print(f"  ✅ Done — {len(os.listdir(OUT_DIR))} files in {OUT_DIR}/")
    print("=" * 60)
    print("\nDÙNG TRONG BÁO CÁO:")
    print("  • Hình chủ đạo: 01_overview_dashboard.png")
    print("  • Phân tích split: 02_split_distribution.png")
    print("  • Phân tích đa dạng: 03_question_type.png + 11_heatmap...")
    print("  • Phân tích nội dung: 07_top_answers.png + 08+09 wordclouds")
    print("  • Ảnh mẫu cho slide: 10_sample_images_grid.png")


if __name__ == "__main__":
    main()
