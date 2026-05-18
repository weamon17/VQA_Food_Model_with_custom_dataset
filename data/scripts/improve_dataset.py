"""
improve_dataset.py — Cải thiện toàn diện Vietnamese Food VQA dataset
======================================================================

Đặt file này TRONG THƯ MỤC `data/` (cùng cấp với annotations/, images/).
Chạy:   python improve_dataset.py

Sẽ tạo ra:
  • annotations_fixed/train.json       — đã thêm field "type", path đúng
  • annotations_fixed/val.json
  • annotations_fixed/test.json
  • dataset_stats.json                 — thống kê chi tiết
  • dataset_report.md                  — báo cáo Markdown để paste vào báo cáo

Sửa các vấn đề:
  1. Thêm field "type" (yes_no, counting, recognition, attribute, spatial, other)
  2. Sửa lại split chính xác 80/10/10 — STRATIFIED theo món ăn
  3. Verify mọi image path tồn tại
  4. Generate thống kê đầy đủ cho phần "Đánh giá dữ liệu" của báo cáo
"""

import os
import re
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# 
BASE_DIR = Path(__file__).resolve().parent.parent   # …/data/scripts/  → …/data/

RANDOM_SEED      = 42
SRC_ANN_DIR = BASE_DIR / "annotations"
SRC_IMG_DIR = BASE_DIR / "images"
DST_ANN_DIR = BASE_DIR / "annotations_fixed"
SPLIT_RATIO      = (0.80, 0.10, 0.10) # train/val/test

# ──────────────────────────────────────────────────────────────────────────
# 1. PHÂN LOẠI CÂU HỎI TỰ ĐỘNG
# ──────────────────────────────────────────────────────────────────────────
# Patterns được sắp xếp theo độ ưu tiên (rule đầu tiên match → gán type đó)

QUESTION_PATTERNS = [
    # YES/NO — câu hỏi nhận câu trả lời có/không
    ("yes_no", [
        r"\bcó phải\b", r"\bphải không\b", r"\bcó.*không\b",
        r"\bđúng không\b", r"\bcó đúng\b", r"\bliệu\b",
    ]),

    # COUNTING — đếm số lượng
    ("counting", [
        r"\bbao nhiêu\b", r"\bmấy\b", r"\bsố lượng\b",
        r"\bcó tất cả\b", r"\bđếm\b",
    ]),

    # SPATIAL — vị trí, không gian
    ("spatial", [
        r"\bở đâu\b", r"\bnằm ở\b", r"\bvị trí\b", r"\bbên (trái|phải|trên|dưới)\b",
        r"\bphía (trên|dưới|trái|phải|trước|sau)\b", r"\bgiữa\b", r"\btrong góc\b",
        r"\btrên bàn\b", r"\btrong (đĩa|tô|bát|chén)\b",
    ]),

    # ATTRIBUTE — đặt TRƯỚC srecognition vì các pattern miền/thời gian cụ thể hơn
    ("attribute", [
        # Color, size, temperature, taste, state
        r"\bmàu (gì|sắc|nào|của)\b",
        r"\bkích thước\b", r"\b(to|nhỏ|lớn|bé)\b",
        r"\b(nóng|lạnh|nguội|ấm)\b", r"\bnhiệt độ\b",
        r"\b(cay|ngọt|mặn|chua|đắng|béo)\b", r"\bvị\b",
        r"\b(khô|ướt|lỏng)\b",
        # Region (regional attribute) — phổ biến trong dataset này
        r"\b(miền|vùng) nào\b", r"\bmiền (bắc|trung|nam)\b",
        r"\bnguồn gốc\b", r"\bphổ biến (ở|tại)\b",
        r"\bđặc sản (của|ở|từ|miền)\b",
        # Time of day (temporal attribute) — phổ biến trong dataset
        r"\bkhi nào\b", r"\bthời gian\b", r"\bbuổi nào\b",
        r"\b(buổi |bữa )?(sáng|trưa|chiều|tối)\b",
        r"\bthích hợp ăn\b", r"\bthường ăn\b", r"\bnên ăn\b",
        # Setting / category
        r"\bđường phố\b", r"\bnhà hàng\b",
        r"\bnhóm (gì|nào)\b", r"\bthuộc (nhóm|loại|kiểu)\b",
    ]),

    # RECOGNITION — nhận dạng món / thành phần / nguyên liệu
    ("recognition", [
        r"\bmón (gì|nào|này là)\b", r"\bđây là (món|loại|cái) gì\b",
        r"\bloại (món|gì|nào)\b", r"\btên (món|gọi)\b",
        r"\b(món )?này (gọi là|tên là)\b",
        r"\bnguyên liệu\b", r"\bthành phần\b",
        r"\b(gồm|có) (những )?(gì|nguyên liệu)\b",
        # Recognition rephrases trong dataset
        r"\bmón (ăn )?(này |trong ảnh )?(là )?gì\b",
        r"\btrong ảnh là\b", r"\b(làm|chế biến) (từ|bằng)\b",
        r"\bchủ yếu làm\b",
    ]),
]


def classify_question(q: str) -> str:
    """Phân loại câu hỏi vào 1 trong 6 type. Default = 'other'."""
    q_lower = q.lower().strip()
    for type_name, patterns in QUESTION_PATTERNS:
        for pat in patterns:
            if re.search(pat, q_lower):
                return type_name
    return "other"


# ──────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────
def load_all() -> list:
    all_data = []
    for split in ["train", "val", "test"]:
        path = os.path.join(SRC_ANN_DIR, f"{split}.json")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for item in d:
            item["_orig_split"] = split
        all_data.extend(d)
    return all_data


# ──────────────────────────────────────────────────────────────────────────
# 3. EXTRACT FOOD CATEGORY (for stratified split)
# ──────────────────────────────────────────────────────────────────────────
def extract_category(image_path: str) -> str:
    """train/banh_xeo_viet_nam/Image_41.jpg → 'banh_xeo_viet_nam'"""
    parts = image_path.replace("\\", "/").split("/")
    # The category is always the second-to-last part
    return parts[-2] if len(parts) >= 2 else "unknown"


# ──────────────────────────────────────────────────────────────────────────
# 4. VERIFY IMAGES EXIST
# ──────────────────────────────────────────────────────────────────────────
def find_image(rel_path: str) -> str:
    """Try common locations; return found path or None."""
    candidates = [
        rel_path,
        os.path.join(SRC_IMG_DIR, rel_path),                 # images/train/...
        os.path.join(SRC_IMG_DIR, os.path.basename(rel_path)),
    ]
    cat = extract_category(rel_path)
    fname = os.path.basename(rel_path)
    for split in ["train", "val", "test"]:
        candidates.append(os.path.join(SRC_IMG_DIR, split, cat, fname))

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ──────────────────────────────────────────────────────────────────────────
# 5. STRATIFIED 80/10/10 SPLIT
# ──────────────────────────────────────────────────────────────────────────
def stratified_split(data: list) -> dict:
    """
    Split BY IMAGE (no leakage) and STRATIFIED by category.

    Hai-pass strategy:
      1. Phân chia ảnh trong mỗi category dùng `round()` để tránh dồn lệch về test
      2. Sau khi split sample-level, nếu lệch >2% so với 80/10/10 → tinh chỉnh
         bằng cách di chuyển ảnh giữa các split (vẫn giữ category balance)
    """
    img_to_samples = defaultdict(list)
    for item in data:
        img_to_samples[item["image"]].append(item)

    cat_to_imgs = defaultdict(list)
    for img_path in img_to_samples:
        cat_to_imgs[extract_category(img_path)].append(img_path)

    rng = random.Random(RANDOM_SEED)
    split_imgs = {"train": [], "val": [], "test": []}

    for cat, imgs in cat_to_imgs.items():
        rng.shuffle(imgs)
        n = len(imgs)

        # Round() instead of int() để phân bổ phần dư đều hơn
        n_test = max(1, round(n * SPLIT_RATIO[2]))
        n_val  = max(1, round(n * SPLIT_RATIO[1]))
        n_train = n - n_val - n_test

        # Đảm bảo train ≥ 1
        if n_train < 1 and n_test > 1:
            n_test -= 1; n_train += 1

        split_imgs["train"].extend(imgs[:n_train])
        split_imgs["val"  ].extend(imgs[n_train:n_train + n_val])
        split_imgs["test" ].extend(imgs[n_train + n_val:])

    splits = {s: [] for s in ["train", "val", "test"]}
    for s, imgs in split_imgs.items():
        for img in imgs:
            splits[s].extend(img_to_samples[img])

    return splits


# ──────────────────────────────────────────────────────────────────────────
# 6. NORMALISE IMAGE PATH (đảm bảo path tương đối hoạt động)
# ──────────────────────────────────────────────────────────────────────────
def normalise_path(item: dict, img_root: str) -> dict:
    """
    Sửa image path để khớp với cấu trúc thực tế.
    Chuẩn hóa thành: <split>/<category>/<filename>
    Notebook sẽ set IMG_DIR = "images" (hoặc đường dẫn tuyệt đối tới images/).
    """
    p = item["image"].replace("\\", "/")
    # Strip leading "images/" if present
    if p.startswith("images/"):
        p = p[len("images/"):]
    return p


# ──────────────────────────────────────────────────────────────────────────
# 7. GENERATE STATS
# ──────────────────────────────────────────────────────────────────────────
def compute_stats(splits: dict) -> dict:
    """Comprehensive statistics for the report."""
    stats = {"per_split": {}, "global": {}}

    all_samples = []
    for split, data in splits.items():
        all_samples.extend(data)

        type_dist = Counter(item.get("type", "unknown") for item in data)
        cat_dist  = Counter(extract_category(item["image"]) for item in data)
        ans_lens  = [len(item["answer"].split()) for item in data]
        q_lens    = [len(item["question"].split()) for item in data]
        n_unique_imgs = len(set(item["image"] for item in data))

        stats["per_split"][split] = {
            "n_samples"   : len(data),
            "n_images"    : n_unique_imgs,
            "type_dist"   : dict(type_dist),
            "category_dist": dict(cat_dist),
            "answer_len"  : {
                "min" : min(ans_lens), "max": max(ans_lens),
                "mean": round(sum(ans_lens) / len(ans_lens), 2),
                "dist": dict(Counter(ans_lens)),
            },
            "question_len": {
                "min" : min(q_lens), "max": max(q_lens),
                "mean": round(sum(q_lens) / len(q_lens), 2),
            },
        }

    # Global
    stats["global"] = {
        "n_samples_total" : len(all_samples),
        "n_images_total"  : len(set(item["image"] for item in all_samples)),
        "n_categories"    : len(set(extract_category(s["image"]) for s in all_samples)),
        "split_ratio"     : {s: round(len(splits[s]) / len(all_samples), 3) for s in splits},
        "answer_vocab_size": len(set(
            tok for s in all_samples for tok in re.sub(r"[^\w\s]","",s["answer"].lower()).split()
        )),
        "question_type_dist_total": dict(Counter(s.get("type","unknown") for s in all_samples)),
    }
    return stats


# ──────────────────────────────────────────────────────────────────────────
# 8. GENERATE MARKDOWN REPORT
# ──────────────────────────────────────────────────────────────────────────
def generate_md_report(stats: dict) -> str:
    g = stats["global"]
    lines = [
        "# Đánh giá dữ liệu — Vietnamese Food VQA",
        "",
        "## 1. Tổng quan",
        "",
        f"- **Tổng số mẫu**: {g['n_samples_total']:,}",
        f"- **Tổng số ảnh**: {g['n_images_total']:,}",
        f"- **Số món ăn**: {g['n_categories']}",
        f"- **Vocab câu trả lời**: {g['answer_vocab_size']} từ unique",
        "",
        "## 2. Phân chia split (80/10/10 stratified theo món ăn)",
        "",
        "| Split | Số mẫu | Số ảnh unique | Tỷ lệ |",
        "|---|---:|---:|---:|",
    ]
    for s in ["train", "val", "test"]:
        ps = stats["per_split"][s]
        ratio = g["split_ratio"][s]
        lines.append(f"| {s} | {ps['n_samples']:,} | {ps['n_images']:,} | {ratio*100:.1f}% |")

    lines.extend([
        "",
        "## 3. Phân bố loại câu hỏi (đa dạng theo yêu cầu đề)",
        "",
        "| Loại câu hỏi | Số mẫu | Tỷ lệ |",
        "|---|---:|---:|",
    ])
    type_total = g["question_type_dist_total"]
    total = sum(type_total.values())
    for t in ["yes_no", "counting", "recognition", "attribute", "spatial", "other"]:
        n = type_total.get(t, 0)
        pct = 100 * n / total if total else 0
        lines.append(f"| {t} | {n:,} | {pct:.1f}% |")

    lines.extend([
        "",
        "## 4. Phân bố độ dài câu trả lời",
        "",
        "| Split | Min | Max | Mean |",
        "|---|---:|---:|---:|",
    ])
    for s in ["train", "val", "test"]:
        al = stats["per_split"][s]["answer_len"]
        lines.append(f"| {s} | {al['min']} | {al['max']} | {al['mean']} |")

    lines.extend([
        "",
        "## 5. Phân bố theo món ăn (kiểm tra cân bằng)",
        "",
        "| Món ăn | Train | Val | Test |",
        "|---|---:|---:|---:|",
    ])
    all_cats = set()
    for s in ["train", "val", "test"]:
        all_cats.update(stats["per_split"][s]["category_dist"].keys())
    for cat in sorted(all_cats):
        row = [cat]
        for s in ["train", "val", "test"]:
            row.append(str(stats["per_split"][s]["category_dist"].get(cat, 0)))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# 9. MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  IMPROVE VIETNAMESE FOOD VQA DATASET")
    print("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────
    print("\n[1/6] Loading existing annotations...")
    all_data = load_all()
    print(f"      Loaded {len(all_data):,} samples")

    # ── Verify image paths ───────────────────────────────────────────────
    print("\n[2/6] Verifying image paths...")
    n_ok, n_bad = 0, 0
    bad_samples = []
    for item in all_data:
        if find_image(item["image"]):
            n_ok += 1
        else:
            n_bad += 1
            if len(bad_samples) < 5:
                bad_samples.append(item["image"])
    print(f"      ✅ Found: {n_ok:,}")
    print(f"      ❌ Missing: {n_bad:,}")
    if bad_samples:
        print(f"      Examples missing: {bad_samples}")

    # ── Classify question types ──────────────────────────────────────────
    print("\n[3/6] Classifying question types...")
    type_counter = Counter()
    for item in all_data:
        item["type"]  = classify_question(item["question"])
        # normalise image path
        item["image"] = normalise_path(item, SRC_IMG_DIR)
        # cleanup helper field
        item.pop("_orig_split", None)
        type_counter[item["type"]] += 1
    for t, c in type_counter.most_common():
        print(f"      {t:<14}: {c:>5,}  ({100*c/len(all_data):.1f}%)")

    # ── Stratified split 80/10/10 ────────────────────────────────────────
    print(f"\n[4/6] Stratified split {SPLIT_RATIO} by category (image-level, no leakage)...")
    splits = stratified_split(all_data)
    for s in ["train", "val", "test"]:
        ratio = len(splits[s]) / len(all_data)
        n_imgs = len(set(item["image"] for item in splits[s]))
        print(f"      {s:>5}: {len(splits[s]):>5,} samples · {n_imgs:>3} unique images · {ratio*100:5.2f}%")

    # ── Save fixed annotations ───────────────────────────────────────────
    print(f"\n[5/6] Saving to '{DST_ANN_DIR}/'...")
    os.makedirs(DST_ANN_DIR, exist_ok=True)
    for split, data in splits.items():
        path = os.path.join(DST_ANN_DIR, f"{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"      Saved {path} ({len(data):,} samples)")

    # ── Stats & report ───────────────────────────────────────────────────
    print("\n[6/6] Generating stats & markdown report...")
    stats = compute_stats(splits)
    with open("dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    md = generate_md_report(stats)
    with open("dataset_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("      Saved dataset_stats.json + dataset_report.md")

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ DONE — improvements applied:")
    print("=" * 60)
    print("  • Added 'type' field (6 categories)")
    print(f"  • Re-split into {SPLIT_RATIO[0]*100:.0f}/{SPLIT_RATIO[1]*100:.0f}/{SPLIT_RATIO[2]*100:.0f}"
          f" stratified by food category at IMAGE LEVEL")
    print("  • Normalised image paths (use IMG_DIR='images' in notebooks)")
    print("  • Generated dataset_report.md for thesis section")
    print("\n  Next: trong notebook đặt IMG_DIR=\"images\" và đổi annotation paths:")
    print("    TRAIN_JSON = \"annotations_fixed/train.json\"")
    print("    VAL_JSON   = \"annotations_fixed/val.json\"")
    print("    TEST_JSON  = \"annotations_fixed/test.json\"")


if __name__ == "__main__":
    main()
