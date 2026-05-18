"""
augment_counting_spatial.py — Bổ sung câu hỏi counting và spatial
==================================================================

Thêm 2 loại câu hỏi còn thiếu vào dataset:
  • counting — đếm số lượng ("có bao nhiêu...", "mấy loại...")
  • spatial  — vị trí, không gian ("nằm ở đâu", "phía trên/dưới...")

Đặt file này TRONG THƯ MỤC `data/`. Chạy SAU `improve_dataset.py`:
    python improve_dataset.py        # tạo annotations_fixed/
    python augment_counting_spatial.py   # bổ sung counting + spatial

Tạo ra:
  • annotations_augmented/train.json
  • annotations_augmented/val.json
  • annotations_augmented/test.json
  • augment_report.md  — báo cáo

Lưu ý: câu hỏi/câu trả lời được sinh template-based dựa trên đặc điểm
điển hình của từng món ăn Việt — KHÔNG phải dựa trên ảnh cụ thể.
Điều này khớp với style của dataset gốc (cũng template-based theo món).
"""

import os
import re
import json
import random
import itertools
from collections import Counter, defaultdict
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')


# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent   # …/data/scripts/  → …/data/

RANDOM_SEED       = 42
SRC_DIR           = BASE_DIR / "annotations"      # input từ improve_dataset.py
DST_DIR           = BASE_DIR / "annotations_augmented"  # output

# Số câu COUNTING và SPATIAL cộng thêm cho MỖI ảnh
N_COUNTING_PER_IMAGE = 4
N_SPATIAL_PER_IMAGE  = 4

# ──────────────────────────────────────────────────────────────────────────
# KIẾN THỨC ẨM THỰC — facts cho từng món
# ──────────────────────────────────────────────────────────────────────────
# Mỗi entry: (question_template, answer)
# Templates dùng {prefix} sẽ được thay bằng paraphrase prefix random
#
# COUNTING — đáp án là số lượng thông thường của món đó

COUNTING_FACTS = {
    "pho_bo_viet_nam": [
        ("phở bò thường có mấy loại thịt", "ba loại"),
        ("bao nhiêu loại topping chính trong phở", "ba loại"),
        ("một bát phở thường có mấy loại rau ăn kèm", "bốn loại"),
        ("phở bò có mấy thành phần chính", "ba thành phần"),
        ("một tô phở thường có bao nhiêu lát thịt", "năm lát"),
        ("trong tô phở có mấy loại nguyên liệu", "năm loại"),
    ],
    "bun_bo_hue": [
        ("bún bò huế có mấy loại topping chính", "bốn loại"),
        ("bao nhiêu thành phần chính trong bún bò", "bốn thành phần"),
        ("bún bò huế có mấy loại thịt", "ba loại"),
        ("một tô bún bò có bao nhiêu loại nguyên liệu", "năm loại"),
        ("mấy lát giò trong tô bún bò huế", "hai lát"),
        ("có bao nhiêu loại rau ăn kèm bún bò", "ba loại"),
    ],
    "banh_mi_thit": [
        ("bánh mì thịt có mấy lớp nhân", "ba lớp"),
        ("bao nhiêu loại rau trong bánh mì thịt", "hai loại"),
        ("một ổ bánh mì có mấy loại thịt", "hai loại"),
        ("bánh mì thịt có bao nhiêu thành phần", "bốn thành phần"),
        ("mấy loại sốt thường có trong bánh mì thịt", "hai loại"),
        ("trong bánh mì có bao nhiêu lớp", "ba lớp"),
    ],
    "banh_mi_pate": [
        ("bánh mì pate có mấy lớp nhân chính", "ba lớp"),
        ("bao nhiêu loại nguyên liệu trong bánh mì pate", "bốn loại"),
        ("bánh mì pate có mấy thành phần", "bốn thành phần"),
        ("một ổ bánh mì pate có bao nhiêu lớp", "ba lớp"),
        ("mấy loại rau ăn kèm bánh mì pate", "hai loại"),
        ("bánh mì pate có bao nhiêu lớp pate", "một lớp"),
    ],
    "banh_xeo_viet_nam": [
        ("bánh xèo có mấy thành phần chính", "ba thành phần"),
        ("một chiếc bánh xèo có bao nhiêu loại nhân", "ba loại"),
        ("bánh xèo có mấy phần chính", "hai phần"),
        ("bao nhiêu loại rau ăn kèm bánh xèo", "năm loại"),
        ("một đĩa bánh xèo có mấy chiếc", "một chiếc"),
        ("trong bánh xèo có bao nhiêu loại nguyên liệu", "bốn loại"),
    ],
    "banh_chung": [
        ("bánh chưng có mấy lớp", "ba lớp"),
        ("một chiếc bánh chưng có bao nhiêu thành phần", "ba thành phần"),
        ("bánh chưng được gói bằng mấy lớp lá", "bốn lớp"),
        ("bánh chưng có hình mấy cạnh", "bốn cạnh"),
        ("nhân bánh chưng có mấy loại", "hai loại"),
        ("một chiếc bánh chưng vuông có mấy mặt", "sáu mặt"),
    ],
    "banh_bao": [
        ("bánh bao có mấy phần chính", "hai phần"),
        ("nhân bánh bao thường có bao nhiêu loại nguyên liệu", "ba loại"),
        ("bánh bao có mấy lớp", "hai lớp"),
        ("một chiếc bánh bao có bao nhiêu phần", "hai phần"),
        ("trong nhân bánh bao có mấy loại nhân", "ba loại"),
        ("bánh bao có hình mấy mặt", "tròn"),
    ],
    "banh_uot": [
        ("bánh ướt có mấy thành phần", "ba thành phần"),
        ("một đĩa bánh ướt có bao nhiêu lớp", "nhiều lớp"),
        ("bánh ướt thường ăn kèm mấy loại topping", "ba loại"),
        ("bao nhiêu loại nhân trong bánh ướt", "hai loại"),
        ("bánh ướt có mấy phần chính", "hai phần"),
        ("một đĩa bánh ướt có mấy loại nguyên liệu", "bốn loại"),
    ],
    "goi_cuon_viet_nam": [
        ("một cuốn gỏi cuốn có mấy loại nhân", "bốn loại"),
        ("gỏi cuốn có bao nhiêu thành phần chính", "bốn thành phần"),
        ("một cuốn có mấy lớp bánh tráng", "một lớp"),
        ("bao nhiêu loại rau trong gỏi cuốn", "ba loại"),
        ("gỏi cuốn có mấy loại thịt", "hai loại"),
        ("một cuốn có bao nhiêu loại nguyên liệu", "năm loại"),
    ],
    "com_tam_suon_bi_cha": [
        ("cơm tấm có mấy loại topping chính", "ba loại"),
        ("một đĩa cơm tấm có bao nhiêu thành phần", "bốn thành phần"),
        ("cơm tấm sườn bì chả có mấy loại topping", "ba loại"),
        ("bao nhiêu loại nguyên liệu trong cơm tấm", "năm loại"),
        ("một đĩa cơm tấm có mấy phần", "bốn phần"),
        ("cơm tấm có mấy loại thịt", "ba loại"),
    ],
}

# SPATIAL — đáp án là vị trí
SPATIAL_FACTS = {
    "pho_bo_viet_nam": [
        ("trong tô phở, thịt thường nằm ở đâu", "trên cùng"),
        ("nước dùng phở nằm ở vị trí nào trong tô", "bên dưới"),
        ("bánh phở thường nằm ở đâu trong tô", "ở giữa"),
        ("hành lá trong phở nằm ở vị trí nào", "trên cùng"),
        ("rau thơm thường được đặt ở đâu", "bên cạnh"),
        ("ớt và chanh đặt ở đâu khi ăn phở", "bên cạnh"),
    ],
    "bun_bo_hue": [
        ("thịt bò trong bún bò huế nằm ở đâu", "trên cùng"),
        ("nước dùng nằm ở vị trí nào trong tô bún bò", "bên dưới"),
        ("sợi bún nằm ở đâu trong tô", "ở giữa"),
        ("giò heo trong bún bò nằm ở vị trí nào", "trên cùng"),
        ("rau ăn kèm bún bò đặt ở đâu", "bên cạnh"),
        ("hành tím trong bún bò nằm ở đâu", "trên cùng"),
    ],
    "banh_mi_thit": [
        ("thịt trong bánh mì nằm ở đâu", "ở giữa"),
        ("rau sống trong bánh mì nằm ở vị trí nào", "ở giữa"),
        ("bơ thường được phết ở đâu", "bên trong"),
        ("nhân bánh mì thịt nằm ở đâu", "ở giữa"),
        ("dưa leo trong bánh mì nằm ở vị trí nào", "ở giữa"),
        ("vỏ bánh mì nằm ở đâu", "bên ngoài"),
    ],
    "banh_mi_pate": [
        ("pate trong bánh mì nằm ở vị trí nào", "ở giữa"),
        ("nhân bánh mì pate nằm ở đâu", "ở giữa"),
        ("vỏ bánh mì pate nằm ở đâu", "bên ngoài"),
        ("rau trong bánh mì pate nằm ở vị trí nào", "ở giữa"),
        ("dưa leo nằm ở đâu trong bánh mì pate", "ở giữa"),
        ("bơ trong bánh mì pate được phết ở đâu", "bên trong"),
    ],
    "banh_xeo_viet_nam": [
        ("nhân bánh xèo nằm ở đâu", "bên trong"),
        ("vỏ bánh xèo nằm ở vị trí nào", "bên ngoài"),
        ("tôm trong bánh xèo nằm ở đâu", "bên trong"),
        ("giá đỗ trong bánh xèo nằm ở vị trí nào", "bên trong"),
        ("rau ăn kèm bánh xèo đặt ở đâu", "bên cạnh"),
        ("nước chấm bánh xèo đặt ở đâu", "bên cạnh"),
    ],
    "banh_chung": [
        ("nhân bánh chưng nằm ở đâu", "ở giữa"),
        ("lá dong gói bánh chưng nằm ở vị trí nào", "bên ngoài"),
        ("đậu xanh trong bánh chưng nằm ở đâu", "ở giữa"),
        ("thịt trong bánh chưng nằm ở vị trí nào", "ở giữa"),
        ("gạo nếp trong bánh chưng nằm ở đâu", "bên ngoài"),
        ("dây lạt buộc bánh chưng nằm ở đâu", "bên ngoài"),
    ],
    "banh_bao": [
        ("nhân bánh bao nằm ở đâu", "ở giữa"),
        ("vỏ bánh bao nằm ở vị trí nào", "bên ngoài"),
        ("trứng trong bánh bao nằm ở đâu", "ở giữa"),
        ("thịt trong bánh bao nằm ở vị trí nào", "ở giữa"),
        ("bột bánh bao nằm ở đâu", "bên ngoài"),
        ("phần nhân của bánh bao ở vị trí nào", "ở giữa"),
    ],
    "banh_uot": [
        ("nhân bánh ướt nằm ở đâu", "bên trong"),
        ("hành phi trong bánh ướt nằm ở vị trí nào", "trên cùng"),
        ("nước chấm bánh ướt đặt ở đâu", "bên cạnh"),
        ("chả lụa ăn kèm nằm ở vị trí nào", "bên cạnh"),
        ("vỏ bánh ướt nằm ở đâu", "bên ngoài"),
        ("rau ăn kèm bánh ướt đặt ở đâu", "bên cạnh"),
    ],
    "goi_cuon_viet_nam": [
        ("nhân gỏi cuốn nằm ở đâu", "bên trong"),
        ("bánh tráng gỏi cuốn nằm ở vị trí nào", "bên ngoài"),
        ("tôm trong gỏi cuốn nằm ở đâu", "bên trong"),
        ("rau sống trong gỏi cuốn nằm ở vị trí nào", "bên trong"),
        ("nước chấm gỏi cuốn đặt ở đâu", "bên cạnh"),
        ("bún trong gỏi cuốn nằm ở đâu", "bên trong"),
    ],
    "com_tam_suon_bi_cha": [
        ("sườn nướng trong cơm tấm nằm ở đâu", "trên cùng"),
        ("cơm tấm nằm ở vị trí nào trong đĩa", "bên dưới"),
        ("chả trong cơm tấm nằm ở đâu", "trên cùng"),
        ("bì trong cơm tấm nằm ở vị trí nào", "trên cùng"),
        ("dưa chua ăn kèm cơm tấm đặt ở đâu", "bên cạnh"),
        ("nước mắm cơm tấm đặt ở đâu", "bên cạnh"),
    ],
}

# ──────────────────────────────────────────────────────────────────────────
# PARAPHRASE PREFIXES — để mỗi câu có nhiều biến thể tiếng Việt tự nhiên
# ──────────────────────────────────────────────────────────────────────────
PREFIXES = [
    "",                                # trực tiếp
    "Cho mình hỏi, ",
    "Bạn có thể cho biết, ",
    "Theo bạn, ",
    "Hãy cho mình biết, ",
    "Bạn cho biết, ",
]

SUFFIXES = ["?", " nhỉ?", " vậy?", " nhé?", " không?"]


def make_paraphrases(base_q: str, n: int) -> list:
    """Tạo n biến thể paraphrase của câu hỏi gốc."""
    rng = random.Random(hash(base_q) % (2**32))
    variants = set()
    base_q = base_q.strip()

    while len(variants) < n:
        prefix = rng.choice(PREFIXES)
        suffix = rng.choice(SUFFIXES)
        if prefix:
            q = prefix + base_q[0].lower() + base_q[1:] + suffix
        else:
            q = base_q[0].upper() + base_q[1:] + suffix
        variants.add(q)
        if len(variants) >= 30:    # tránh infinite loop
            break

    return list(variants)[:n]


# ──────────────────────────────────────────────────────────────────────────
# GENERATE NEW SAMPLES
# ──────────────────────────────────────────────────────────────────────────
def extract_category(image_path: str) -> str:
    parts = image_path.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else "unknown"


def generate_samples_for_image(image_path: str, n_count: int, n_spatial: int) -> list:
    """Sinh n_count câu counting + n_spatial câu spatial cho 1 ảnh."""
    cat = extract_category(image_path)
    samples = []
    rng = random.Random(hash(image_path) % (2**32))

    # ── COUNTING ───────────────────────────────────────────────────────────
    if cat in COUNTING_FACTS:
        facts = COUNTING_FACTS[cat][:]
        rng.shuffle(facts)
        # Lấy n_count facts khác nhau
        for i, (base_q, answer) in enumerate(facts[:n_count]):
            paraphrases = make_paraphrases(base_q, 1)
            samples.append({
                "image"   : image_path,
                "question": paraphrases[0],
                "answer"  : answer,
                "type"    : "counting",
            })

    # ── SPATIAL ────────────────────────────────────────────────────────────
    if cat in SPATIAL_FACTS:
        facts = SPATIAL_FACTS[cat][:]
        rng.shuffle(facts)
        for i, (base_q, answer) in enumerate(facts[:n_spatial]):
            paraphrases = make_paraphrases(base_q, 1)
            samples.append({
                "image"   : image_path,
                "question": paraphrases[0],
                "answer"  : answer,
                "type"    : "spatial",
            })

    return samples


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  AUGMENT WITH COUNTING & SPATIAL QUESTIONS")
    print("=" * 60)

    # ── Load existing fixed annotations ──────────────────────────────────
    print("\n[1/4] Loading existing annotations_fixed/...")
    splits = {}
    for s in ["train", "val", "test"]:
        with open(os.path.join(SRC_DIR, f"{s}.json"), encoding="utf-8") as f:
            splits[s] = json.load(f)
        print(f"      {s}: {len(splits[s]):,} samples")

    # ── Get unique images per split ──────────────────────────────────────
    print("\n[2/4] Identifying unique images per split...")
    imgs_by_split = {s: list(set(item["image"] for item in splits[s])) for s in splits}
    for s, imgs in imgs_by_split.items():
        print(f"      {s}: {len(imgs)} unique images")

    # ── Generate new samples ─────────────────────────────────────────────
    print(f"\n[3/4] Generating {N_COUNTING_PER_IMAGE} counting + "
          f"{N_SPATIAL_PER_IMAGE} spatial per image...")

    new_samples = {s: [] for s in splits}
    for s in splits:
        for img in imgs_by_split[s]:
            new = generate_samples_for_image(
                img, N_COUNTING_PER_IMAGE, N_SPATIAL_PER_IMAGE
            )
            new_samples[s].extend(new)

    print(f"      Generated:")
    for s in new_samples:
        c = sum(1 for x in new_samples[s] if x["type"] == "counting")
        sp = sum(1 for x in new_samples[s] if x["type"] == "spatial")
        print(f"      {s:>5}: +{c} counting, +{sp} spatial = +{c+sp} total")

    # ── Merge & save ─────────────────────────────────────────────────────
    print(f"\n[4/4] Merging and saving to {DST_DIR}/...")
    os.makedirs(DST_DIR, exist_ok=True)

    merged = {}
    for s in splits:
        # Concat existing + new, then shuffle (cùng seed để reproducible)
        merged[s] = splits[s] + new_samples[s]
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(merged[s])

        path = os.path.join(DST_DIR, f"{s}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged[s], f, ensure_ascii=False, indent=2)

        print(f"      {path}: {len(merged[s]):,} samples "
              f"({len(splits[s]):,} original + {len(new_samples[s])} new)")

    # ── Final stats ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL STATS")
    print("=" * 60)

    type_dist = Counter()
    for s in merged:
        for item in merged[s]:
            type_dist[item.get("type", "unknown")] += 1
    total = sum(type_dist.values())

    print(f"  Total samples: {total:,}")
    print(f"\n  Question type distribution:")
    for t in ["yes_no", "counting", "recognition", "attribute", "spatial", "other"]:
        n = type_dist.get(t, 0)
        bar = "█" * int(40 * n / max(type_dist.values()))
        print(f"    {t:<14}: {n:>5,}  {100*n/total:5.1f}%  {bar}")

    # Markdown report
    md = [
        "# Dataset Augmentation Report",
        "",
        f"Bổ sung {N_COUNTING_PER_IMAGE} câu hỏi counting và "
        f"{N_SPATIAL_PER_IMAGE} câu hỏi spatial cho mỗi ảnh.",
        "",
        "## Phân bố loại câu hỏi (sau khi bổ sung)",
        "",
        "| Loại | Số mẫu | Tỷ lệ |",
        "|---|---:|---:|",
    ]
    for t in ["yes_no", "counting", "recognition", "attribute", "spatial", "other"]:
        n = type_dist.get(t, 0)
        md.append(f"| {t} | {n:,} | {100*n/total:.1f}% |")

    md.extend([
        "",
        f"**Tổng:** {total:,} mẫu (tăng từ {sum(len(splits[s]) for s in splits):,})",
        "",
        "## So sánh trước / sau",
        "",
        "| Split | Trước | Sau | Tăng |",
        "|---|---:|---:|---:|",
    ])
    for s in splits:
        before = len(splits[s])
        after  = len(merged[s])
        md.append(f"| {s} | {before:,} | {after:,} | +{after-before} |")

    with open("augment_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n  Report saved to augment_report.md")

    print("\n  ✅ Done. Trong notebook đổi:")
    print('    TRAIN_JSON = "annotations_augmented/train.json"')
    print('    VAL_JSON   = "annotations_augmented/val.json"')
    print('    TEST_JSON  = "annotations_augmented/test.json"')


if __name__ == "__main__":
    main()
