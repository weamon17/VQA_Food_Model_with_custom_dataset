# Đánh giá dữ liệu — Vietnamese Food VQA

## 1. Tổng quan

- **Tổng số mẫu**: 11,627
- **Tổng số ảnh**: 286
- **Số món ăn**: 10
- **Vocab câu trả lời**: 63 từ/cụm unique

## 2. Phân chia split (80/10/10 stratified theo món ăn)

| Split | Số mẫu | Số ảnh unique | Tỷ lệ |
|---|---:|---:|---:|
| train | 9,277 | 228 | 79.8% |
| val | 1,178 | 29 | 10.1% |
| test | 1,172 | 29 | 10.1% |

## 3. Phân bố loại câu hỏi

| Loại câu hỏi | Số mẫu | Tỷ lệ |
|---|---:|---:|
| yes_no | 4,114 | 35.4% |
| attribute | 3,331 | 28.6% |
| recognition | 1,730 | 14.9% |
| spatial | 1,144 | 9.8% |
| counting | 1,144 | 9.8% |
| other | 164 | 1.4% |

## 4. Phân bố độ dài câu trả lời

| Split | Min | Max | Mean |
|---|---:|---:|---:|
| train | 1 | 4 | 1.49 |
| val | 1 | 4 | 1.49 |
| test | 1 | 4 | 1.49 |

## 5. Phân bố theo món ăn (kiểm tra cân bằng)

| Món ăn | Train | Val | Test |
|---|---:|---:|---:|
| banh_bao | 811 | 122 | 120 |
| banh_chung | 855 | 123 | 123 |
| banh_mi_pate | 821 | 80 | 82 |
| banh_mi_thit | 1047 | 121 | 123 |
| banh_uot | 902 | 123 | 119 |
| banh_xeo_viet_nam | 849 | 125 | 123 |
| bun_bo_hue | 970 | 119 | 121 |
| com_tam_suon_bi_cha | 1065 | 121 | 123 |
| goi_cuon_viet_nam | 890 | 124 | 120 |
| pho_bo_viet_nam | 1067 | 120 | 118 |

## 6. Tính chất quan trọng (cần lưu ý khi đọc kết quả)

- Không có image leak giữa train/val/test (đã verify).
- Trung bình ~40 câu hỏi/ảnh trong train → cùng 1 ảnh xuất hiện nhiều lần với câu hỏi khác.
- Vocab đáp án chỉ ~60 từ/cụm → bài toán gần như **classification thay vì generation**.
- Yes/no chiếm 35.4% — majority baseline ~52% trên subset này.
- Counting + spatial là **templated theo món ăn** (cùng category → cùng đáp án) — metric trên 2 type này phản ánh khả năng phân loại món hơn là visual reasoning.