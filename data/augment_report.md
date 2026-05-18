# Dataset Augmentation Report

Bổ sung 4 câu hỏi counting và 4 câu hỏi spatial cho mỗi ảnh, sau đó dedupe (image, question) pairs.

## Phân bố loại câu hỏi (final, sau augment + dedupe)

| Loại        | Số mẫu | Tỷ lệ |
|---          |   ---: |  ---: |
| yes_no      |  4,114 | 35.4% |
| attribute   |  3,331 | 28.6% |
| recognition |  1,730 | 14.9% |
| spatial     |  1,144 |  9.8% |
| counting    |  1,144 |  9.8% |
| other       |    164 |  1.4% |

**Tổng:** 11,627 mẫu

## Pipeline 3 giai đoạn

| Giai đoạn                          | Train | Val   | Test  | Tổng    |
|---                                 | ---:  | ---:  | ---:  | ---:    |
| (a) Gốc                            | 7,469 |   948 |   941 |  9,358  |
| (b) Sau augment counting + spatial | 9,293 | 1,180 | 1,173 | 11,646  |
| (c) Sau dedupe (image, question)   | 9,277 | 1,178 | 1,172 | 11,627  |

Dedupe giai đoạn (c): drop 19 cặp `(image, question)` trùng — chiếm 0.16%.
