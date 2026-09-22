# BÁO CÁO TOÀN DIỆN HIỆN TRẠNG HỆ THỐNG VÀ KẾT QUẢ THỬ NGHIỆM REPAROS (BASE V1 -> V2 -> V3 -> V3.1 FT)

> **Tài liệu Kỹ thuật Nội bộ — Dự án Query Understanding (QU-Solution)**  
> **Tác giả:** Đội ngũ Kỹ thuật ReparoS  
> **Thời điểm cập nhật:** 22/09/2026  
> **Mục tiêu:** Ghi nhận trung thực, đầy đủ toàn bộ hiện trạng thực tế, các thử nghiệm đã thực hiện, những điểm đã cải thiện, các nút thắt kỹ thuật đang đối mặt, và mổ xẻ nguyên nhân gốc rễ trước khi đưa ra quyết định chiến lược tiếp theo.

---

## 1. BỐI CẢNH VÀ VAI TRÒ CỦA REPAROS TRONG PIPELINE SẢN PHẨM

Trong kiến trúc tổng thể của hệ thống Tìm kiếm & Địa chỉ, **ReparoS** đóng vai trò là **tầng tiền xử lý đầu tiên (First-pass Query Understanding & Autocorrect)** ngay khi nhận truy vấn từ người dùng, trước khi chuyển giao kết quả sang tầng tiếp theo:

```
[User Query] 
     │
     ▼
[ReparoS Normalizer / Autocorrect]  <--- VỊ TRÍ CỦA MÔ HÌNH HIỆN TẠI
     │ (Chuẩn hóa chính tả, gõ phím, dấu, tiền tố hành chính)
     ▼
[Semantic Parser / NER / Slot-Filling]
     │ (Bóc tách: số nhà, tên đường, phường, quận, tỉnh, POI)
     ▼
[Geocoding / Search Engine Index]
```

### Các ràng buộc cốt tử đối với sản phẩm (Product Constraints):
1. **Nhiệm vụ chuẩn bị dữ liệu cho Semantic Parser**:
   * Mô hình phải chuẩn hóa sạch sẽ các tiền tố viết tắt từ loại hành chính:
     * `đ / d / dg nguyễn trãi` $\to$ `đường nguyễn trãi`
     * `p bến nghé` $\to$ `phường bến nghé`
     * `q1 / q.1 / q 1` $\to$ `quận 1`
     * `tp / tp. hcm` $\to$ `thành phố hồ chí minh`
     * `ubnd` $\to$ `ủy ban nhân dân`, `bv` $\to$ `bệnh viện`
   * *Nếu tầng này không chuẩn hóa sạch, Semantic Parser phía sau sẽ không nhận diện được các entity slot (`street`, `ward`, `district`, `city`) và gãy toàn bộ pipeline.*
2. **Bảo tồn câu sạch (Clean Preservation) $\ge 95\%$**:
   * Tuyệt đối không được "sửa lợn lành thành lợn què". Người dùng gõ đúng thì mô hình không được tự ý sửa đổi làm biến dạng ý định tìm kiếm.
3. **Ngân sách độ trễ siêu khắt khe (Latency Budget)**:
   * Do đứng ở cổng vào của hệ thống chịu tải cao, mô hình phải chạy trên CPU với độ trễ trung bình **$\le 2.0\text{ ms/query}$**, RAM tiêu thụ thấp ($< 50\text{ MB}$).

---

## 2. NHẬT KÝ CÁC THẾ HỆ MÔ HÌNH VÀ THỬ NGHIỆM ĐÃ TRIỂN KHAI

### 2.1. Quá trình tiến hóa kiến trúc và dữ liệu

| Thế hệ | Cấu hình Kiến trúc | Dung lượng & Tham số | Dữ liệu Huấn luyện | Mục tiêu Thử nghiệm |
|---|---|---:|---|---|
| **Base V1** | Enc 1, Dec 1, $d=128$, $d_{ff}=2048$, Vocab 8k | ~4.9M params (INT8: 4.8MB) | 50,000 steps, tập train cơ sở hẹp | Baseline ban đầu |
| **Base V2** | Enc 1, Dec 1, $d=128$, $d_{ff}=2048$, Vocab 8k | ~4.9M params (INT8: 4.8MB) | 32,000 steps, Curriculum V2 | Đẩy mạnh sửa lỗi tổ hợp đa tầng |
| **Base V3** | Enc 2, Dec 1, $d=128$, $d_{ff}=2048$, Vocab 12k | ~6.5M params (INT8: 6.5MB) | 50,000 steps trên **4,000,000 cặp** (40% L1 gõ phím, 25% L2 địa chỉ, 25% L3 clean/brand, 10% L4 DAE) | Mở rộng vocab 12k, nâng 2 layers encoder, bao phủ toàn diện 63 tỉnh thành |
| **Base V3.1 FT** | Warm-start từ Base V3 Step 50k, giữ nguyên 2-1 | ~6.5M params (INT8: 6.5MB) | 6,000 steps (Step 50k $\to$ 56k) trên **1,000,000 cặp** chuyên sâu (60% Replay Anchor, 20% Address Symbol, 15% Nested Acronym, 5% DAE) | Thử nghiệm vá điểm nghẽn `address_symbol` và viết tắt hành chính |

---

## 3. NHỮNG ĐIỂM ĐÃ TỐT LÊN (THÀNH CÔNG ĐÃ ĐƯỢC ĐO ĐẠC KIỂM CHỨNG)

### 3.1. Bảng so sánh đối đầu toàn diện qua 4 thế hệ

#### Benchmark 1: `USER-CENTRIC-V2` (88 câu truy vấn người dùng thực tế)
| Chỉ số | Base V1 | Base V2 | Base V3 | Base V3.1 FT | Tiến bộ V3.1 vs V1 |
|---|---:|---:|---:|---:|---:|
| **Exact Match (Recall@1)** | 32.95% | 47.73% | 50.00% | **55.68%** | **+22.73%** |
| **Case-Insensitive Recall@1** | 42.05% | 55.68% | 57.95% | **63.64%** | **+21.59%** |
| **Recall@5** | 38.64% | 60.23% | 56.82% | **65.91%** | **+27.27%** |
| **Recall@10** | 43.18% | 60.23% | 59.09% | **67.05%** | **+23.87%** |
| **Case-Insensitive Recall@10** | 54.55% | 71.59% | 71.59% | **79.55%** | **+25.00%** |
| **Autocorrect F1** | 38.16% | 54.19% | 56.41% | **60.87%** | **+22.71%** |
| **CER Net Reduction (Giảm khoảng cách lỗi)** | 29.71% | 48.91% | 49.55% | **63.64%** | **+33.93%** |

#### Benchmark 2: `COMPOSITIONAL-4K` (4,000 câu lỗi tổ hợp đa tầng)
| Chỉ số | Base V1 | Base V2 | Base V3 | Base V3.1 FT | Tiến bộ V3.1 vs V1 |
|---|---:|---:|---:|---:|---:|
| **Exact Match (Recall@1)** | 43.18% | 64.05% | 62.95% | **62.72%** | **+19.54%** |
| **Recall@10** | 67.05% | 86.30% | 86.45% | **85.78%** | **+18.73%** |
| **CER Net Reduction** | 39.67% | 78.20% | 78.56% | **77.53%** | **+37.86%** |
| **Tỷ lệ gây thoái lui (Regression Rate)** | 11.88% | 4.32% | 4.58% | **5.17%** | **-6.71%** (an toàn hơn) |

#### Benchmark 3: `DIAGNOSTIC-10K` (10,000 câu chuyên sâu phân rã lỗi)
| Nhóm lỗi chuyên biệt | Base V1 | Base V2 | Base V3 | Base V3.1 FT | Nhận xét |
|---|---:|---:|---:|---:|---|
| **vni_leak (Lỗi gõ nhầm VNI)** | 88.50% | 89.12% | **93.38%** | **91.75%** | Đạt đỉnh cao năng lực giải mã VNI |
| **telex_leak (Lỗi gõ nhầm Telex)** | 93.25% | 89.75% | 92.38% | **93.00%** | Giữ vững xuất sắc mốc >92% |
| **missing_diacritics (Khôi phục dấu)** | 82.50% | 75.50% | 80.00% | **80.00%** | Duy trì ổn định |
| **word_boundary (Dính/tách từ)** | 79.88% | 80.50% | 80.38% | **80.38%** | Duy trì ổn định |

### 3.2. Đo đạc thực tế về Tốc độ (CPU Latency Profile)
Khi tiến hành đo kiểm thực tế trên CPU (4 threads, tập test 88 câu), các kết quả đo đạc chính xác như sau:
* **Chạy Float32 (Bản unquantized, Beam 10)**: $4.82 - 5.52\text{ ms/query}$.
* **Chạy INT8 chuẩn hóa (Quantized)**:
  * **Beam 1 (Greedy Search)**: **$0.72\text{ ms/query}$** (~1,383 QPS).
  * **Beam 3**: **$1.62\text{ ms/query}$** (~616 QPS).
  * **Beam 5**: **$2.44\text{ ms/query}$** (~409 QPS).
* **Phát hiện quan trọng về độ chính xác**:
  * Chạy Beam 1 INT8 đạt **61.36%** độ chính xác Top-1 trên `user_centric_v2`, thậm chí cao hơn Beam 10 (60.23%) do Beam to tạo ra các nhánh sinh ảo giác thừa.

---

## 4. NHỮNG NÚT THẮT BẾ TẮC VÀ LỖI NGHIÊM TRỌNG ĐANG GẶP PHẢI

Mặc dù các chỉ số tổng thể tăng vọt so với V1, mô hình **chưa thể đưa vào production** vì 6 vấn đề kỹ thuật cốt lõi sau:

### Nút thắt 1: Hiện tượng sinh lặp từ (Repetition Hallucination)
* **Triệu chứng thực tế**:
  * Input: `VinFast` $\to$ Model sinh: **`vinfast vinfast`** *(Top 1)*
  * Input: `sanbay noibai` $\to$ Model sinh: **`sân bay nội bài nội`**
* **Nguyên nhân gốc rễ đã kiểm chứng**:
  1. **Nhiễm bẩn từ nhãn huấn luyện**: Quét toàn bộ tập 1,000,000 mẫu [train.tgt](file:///d:/New%20folder/QU-solution/data/base_v3_finetune/train.tgt) phát hiện **9,859 dòng bị lặp từ `"đường đường"`** (như `565 d đường số 15 -> 565 đường đường số 15`). Do script sinh dữ liệu ghép tiền tố `đường` vào chuỗi vốn đã có chữ `đường`.
  2. **Thiếu Repetition Penalty khi giải mã**: CTranslate2 mặc định chạy `repetition_penalty = 1.0`. Khi kiểm chứng bằng code với `repetition_penalty = 1.15`, hiện tượng lặp từ này biến mất hoàn toàn (`vinfast` $\to$ `vinfast`, `sanbay noibai` $\to$ `sân bay nội bài`).

### Nút thắt 2: Bóp méo tiền tố & Ép đoán mò Acronym 3 chữ cái không căn cứ
* **Mâu thuẫn bài toán**:
  * Pipeline rất cần chuẩn hóa `đ nguyễn trãi` $\to$ `đường nguyễn trãi` để Semantic Parser nhận diện entity.
  * Nhưng dữ liệu lại chứa các ca viết tắt 3 chữ cái tên riêng rời rạc vô căn cứ trong bộ `CURATED_ACRONYMS`:
    * `ltk` $\to$ ép thành `lý thường kiệt` (Sao không phải Lê Trọng Tấn, Lương Thế Vinh?)
    * `dbp` $\to$ ép thành `điện biên phủ`
    * `pvh` $\to$ ép thành `phạm văn hai`
  * Hậu quả: Mô hình không thể học nổi các quy tắc vô căn cứ này, dẫn đến đoán mò dị thường: `pvh` sinh ra `phường chũ`, `dbp` sinh ra `đường bp`, `ntmk` sinh ra `nttk`.

### Nút thắt 3: Bài toán số nhà đoán mò (`address_symbol` kẹt cứng ở 26%)
* **Triệu chứng**: Nhóm lỗi `address_symbol` trên 800 mẫu diagnostic chỉ đạt **26.00%** (74% câu bị đoán sai).
* **Nguyên nhân**:
  * Dữ liệu test chứa các ca mất thông tin nhân tạo:
    * `32322 minh phụng` $\to$ ép thành `323/22 minh phụng`
    * `16l` $\to$ ép thành `1/6l`
    * `quốc lộ 1a đường trần phú` $\to$ ép thành `quốc lộ 1a - đường trần phú`
  * Đây là bài toán **thiếu thông tin trầm trọng (Ill-posed problem)**. Ép mô hình học thuộc lòng việc tự chèn dấu xuyệt vào các con số nén sẽ khiến mô hình sinh ảo giác sang cả các số nhà bình thường (ví dụ: `16 lê lợi` bị sửa bậy thành `1/6 lê lợi`).

### Nút thắt 4: Tụt Clean Preservation (Hiệu ứng bập bênh Gradient)
* **Triệu chứng**: Khi nạp 350,000 mẫu số nhà và viết tắt vào đợt fine-tune, tỷ lệ bảo toàn câu sạch (Clean Preservation) bị kéo tụt từ **90.23% xuống 88.67%**.
* **Ý nghĩa thực tế**: Cứ 100 khách hàng gõ câu hoàn toàn đúng, có hơn 11 người bị mô hình tự ý sửa đổi (False Correction Rate = 11.33%). Trong Search Engine, đây là lỗi gây ức chế hàng đầu cho người dùng.

### Nút thắt 5: Tokenizer V3 bị vỡ mã Byte với chữ hoa (Casing Byte-fallback)
* **Triệu chứng**:
  * `vinfast` (chữ thường) $\to$ Tokenizer ra 1 token: `[' vinfast']`.
  * `VinFast` (chữ hoa) $\to$ Bị vỡ thành các byte thô: `[' ', '<0x56>', 'in', '<0x46>', 'a', 'st']`.
  * Hậu quả: Decoder nhìn thấy mã byte lạ liền sinh ảo giác rác (`ahn jiin homestay`). Nếu input không được lowercase đồng bộ trước khi tokenize thì mô hình hoàn toàn gãy trên từ viết hoa.

### Nút thắt 6: Sự bất lực của việc Fine-tuning trên nền Base cũ (Căn nguyên lớn nhất)
* Đúng như nhận định thực chiến: **Fine-tuning chỉ là giải pháp chắp vá (band-aid).**
* Base V3 hiện tại đã được tôi luyện 50,000 steps trên 4M mẫu cũ vốn đã chứa sẵn các nhãn bẩn (`đường đường`), danh sách acronym ép uổng, và tỷ lệ phân bố chưa cân đối.
* Khi nền Base đã có "vết nứt", việc fine-tune đè 6,000 steps ở trên tạo ra hiện tượng **Gradient Interference (xung đột gradient)**:
  * Kéo được số nhà nhích lên một chút (18% $\to$ 26%) thì đạp câu sạch tụt xuống (90% $\to$ 88%).
  * Mô hình bị chạm trần năng lực và không thể bứt phá được nữa.

---

## 5. TỔNG KẾT BỨC TRANH THỰC TẠI

```
                  ┌───────────────────────────────────────────────┐
                  │           HIỆN TRẠNG REPAROS HIỆN TẠI         │
                  └───────────────────────┬───────────────────────┘
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
│       MẶT MẠNH ĐÃ ĐẠT ĐƯỢC           │    │       ĐIỂM NGHẼN CHƯA THỂ RA PROD    │
├──────────────────────────────────────┤    ├──────────────────────────────────────┤
│ 1. Giải mã VNI/Telex đỉnh cao (92-93%)│    │ 1. Bị tật sinh lặp từ (vinfast x2)   │
│ 2. Lỗi tổ hợp giảm 78% CER           │    │ 2. Nhãn train bị 10k lỗi "đường đường"│
│ 3. Query thật R@1 tăng +22% vs V1    │    │ 3. Acronym tên riêng 3 chữ đoán mò   │
│ 4. Độ trễ INT8 CPU cực nhanh (0.7-1.6ms)│ │ 4. Số nhà xuyệt kẹt ở 26% (bất khả thi)│
│ 5. Dung lượng siêu nhẹ (6.5MB INT8)  │    │ 5. Clean Preservation tụt xuống 88.7%│
│                                      │    │ 6. Fine-tuning chạm trần do Base lỗi │
└──────────────────────────────────────┘    └──────────────────────────────────────┘
```

Bản báo cáo này xác lập chính xác ranh giới giữa **những gì hệ thống đã làm tốt** và **những gì đang thực sự cản trở việc đưa ra Production**, làm cơ sở vững chắc để cùng thảo luận và quyết định giải pháp căn cơ tiếp theo.
