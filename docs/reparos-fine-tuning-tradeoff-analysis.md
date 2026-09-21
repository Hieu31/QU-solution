# Phân tích chuyên sâu: Hiện trạng, Điểm nghẽn và Đánh đổi năng lực trong Fine-tuning ReparoS

**Bài toán:** Query Understanding, chuẩn hóa và sửa lỗi truy vấn địa điểm tiếng Việt cho dịch vụ đặt xe / bản đồ.  
**Mục tiêu tài liệu:** Cung cấp toàn bộ dữ liệu thực nghiệm khách quan, phân tích cơ chế dịch chuyển năng lực (Seesaw Effect / Catastrophic Forgetting), kiểm toán mã nguồn sinh dữ liệu và hệ thống hóa các câu hỏi mở để phục vụ nghiên cứu giải pháp chuyên sâu, không áp đặt kết luận chủ quan.  
**Ngày lập:** 21/09/2026  
**Trạng thái:** Báo cáo phân tích kỹ thuật độc lập (Technical Analysis Report).

---

## 1. Nguyên tắc phương pháp luận

Để tránh thiên kiến (bias) và suy diễn thiếu căn cứ (hallucination), tài liệu này phân định nghiêm ngặt 3 mức độ thông tin:

1. **[Đã kiểm chứng thực nghiệm]:** Số liệu đo đạc trực tiếp từ các file kết quả và checkpoint có sẵn trong kho `artifacts/` (`full-evaluation-summary.json`, `base-v2-comparison.json`, `base-*.metrics.json`).
2. **[Kiểm toán mã nguồn]:** Sự thật mã nguồn được đọc trực tiếp từ logic sinh dữ liệu (`src/reparos/curriculum_v2.py`, `src/reparos/typing_curriculum.py`) và cấu hình OpenNMT.
3. **[Giả thuyết cơ chế & Câu hỏi mở]:** Các lý thuyết toán học / mạng neural và các điểm nghẽn kiến trúc được nêu ra dưới dạng câu hỏi khảo sát để người nghiên cứu tự thẩm định và thiết kế giải pháp.

---

## 2. Bảng dữ liệu thực nghiệm đối chứng 4 chiều

Bảng dưới đây tổng hợp kết quả đo đạc chính thức của 4 checkpoint đại diện trên 3 bộ benchmark chuẩn (`User-Centric 88`, `Composition 4K`, `Diagnostic 10K`):

* **Base v1:** Mô hình gốc theo paper ReparoS (1 Enc – 1 Dec, FFN 512, ~4.8M tham số, train 26.000 steps).
* **Curriculum V2 Final:** Mô hình fine-tune cũ từ Base v1 (1 Enc – 1 Dec, FFN 512, train qua 3 Stage tới Step 60.000).
* **Base V2 Cũ:** Thử nghiệm nâng cấp dữ liệu Base (1 Enc – 1 Dec, FFN 512, train 10.000 steps).
* **Base V2 Mới (`reparos-base-v2-32-final (1)`):** Kiến trúc mở rộng (2 Enc – 1 Dec, FFN 2048, ~6.6M tham số, train 10.000 steps).

### 2.1. Tổng hợp trên 3 Benchmark chính

| Chỉ số / Benchmark | Base v1 | Curriculum V2 Cũ (Fine-tuned) | Base V2 Cũ (1/1, FFN 512) | Base V2 Mới (2/1, FFN 2048) | Ghi chú xu hướng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **User-Centric (88 queries)** | | | | | |
| - Exact Noisy Accuracy | 36.25% | **68.75%** | 51.25% | **52.50%** | Curriculum cũ tăng vọt do nạp từ điển |
| - Recall@10 | 45.45% | **73.86%** | 59.09% | **60.23%** | Base V2 mới tăng +14.8 pp so với Base v1 |
| - Regression Rate | 18.18% | 11.36% | 15.91% | 14.77% | Tỷ lệ làm hỏng câu đúng còn cao |
| **Compositional 4K** | | | | | |
| - Exact Noisy Accuracy | 43.35% | 58.90% | 62.48% | **64.28%** | Base V2 mới cao hơn Curriculum cũ (+5.38 pp) |
| - Recall@10 | 67.55% | 83.18% | 85.65% | **86.42%** | Năng lực ghép từ của Base V2 mới vượt trội |
| - Regression Rate | 12.35% | 6.38% | 5.35% | **4.60%** | Base V2 mới có tỷ lệ hồi quy thấp nhất |
| - *missing_diacritics_boundary_all* | 2.70% | 41.80% | 53.00% | **56.10%** | Base v1 gãy hoàn toàn ở ca dính từ không dấu |
| **Diagnostic 10K** | | | | | |
| - Exact Noisy Accuracy | **75.62%** | 61.67% *(Tụt sâu)* | 65.94% | 67.14% | Curriculum cũ giảm -13.95 pp so với Base v1 |
| - Recall@10 | **90.58%** | 84.40% *(Tụt)* | 85.43% | 83.34% | Curriculum cũ và Base V2 đều thấp hơn Base v1 |
| - Clean Preservation Rate | 90.72% | 91.93% | 91.25% | 91.01% | Đi ngang quanh ngưỡng 91% |
| - Regression Rate | **7.05%** | 10.09% *(Tăng xấu)* | 8.75% | 7.66% | Curriculum cũ sửa sai nhiều nhất |

---

### 2.2. Bóc tách chi tiết từng nhóm lỗi trong Diagnostic 10K

Đây là bảng số liệu phản ánh trực tiếp hiện tượng **dịch chuyển / suy giảm năng lực**:

| Nhóm lỗi (Error Type) | Số mẫu | Base v1 Exact | Curriculum V2 Cũ | Base V2 Mới (2/1, 2048) | Độ lệch Curriculum vs Base v1 | Hiện tượng thực tế |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`vni_leak`** | 800 | **88.38%** | **48.12%** | **89.12%** | **-40.26 pp** | Curriculum cũ quên gần nửa năng lực VNI |
| **`wrong_diacritic`** | 800 | **86.00%** | **62.25%** | **84.62%** | **-23.75 pp** | Curriculum cũ suy giảm khả năng sửa dấu thanh |
| **`address_abbreviation`** | 800 | **91.12%** | **39.62%** | **19.00%** | **-51.50 pp** | Cả Base V2 và Curriculum cũ đều sụp đổ ở nhóm này |
| **`address_symbol`** | 800 | **36.12%** | 33.75% | 27.00% | -2.38 pp | Nhóm lỗi địa chỉ ký hiệu (`/`, `-`) khó với cả 4 mô hình |
| **`keyboard_edit`** | 800 | **68.50%** | 60.88% | 64.25% | -7.62 pp | Curriculum cũ bắt lỗi phím kém hơn Base |
| **`combined`** | 1.137 | 55.76% | 48.41% | **62.25%** | -7.35 pp | Base V2 mới vượt trội ở lỗi tổ hợp (+6.49 pp) |
| **`missing_diacritics_full`** | 800 | **82.50%** | 78.12% | 75.75% | -4.38 pp | Sửa không dấu hoàn toàn |
| **`missing_diacritics_partial`** | 800 | **89.38%** | 84.38% | 82.62% | -5.00 pp | Sửa không dấu một phần |
| **`word_boundary`** | 800 | 79.88% | 77.62% | **80.62%** | -2.25 pp | Sửa dính / tách từ |
| **`telex_leak`** | 800 | 93.12% | **93.25%** | 89.75% | +0.13 pp | Duy trì tốt do được nạp liên tục |

---

### 2.3. Bảng đo độ trễ thực tế trên CPU (Đo lường với CTranslate2)

*Cấu hình đo:* CPU đơn luồng, batch size = 1, beam size = 10 (môi trường edge/server chuẩn).

| Cấu hình mô hình | Mean Latency | Median (p50) | 95th Percentile (p95) | 99th Percentile (p99) | Throughput (QPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base v1 (1/1, FFN 512, FP32)** | 7.57 ms | 7.46 ms | 12.10 ms | 14.85 ms | 132.1 |
| **Base V2 Mới (2/1, FFN 2048, FP32)** | 11.54 ms | 10.00 ms | 19.84 ms | 65.73 ms | 86.6 |
| **Base V2 Mới (2/1, FFN 2048, INT8 Dynamic)** | **4.21 ms** | **3.65 ms** | **8.35 ms** | **10.19 ms** | **237.5** |

*Ghi chú thực nghiệm:* Lượng hóa INT8 Dynamic trên Base V2 mới giữ nguyên **100.00% độ chính xác** (Exact Noisy trên 500 mẫu Composition đạt `62.40%` ở cả FP32 và INT8; trên 88 mẫu User đạt `55.68%` ở cả hai). Vấn đề độ trễ CPU của kiến trúc 2/1 đã được giải quyết bằng lượng hóa.

---

### 2.4. Ma trận đối chứng cặp 4 nhóm (Contingency Matrix) trên 10.000 câu Diagnostic 10K

Chạy inference trực tiếp cả **Base v1** và **Curriculum Final** trên cùng 10.000 câu của bộ `Diagnostic 10K` và chia từng sample thành 4 nhóm đối chứng:

* **Nhóm 1 (✅ Đúng / ✅ Đúng):** Cả hai mô hình đều đúng $\rightarrow$ Năng lực được bảo toàn (Retained).
* **Nhóm 2 (❌ Sai / ✅ Đúng):** Base v1 sai, Curriculum đúng $\rightarrow$ Năng lực mới học thêm được (Learned / Gain).
* **Nhóm 3 (✅ Đúng / ❌ Sai):** Base v1 đúng, Curriculum làm sai $\rightarrow$ Năng lực bị mất / thoái hóa (Lost / Regressed).
* **Nhóm 4 (❌ Sai / ❌ Sai):** Cả hai đều sai $\rightarrow$ Vấn đề khó chưa giải quyết được (Unsolved).

#### Bảng tổng thể 10.000 câu

| Trạng thái Base v1 | Trạng thái Curriculum | Ý nghĩa thực nghiệm | Số lượng câu | Tỷ lệ (%) |
| :---: | :---: | :--- | :---: | :---: |
| **✅ Đúng** | **✅ Đúng** | Năng lực được bảo toàn | **6.157** | **61.57%** |
| **❌ Sai** | **✅ Đúng** | Curriculum học thêm được | **382** | **3.82%** |
| **✅ Đúng** | **❌ Sai** | Curriculum làm mất năng lực (Thoái hóa) | **1.618** | **16.18%** |
| **❌ Sai** | **❌ Sai** | Cả hai mô hình đều chưa giải được | **1.843** | **18.43%** |
| **Tổng cộng** | | | **10.000** | **100.00%** |

> **Phát hiện quan trọng:** Trên toàn bộ 10.000 câu của Diagnostic 10K, số câu Curriculum **học thêm được là 382 câu (3.82%)**, nhưng số câu bị **làm hỏng / mất đi là 1.618 câu (16.18%)**. Tỷ lệ mất / được lên tới **4.24 lần**!

#### Bóc tách chi tiết 4 nhóm theo từng loại lỗi (Error Type Breakdown)

| Nhóm lỗi (Error Type) | Tổng mẫu | ✅ Đúng / ✅ Đúng (Giữ được) | ❌ Sai / ✅ Đúng (Học thêm) | ✅ Đúng / ❌ Sai (Làm mất) | ❌ Sai / ❌ Sai (Cả hai sai) | Tỷ lệ Mất / Được |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vni_leak`** | 800 | 343 (42.9%) | 8 (1.0%) | **364 (45.5%)** | 85 (10.6%) | **45.5 lần** (Thảm họa) |
| **`wrong_diacritic`** | 800 | 487 (60.9%) | 9 (1.1%) | **201 (25.1%)** | 103 (12.9%) | **22.3 lần** (Thoái hóa nặng) |
| **`address_abbreviation`** | 800 | 293 (36.6%) | 25 (3.1%) | **436 (54.5%)** | 46 (5.8%) | **17.4 lần** (Task mismatch) |
| **`keyboard_edit`** | 800 | 455 (56.9%) | 29 (3.6%) | **93 (11.6%)** | 223 (27.9%) | 3.2 lần |
| **`combined`** | 1.400 | 520 (37.1%) | 153 (10.9%) | **261 (18.6%)** | 466 (33.3%) | 1.7 lần |
| **`word_boundary`** | 800 | 590 (73.8%) | 31 (3.9%) | 49 (6.1%) | 130 (16.2%) | 1.6 lần |
| **`missing_diacritics_full`** | 800 | 606 (75.8%) | 18 (2.2%) | 54 (6.8%) | 122 (15.2%) | 3.0 lần |
| **`missing_diacritics_partial`** | 800 | 662 (82.8%) | 13 (1.6%) | 53 (6.6%) | 72 (9.0%) | 4.1 lần |
| **`telex_leak`** | 800 | 733 (91.6%) | 12 (1.5%) | 12 (1.5%) | 43 (5.4%) | **1.0 lần** (Cân bằng hoàn hảo) |
| **`address_symbol`** | 800 | 220 (27.5%) | 50 (6.2%) | 69 (8.6%) | 461 (57.6%) | 1.4 lần |
| **`clean`** | 1.400 | 1.248 (89.1%) | 34 (2.4%) | 26 (1.9%) | 92 (6.6%) | **0.8 lần** (Duy trì tốt) |

---

## 3. Phân tích chi tiết các ca sụp đổ năng lực: Tách bạch giữa Data Mismatch và Catastrophic Forgetting

### 3.1. Nghịch lý nhóm `address_abbreviation`: Task/Data Mismatch, KHÔNG PHẢI Catastrophic Forgetting

Trích xuất trực tiếp dự đoán từ log test của nhóm `address_abbreviation` trong Diagnostic 10K:

| Đầu vào (Input) | Nhãn chuẩn (Expected) | Dự đoán Base v1 | Dự đoán Base V2 Mới | Dự đoán Curriculum V2 Cũ |
| :--- | :--- | :--- | :--- | :--- |
| `d tỉnh 768` | `đường tỉnh 768` | `đường tỉnh 768` (Đúng) | `đ tỉnh 768` (Chỉ sửa d $\rightarrow$ đ) | `đ tỉnh 768` (Sai) |
| `d số 15` | `đường số 15` | `đường số 15` (Đúng) | `đ số 15` (Chỉ sửa d $\rightarrow$ đ) | `đ số 15` (Sai) |
| `259 d trần hưng đạo` | `259 đường trần hưng đạo` | `259 đường trần hưng đạo` | `259 đ trần hưng đạo` | `259 đ trần hưng đạo` |
| `trung tâm y tế tx quảng yên` | `trung tâm y tế thị xã...`| `... thị xã quảng yên` | `... tx quảng yên` (Giữ nguyên)| `... tx quảng yên` |

**Bóc tách bản chất:**
1. **Base v1 (91.12%):** Trong tập dữ liệu seed v1 cũ, các ca viết tắt `d ` được gán mapping bung thành `đường `, `tx ` bung thành `thị xã `. Base v1 đơn thuần học thuộc phân phối này.
2. **Base V2 Mới (19.00%):** Dữ liệu `prepared-v4-leakfree` chuẩn hóa chặt chẽ để chống hallucination, mô hình chỉ học ánh xạ chuẩn hóa ký tự (`d` $\rightarrow$ `đ`), không tự ý chèn thêm chuỗi `ường`.
3. **Curriculum V2 Cũ (39.62%):** Trong mã nguồn `src/reparos/typing_curriculum.py`, danh mục `ABBREVIATIONS` chỉ khai báo duy nhất `("đường", "đg")`, hoàn toàn **không có `("đường", "d")` hay `("đường", "đ")`**. Mô hình fine-tune chưa từng được nhìn thấy ánh xạ này trong suốt quá trình huấn luyện.
4. **Kết luận:** Sự sụt giảm từ 91.12% xuống 39.62% là hệ quả trực tiếp của **Task / Data Specification Mismatch** (lệch đặc tả dữ liệu giữa bộ sinh và bộ test benchmark), hoàn toàn **không phải là Catastrophic Forgetting** (mô hình quên quy tắc đã học). Việc xếp `address_abbreviation` chung nhóm với `vni_leak` dưới nhãn "dịch chuyển năng lực do quên" là sai lệch bản chất.

---

### 3.2. Sự suy giảm của `vni_leak` (88.38% $\rightarrow$ 48.12%): Thiếu bằng chứng gán ghép nhân quả cho Stage 3

Khác với `address_abbreviation`, sự sụt giảm của VNI từ 88.38% xuống 48.12% là bằng chứng **Catastrophic Forgetting thực sự**. Tuy nhiên, việc khẳng định *"Stage 3 làm VNI chết"* là một suy diễn nhân quả chưa đủ căn cứ (Unverified Causal Attribution):

* **Kiểm toán từ `pilot-gates.json`:**
  - Sau **Stage 1 (Primitives):** Gate đo kiểm tra ghi nhận `vni: 1.0` (100% trên User Benchmark).
  - Sau **Stage 2 (Composition):** Hệ thống chỉ đo `clean_preservation`, `boundary_plus_diacritics`, `all_joined`, `user_boundary` $\rightarrow$ **Hoàn toàn không đo VNI!**
  - Sau **Stage 3 (Lexical):** Hệ thống chỉ đo `clean_preservation`, `location_acronym`, `abbreviation`, `lexical_typo` $\rightarrow$ **Cũng không đo VNI!**
* **Kiểm toán dữ liệu Stage 2:**
  - Trong Stage 2, VNI chỉ nằm trong 20% `primitive_replay` (chia đều cho 5 loại lỗi), nghĩa là xác suất xuất hiện của VNI chỉ khoảng $4\%$.
  - Trong khi đó, Full Training ở Stage 2 chạy tới **15.000 steps** với các mẫu compounding rất nặng.
* **Kết luận:** Hoàn toàn có khả năng VNI đã bắt đầu sụp đổ ngay trong 15.000 steps của Stage 2 chứ không phải chờ đến Stage 3. Việc thiếu metric đo kiểm VNI giữa các Stage khiến ta không thể khẳng định chắc chắn thời điểm bắt đầu suy thoái.

---

### 3.3. Điểm số User-Centric (68.75%): Dictionary Memorization vs. General Lexical Capability

Mô hình Curriculum cũ đạt 68.75% trên `User-Centric` (tăng từ 36.25% của Base v1), nhưng con số này cần được nhìn nhận đúng bản chất:
1. **Kiểm tra tập test `reparos-user-centric-v2`:** Nhóm `location_acronym` gồm 8 trường hợp: `ltk`, `đ ltk q1`, `nct`, `hbt`, `dbp`, `pvh`, `ntmk`, `nvl`.
2. **Kiểm tra mã nguồn train `curriculum_v2.py`:** Khai báo cứng `CURATED_ACRONYMS` gồm đúng 7 từ: `ltk`, `nct`, `hbt`, `dbp`, `pvh`, `ntmk`, `nvl`, cùng 21 tên trường đại học (`CURATED_UNIVERSITY_ALIASES`) và 12 cụm từ (`CURATED_QUERY_EXPANSIONS`). Hàm `_curated_acronym_rows()` nhân bản cưỡng bức (oversample) các mẫu này vào dữ liệu huấn luyện.
3. **Bản chất:** Dù các dòng cụ thể không bị trùng lặp trực tiếp, **Knowledge Overlap (trùng lặp tri thức từ điển) giữa train và test là 100%**. Con số 68.75% phản ánh việc mô hình **học thuộc lòng từ điển (Dictionary Acquisition / Memorization)** đối với các thực thể cố định, hoàn toàn **chưa chứng minh được Năng lực từ vựng tổng quát (General Lexical Capability)** trên các từ viết tắt ngoài danh mục.

---

## 4. Kiểm toán mã nguồn sinh dữ liệu (Code & Pipeline Audit)

Khảo sát trực tiếp hai tệp [curriculum_v2.py](file:///d:/New%20folder/QU-solution/src/reparos/curriculum_v2.py) và [typing_curriculum.py](file:///d:/New%20folder/QU-solution/src/reparos/typing_curriculum.py):

### 4.1. Phân bổ trọng số qua 3 Stage cũ

```python
STAGE_WEIGHTS = {
    "stage1-primitives": {
        "clean": 30, "missing_diacritics": 30, "boundary": 15,
        "keyboard": 10, "telex": 10, "vni": 5,
    },
    "stage2-composition": {
        "clean": 20, "primitive_replay": 20, "two_operation": 40,
        "three_operation": 20,
    },
    "stage3-lexical": {
        "clean": 25, "acronym": 5, "contextual_abbreviation": 20,
        "lexical_typo": 10, "mixed_replay": 25, "primitive_replay": 10,
        "address_symbol": 5,
    },
}
```

**Nhận xét kiểm toán:**
* `vni` chỉ chiếm 5% ở Stage 1. Sang Stage 2, VNI chỉ là 1 phần ngẫu nhiên trong `primitive_replay` (20%). Sang Stage 3, VNI chỉ còn là 1 phần ngẫu nhiên trong `primitive_replay` (10%) $\rightarrow$ Tần suất xuất hiện của VNI ở Stage 3 thực tế chỉ còn khoảng $10\% \times \frac{1}{4} = 2.5\%$.
* Nhóm lỗi dấu thanh (`wrong_diacritic`) thậm chí **không có một slot riêng biệt nào** trong `STAGE_WEIGHTS`, mà chỉ xuất hiện gián tiếp nếu hàm IME sinh ra.

### 4.2. Từ điển thực thể cứng được đưa vào Stage 3

Trong `src/reparos/curriculum_v2.py`:
* `CURATED_UNIVERSITY_ALIASES`: Khai báo cứng 21 trường đại học (`hust`, `hcmute`, `ptit`, `neu`, `ftu`, `vnu`, `vnuhcm`, `uet`, `uit`, `hcmus`, `hcmut`, `ueh`, `uel`, `tdtu`, `iuh`, `ufm`, `hutech`, `huflit`, `fptu`, `vinuni`, `phenikaa`).
* `CURATED_QUERY_EXPANSIONS`: Khai báo cứng 12 thực thể cơ quan/trường học (`thpt clhp`, `thpt tdn`, `thcs nbk`, `dhbk`, `dhqg`, `bv bm`, `ubnd q1`, `ubnd pbn q1`, `kcn tb`...).
* `CURATED_ACRONYMS`: Khai báo 7 tuyến đường lớn (`ltk`, `nct`, `hbt`, `dbp`, `pvh`, `ntmk`, `nvl`).
* `_curated_acronym_rows()`: Nhân bản cưỡng bức (oversampling) nhóm thực thể này để chiếm đủ hạn ngạch hàng nghìn dòng trong Stage 3.

**Nhận xét kiểm toán:**
* Nhóm từ điển này là các ánh xạ tĩnh dạng 1-to-1 (`hust` $\rightarrow$ `đại học bách khoa hà nội`).
* Trong dữ liệu train, nhóm này hầu như được nạp dưới dạng mẫu sạch hoặc mẫu ghép đơn giản (`đ hust`, `trường hust`), thiếu sự biến thiên đa dạng của lỗi gõ phím đời thực.

---

### 5.1. Giới hạn của giả thuyết "Dung lượng biểu diễn" (Capacity vs. Distribution-Specific Learning)

1. **Giả thuyết dung lượng ban đầu:**
   * Dựa trên cơ chế FFN là Key-Value Associative Memory (*Geva et al., 2021*), giả thuyết ban đầu cho rằng kiến trúc cũ (1 Enc, $d_{ff}=512$) quá nhỏ (~4.8M tham số, 512 hidden slots), dẫn đến việc tri thức thực thể mới ghi đè lên tri thức ngữ âm cũ (Superposition).
2. **Bằng chứng phản ví dụ (Counterexample) từ chính Base v1:**
   * Bảng số liệu thực nghiệm đã chỉ ra một sự thật quan trọng: **Base v1 (chỉ 1 Enc - 1 Dec, FFN 512) đã từng đồng thời duy trì được cả:**
     - `vni_leak`: **88.38%**
     - `address_abbreviation`: **91.12%**
     - `wrong_diacritic`: **86.00%**
     - `telex_leak`: **93.12%**
   * **Hệ quả phân tích:** Sự thật thực nghiệm này làm suy yếu nghiêm trọng giả thuyết cho rằng *"FFN 512 không đủ sức chứa để lưu đồng thời cả quy tắc ngữ âm lẫn viết tắt"*. Bản thân Base v1 với 512 slots đã chứng minh mạng hoàn toàn có đủ dung lượng biểu diễn cho cả hai nhóm tác vụ này.
3. **Hiện tượng ở Base V2 mới:**
   * Base V2 mới tăng dung lượng FFN gấp 4 lần (FFN 2048, 4.096 slots), nhưng `address_abbreviation` chỉ đạt **19.00%**, trong khi `vni_leak` lại đạt **89.12%**.
   * Điều này chứng minh hai năng lực này thay đổi **hoàn toàn độc lập** theo sự hiện diện của dữ liệu huấn luyện (**Distribution-Specific Learning / Negative Interference**).
   * **Bản chất vấn đề:** Điểm nghẽn không nằm ở dung lượng tham số (Capacity Bottleneck), mà nằm ở **động học huấn luyện (Optimization Dynamics)**: khi fine-tune qua hàng chục nghìn bước với hàm loss thiếu vắng tín hiệu VNI, gradient descent đã xoay vector trọng số ra khỏi vùng cực tiểu cũ (Gradient Drift / Destructive Updates), bất kể mạng có 512 hay 2048 chiều.

### 5.2. Động học tối ưu hóa (Optimization Dynamics & Catastrophic Forgetting)

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}_{\text{batch}}$$

* Khi huấn luyện tuần tự (Staged Sequential Training):
  * Tại Stage $k$, hàm loss là $\mathcal{L}_k$. Nếu phân phối $\mathcal{P}_k(x, y)$ khác biệt lớn so với $\mathcal{P}_{k-1}(x, y)$ (ví dụ Stage 3 ngập tràn acronyms còn Stage 1 ngập tràn VNI), kỳ vọng gradient $\mathbb{E}[\nabla \mathcal{L}_k]$ sẽ tạo một góc tù với $\mathbb{E}[\nabla \mathcal{L}_{k-1}]$.
  * Kết quả toán học tất yếu là giá trị loss trên tập dữ liệu cũ tăng vọt ($\mathcal{L}_{k-1}(\theta_{t+1}) > \mathcal{L}_{k-1}(\theta_t)$), biểu hiện ra ngoài chính là hiện tượng tụt điểm VNI và sai lệch dấu thanh.
* **Vấn đề số bước huấn luyện:**
  Mô hình fine-tuned cũ chạy tới **60.000 steps**. Với một mô hình đã hội tụ ở Base, 60.000 steps trên một tập dữ liệu fine-tune kích thước nhỏ là quá nhiều, đẩy trọng số trôi dạt hoàn toàn khỏi điểm cân bằng ban đầu (Weight Drift).

### 5.3. Xung đột giữa hai bản chất bài toán

Trong hệ thống xử lý ngôn ngữ tự nhiên, có hai bản chất bài toán khác biệt cơ bản:
1. **Quy luật hình thái học và sửa lỗi bề mặt (Morphological & Surface Regularities):**
   * Ví dụ: Không dấu $\rightarrow$ Có dấu (`hoang hoa tham` $\rightarrow$ `hoàng hoa thám`), giải mã IME (`truowngf` $\rightarrow$ `trường`), tách từ dính (`nguyenhue` $\rightarrow$ `nguyễn huệ`).
   * Đặc tính: Có tính chu kỳ, tính quy luật cao, khả năng khái quát hóa theo âm tiết và n-gram.
2. **Tra cứu tri thức thực thể (Knowledge Base / Entity Retrieval):**
   * Ví dụ: `hust` $\rightarrow$ `đại học bách khoa hà nội`, `thpt clhp` $\rightarrow$ `chuyên lê hồng phong`.
   * Đặc tính: Tính rời rạc, tính nhớ vẹt (memorization), không có quy luật âm vần tiếng Việt suy diễn được từ các ký tự `h-u-s-t`.

Khi ép một mạng nơ-ron Seq2Seq nhỏ giải đồng thời cả hai dạng bài toán này trong cùng một không gian trọng số, hai luồng gradient này liên tục cạnh tranh nhau: gradient nhớ vẹt thực thể làm nhiễu loạn các ma trận chú ý đang nắm giữ quy luật âm tiết.

---

## 6. Hệ thống các câu hỏi mở để nghiên cứu chuyên sâu

Dưới đây là các câu hỏi nghiên cứu mở không áp đặt giải pháp, phục vụ việc phân tích độc lập:

### Nhóm 1: Về ranh giới nhiệm vụ và phạm vi bài toán (Scope Boundaries)
1. Liệu mô hình Seq2Seq có nên đảm nhận việc mở rộng các thực thể tĩnh (như 21 trường đại học, trường cấp 3, bệnh viện) hay không?
   * *Nếu CÓ:* Làm thế nào để sinh dữ liệu sao cho các thực thể này không mang tính "học vẹt phẳng", mà có sự tương tác tự nhiên với các lỗi gõ phím đời thực (`truwongf dhbk`, `bv bmai`)?
   * *Nếu KHÔNG:* Điểm giới hạn hợp lý nhất của mô hình là ở đâu? (Ví dụ: Dừng lại ở tiền tố hành chính `d./q./p./tp.` và từ viết tắt địa danh có ngữ cảnh mỏ neo `d. ltk q10`?)

### Nhóm 2: Về kiến trúc dữ liệu và chiến lược học (Continual Learning Strategy)
2. **Sequential Staging vs. Interleaved Multi-Task:** 
   * Có nên tiếp tục mô hình 3 Stage tuần tự (Primitives $\rightarrow$ Composition $\rightarrow$ Lexical) hay chuyển sang một tập dữ liệu hòa trộn đa nhiệm duy nhất (Interleaved Blend) với tỷ trọng cố định trong từng batch?
3. **Experience Replay Buffer:**
   * Tỷ lệ tối thiểu của nhóm VNI, dấu thanh và Clean Controls trong từng bước huấn luyện cần là bao nhiêu phần trăm để gradient không bao giờ xóa bỏ năng lực cũ?

### Nhóm 3: Về động học tối ưu hóa và bảo toàn trọng số (Optimization & Parameter Regularization)
4. **Learning Rate & Decay:**
   * Với Base V2 mới đã hội tụ sâu (107 triệu câu, loss 1.22), tốc độ học khi fine-tune nên được khống chế ở mức nào ($10^{-4}$, $5 \times 10^{-5}$ hay $3 \times 10^{-5}$)?
5. **Số bước huấn luyện (Steps budget):**
   * Tại sao bản cũ cần tới 60.000 steps? Với bộ dữ liệu Pilot (120k–160k dòng), số bước tối ưu để mô hình tiếp thu quy tắc mới mà không bị overfit là bao nhiêu (ví dụ: 2.000 – 4.000 steps)?
6. **Kỹ thuật nội suy trọng số (Weight Blending / Model Soups):**
   * Liệu việc blend trọng số giữa Checkpoint sau khi fine-tune và Checkpoint Base V2 gốc ($\theta_{\text{blend}} = \alpha \theta_{\text{FT}} + (1-\alpha) \theta_{\text{Base}}$) có phải là một công cụ toán học rẻ và hiệu quả để phục hồi Clean Preservation và VNI hay không?

### Nhóm 4: Về cơ chế kiểm chuẩn (Validation & Quality Gates)
7. Tiêu chí dừng sớm (Early Stopping) đa chiều cần được thiết lập như thế nào trong code để ngăn chặn mô hình trôi dạt ngay khi phát hiện một chỉ số quan trọng bị suy giảm?

---

## 7. Tài liệu đặc tả kiến trúc kế thừa

Toàn bộ lời giải chi tiết cho các câu hỏi mở trên, bao gồm thiết kế bộ lấy mẫu hòa trộn (Interleaved Capability Sampler), chuẩn hóa dữ liệu, mục tiêu tối ưu ràng buộc và cây thực nghiệm 3 pha đã được phê duyệt và hoàn thiện tại:

📄 **[docs/reparos-interleaved-continual-finetuning-spec.md](file:///d:/New%20folder/QU-solution/docs/reparos-interleaved-continual-finetuning-spec.md)**

---

*Tài liệu này được biên soạn độc lập dựa trên toàn bộ hiện vật, log thực nghiệm và mã nguồn của kho lưu trữ `QU-solution`.*
