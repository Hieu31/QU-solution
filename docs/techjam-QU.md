# Cải Thiện Query Understanding Trong Ngân Sách Production
### ReparoS: Nâng Cao Năng Lực Chuẩn Hoá Truy Vấn Tiếng Việt Có Kiểm Soát

**TechJam 2026 · Search Quality Team**

---
## 1. Problem Statement

QU cần chuẩn hóa các truy vấn địa điểm không chuẩn của người dùng thành dạng mà Search có thể xử lý tốt, **mà không làm thay đổi intent gốc của query**.

1. **Rule-based khó mở rộng:** QU hiện tại phụ thuộc nhiều vào rule và dictionary, xử lý tốt các pattern đã biết nhưng khó generalize với typo, Telex/VNI, từ dính/tách và viết tắt phụ thuộc ngữ cảnh (`bv`, `kcn`, `q.`, `tp hcm`).

2. **Pipeline ngày càng khó bảo trì:** Normalization được xử lý qua nhiều lớp rule. Thêm rule để xử lý case mới có thể tạo interaction với rule cũ và gây regression trên những query đang chạy đúng.

3. **Khó đánh giá chất lượng sau khi sửa:** Một query được normalize thành công về mặt text chưa chắc giúp Search tốt hơn. Cần xác định được phép sửa nào thực sự cải thiện khả năng hiểu/truy xuất của Search và phép sửa nào không mang lại lợi ích.

4. **Sửa sai nguy hiểm hơn không sửa:** QU phải tránh làm thay đổi entity hoặc intent vốn đã đúng. Vì vậy, mục tiêu không phải sửa càng nhiều càng tốt, mà là **tăng coverage trên query nhiễu trong khi kiểm soát regression trên query đang hoạt động tốt**.

---

## 2. Proposal

Đề xuất dùng **Base V3 đã chốt** làm bộ sinh gợi ý, sau đó tách hai câu hỏi: *phép sửa văn bản có hợp lý và giữ được thông tin đã nêu trong query không?* và *query được sửa có tìm ra địa điểm phù hợp cho người dùng không?* Chỉ nhìn một query địa chỉ mơ hồ không thể biết chắc thực thể người dùng định đến; hai điểm này được tính riêng trước khi quyết định kết quả cuối. Đây là thiết kế đề xuất cho Search; các hệ số, ngưỡng và hiệu quả tìm kiếm bên dưới **chưa phải kết quả đã đo**.

### 2.1. Sinh ứng viên bằng Base V3

Với query gốc `q`, chạy CTranslate2 export tại `artifacts/reparos_base_v3_production_checkpoints/checkpoints/base_v3_production/ctranslate2_export` với beam 10 để lấy tối đa 10 cách viết `c₁…c₁₀` cùng `sequence_score` `s₁…s₁₀`. Giữ `q` làm phương án không sửa. Cố định checkpoint, tokenizer, bước tiền xử lý và cấu hình decode khi tạo dữ liệu huấn luyện confidence cũng như khi phục vụ; score của hai cấu hình khác nhau không được so trực tiếp. Các rule xác định vẫn dùng để bảo vệ số nhà, mã tòa, ký hiệu và các trường hợp có quy tắc rõ ràng, thay vì thêm rule riêng cho từng lỗi chính tả.

### 2.2. Chấm confidence **cho từng** ứng viên

Điểm `Cᵢ` trả lời câu hỏi hẹp hơn: **“Theo tín hiệu của Base V3, `cᵢ` có phải một phép sửa văn bản hợp lý và không làm mất thông tin địa chỉ đã nêu trong `q` không?”** Nó chưa khẳng định `cᵢ` trỏ tới đúng địa điểm người dùng định chọn và chưa dùng kết quả Search hoặc vị trí. Với mỗi `cᵢ`, trích xuất `sᵢ`, hạng của ứng viên, khoảng cách điểm `s₁−sᵢ`, margin chung `s₁−s₂`, mức thay đổi ký tự/độ dài, mức giống nhau sau khi bỏ dấu và khoảng cách điểm với ứng viên lân cận. Một logistic regression nhỏ học trọng số từ nhãn của từng ứng viên và trả `Cᵢ = sigmoid(b + Σwⱼfⱼ(q,cᵢ))`. Do đó, ngay cả khi Search đẩy ứng viên hạng 3 lên đầu, nó vẫn có `C₃` riêng; không lấy confidence của top 1 gán cho nó. Mười điểm này không cần cộng thành 1 vì nhiều cách sửa có thể cùng hợp lý, hoặc cả mười cách đều không hợp lý.

Thử nghiệm sơ bộ dùng ba bộ `reparos-diagnostic-10k`, `reparos-compositional-4k` và `reparos-user-centric-v2` (14.088 query), gán nhãn tự động bằng cách so với gold. Trên 3.067 query giữ riêng theo nhóm đáp án, biến thể logistic đầy đủ đạt Brier **0,03182**, so với **0,03866** khi chỉ dùng score model. Đây mới là phép đo **khả năng trùng gold**, chưa phải phép đo trực tiếp cho định nghĩa `Cᵢ` ở trên: một cách sửa khác gold vẫn có thể hợp lý. Bước tiếp theo là rà soát các ứng viên theo ba nhãn `an toàn / có hại / không đủ căn cứ`; ví dụ đổi `ngõ 325` thành `ngõ 35` là có hại, còn việc chọn giữa hai địa danh cùng tên khi query không nêu khu vực có thể không đủ căn cứ. Không ép ca mơ hồ thành đúng hoặc sai. Trước khi dùng ngưỡng tự sửa trong production, cần kiểm tra calibration trên nhãn này và trên mẫu query thật.

### 2.3. Chấm kết quả Search và độ phù hợp địa phương

Gửi `q` và các ứng viên `cᵢ` tới Search theo batch nếu API hỗ trợ. Với mỗi ứng viên, lấy các địa điểm trả về và chọn kết quả tốt nhất `rᵢ`. **Giả định** Search cung cấp điểm liên quan văn bản, tên/địa chỉ, loại địa điểm và tọa độ. Chấm `rᵢ` bằng ba tín hiệu tách biệt: `Rᵢ` là độ liên quan của Search sau khi chuẩn hóa để so được giữa các query; `Lᵢ` là mức phù hợp vị trí người dùng, ví dụ hàm giảm theo khoảng cách; `Aᵢ` là tính nhất quán của số nhà, tên đường, địa danh và loại POI với những thông tin được nêu rõ trong `q`. Vị trí là tín hiệu mềm: khi query nêu rõ một nơi ở xa, khoảng cách gần không được lấn át địa danh được yêu cầu. Ứng viên không có kết quả Search hữu ích không được coi là tốt chỉ vì `Cᵢ` cao.

Điểm để xếp hạng cặp *(cách sửa, địa điểm)* được đề xuất là `Fᵢ = α·Cᵢ + β·Rᵢ + γ·Lᵢ + δ·Aᵢ − η·Hᵢ`, trong đó `Hᵢ` là rủi ro đổi số nhà, mã tòa hoặc thực thể quan trọng. Confidence của model và điểm Search/locality vẫn là **hai nguồn tín hiệu khác nhau**; phép cộng này không biến `Fᵢ` thành xác suất đúng. Các tín hiệu cần đưa về thang đo nhất quán; trọng số và mức rủi ro phải được chọn trên dữ liệu có nhãn kết quả địa điểm, chưa ấn định trong proposal.

### 2.4. Quyết định an toàn và phép đo cần có

Chỉ ưu tiên một cách sửa khi kết quả địa điểm của nó tốt hơn Search trên `q` gốc một mức đủ rõ và không vi phạm các điều kiện bảo vệ ý định. Nếu bằng chứng yếu, giữ nguyên `q` hoặc hiển thị gợi ý để người dùng chọn. Thu thập failure case và tín hiệu lựa chọn kết quả để rà soát, gán nhãn lại và cập nhật confidence theo thời gian; click đơn lẻ không được mặc định là nhãn “sửa đúng”.

Đánh giá theo ba tầng: **(1)** calibration và tỷ lệ sửa sai của `Cᵢ` theo từng hạng/top 10; **(2)** địa điểm đúng ở top 1, khoảng cách tới địa điểm mong muốn và số query có kết quả Search hữu ích sau khi sửa; **(3)** tỷ lệ sửa có hại trên query sạch, số ca đổi số nhà/thực thể và p95/p99 latency của cả luồng. So riêng Search trên query gốc, dùng Base V3 top 1 trực tiếp, dùng confidence, và dùng confidence + Search/locality để biết mỗi tầng thực sự đóng góp gì.

---

## 3. Thực Nghiệm Kiến Trúc Arm E (Đối Chiếu)

Mục 3–4 ghi lại kết quả của Arm E trong nhánh thực nghiệm kiến trúc. **Proposal ở Mục 2 dùng checkpoint Base V3 đã chốt**; không gán số liệu chất lượng hoặc độ trễ của Arm E cho Base V3 và tầng Search mới. Arm E hướng tới chuẩn hoá tiếng Việt chính xác với **ngân sách CPU < 10ms**, không cần máy chủ GPU.

### 3.1. Thiết Kế & Huấn Luyện
* **Kiến trúc (Arm E):** Transformer Seq2Seq **2E / 2D**, **7.1M params** ($d_{model}=128, d_{ff}=2048, 4\text{ heads}$). Thêm 1 tầng Decoder giúp triệt tiêu lỗi nuốt/tráo từ ở câu địa chỉ dài với chi phí trễ chỉ +3.3ms.
* **Tokenizer:** SentencePiece Unigram 12,000 vocab. Bắt buộc chuẩn hóa `Unicode NFC + Lowercase` trước khi tokenize $\to$ **UNK Rate = 0.00%**.
* **Dữ liệu 4M pairs:** 30% Địa danh & tiền tố hành chính (`p19`, `q1`, `bv`), 30% Lỗi gõ Telex/VNI/phím di động, 30% Replay Anchor (`X → X` bảo toàn câu sạch), 10% DAE Noise.
* **Huấn luyện:** OpenNMT-py 3.5.1, Adam, Noam Decay, 50,000 steps. Xuất sang **CTranslate2 INT8** (kích thước chỉ **14.1 MB** trên đĩa).

---

## 4. Kết Quả Thực Nghiệm Arm E & Đo Lường Chất Lượng

Đánh giá toàn diện trên **5 Frozen Suites (3,100 queries cố định)**:

| Bộ Test / Năng Lực Đánh Giá | Quy mô | Top-1 EM (Inline) | Recall@10 (Multi-Cand) | Đánh Giá & Nhận Xét Kỹ Thuật |
|:---|:---:|:---:|:---:|:---|
| **1. Bản đồ OSM (Retention)** | **1,200** | **76.3%** | **87.5%** | **P50 = 7.38ms, CER = 8.97%**. Khôi phục địa danh OSM cực mạnh |
| ↳ Dính phím gõ Telex (`tahnhf` $\to$ `thành`) | 200 | 89.5% | 94.5% | Sửa dính phím Telex tiếng Việt xuất sắc |
| ↳ Dính số gõ VNI (`thoai5` $\to$ `thoại`) | 200 | 85.0% | 95.5% | Khôi phục phím số VNI chuẩn xác |
| ↳ Khôi phục mất dấu bán phần/toàn phần | 200 | 79.5% | 90.5% | Phục hồi dấu câu địa chỉ chuẩn |
| ↳ Sai vị trí dấu thanh (`hoà` $\to$ `hòa`) | 200 | 78.5% | 91.5% | Chuẩn hoá dấu chuẩn Unicode |
| ↳ Tách/dính ranh giới từ (`ngã 4`, `quan an`) | 200 | 70.0% | 84.0% | Tách dính số và từ tốt |
| ↳ Trượt phím lân cận QWERTY (`qura` $\to$ `qua`) | 200 | 55.0% | 69.0% | Lỗi trượt phím phụ thuộc ngữ cảnh |
| **2. Bảo vệ từ quen (Seen Brands)** | **400** | **93.5%** | **97.5%** | **P50 = 6.04ms, CER = 2.33%**. Replay Anchor chống over-correct |
| ↳ Tên thương hiệu đã học (Brand Seen) | 200 | 98.0% | 100.0% | Bảo vệ thương hiệu quen thuộc gần như tuyệt đối |
| ↳ Truy vấn sạch đã học (Clean Seen) | 200 | 89.0% | 95.0% | Giữ nguyên câu đúng không sửa bậy |
| **3. Bảo vệ từ mới (Heldout Brands)** | **500** | **62.0%** | **72.4%** | **P50 = 6.97ms, CER = 7.08%**. Năng lực tổng quát hoá OOD |
| ↳ Truy vấn sạch mới lạ OOD (Clean Heldout) | 200 | 83.5% | 92.5% | Khả năng giữ nguyên câu sạch chưa từng học |
| ↳ Thương hiệu mới chưa học (Brand Heldout) | 300 | 47.7% | 59.0% | Điểm nghẽn: Mô hình nhỏ bị bias subword |
| **4. Truy vấn thực tế (User-Centric)** | **300** | **58.3%** | **66.7%** | **P50 = 7.78ms, CER = 9.38%**. Truy vấn người dùng thật |
| **5. Sửa dấu & Viết tắt (Plasticity)** | **700** | **45.1%** | **49.0%** | **P50 = 8.51ms, CER = 16.59%**. Bị kéo tụt do viết tắt lạ |
| ↳ Sửa lỗi gõ tổ hợp (Composition) | 200 | 78.0% | 85.0% | Sửa lỗi gõ dính phím/Telex rất tốt |
| ↳ Viết tắt địa chỉ (`d.` $\to$ `đường`/`phố`, `p.`, `q.`) | 250 | 56.8% | 61.2% | Bị trừ điểm do lệch nhãn `đường` vs `phố` (Search vẫn đúng) |
| ↳ Viết tắt ngữ cảnh hẹp (`bv bm` $\to$ `bạch mai`) | 250 | 7.2% | 8.0% | Kéo tụt điểm toàn bộ suite do thiếu từ điển bách khoa |
| **TRUNG BÌNH TOÀN BỘ (OVERALL)** | **3,100** | **67.4%** | **75.7%** | **P50 = 7.53ms, CER = 8.87%, UNK = 0.00%** |

### Đánh Giá Chất Lượng Theo Tác Động Tìm Kiếm:
1. **Correct (Chính xác tuyệt đối):** Mở rộng từ viết tắt và phục hồi dấu thanh chuẩn xác ở Rank #1 (`bv cho ray` $\to$ `bệnh viện chợ rẫy`, `d. pasteur q3` $\to$ `đường pasteur quận 3`, Telex đạt **89.5%**).
2. **Acceptable (Lệch text nhưng Search đúng):** Exact Match tính sai nhưng Geocoding định vị đúng toạ độ:
   - Nhập nhằng loại đường: `đ đặng dung` $\to$ `đường đặng dung` (nhãn ghi `phố`).
   - Tự động khử lặp: `hà nội hà nội` $\to$ `hà nội`.
3. **Harmful & Gaps (Cần xử lý ở Stage 2 Fine-Tuning):**
   - Ký tự lạ (`.`, `'`): `co.opmart` $\to$ `ðầ`, `pizza 4p's` $\to$ `pizza 4po s` do vỡ subword.
   - Viết tắt bách khoa quá lạ: `bv bm` $\to$ `bệnh viện mb` (không tự đoán mò được `bạch mai`).

---
