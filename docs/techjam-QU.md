# Cải Thiện Query Understanding Trong Ngân Sách Production
### ReparoS: Nâng Cao Năng Lực Chuẩn Hoá Truy Vấn Tiếng Việt Có Kiểm Soát

**TechJam 2026 · Search Quality Team**

---

## 1. Đặt Vấn Đề Trong Môi Trường Thực Tế (Problem Statement)

### 1.1 Bối Cảnh Sản Phẩm & Trải Nghiệm Khách Hàng (The Product Journey)

Trong ứng dụng gọi xe và giao vận công nghệ **Xanh SM**, hành trình cốt lõi của người dùng luôn bắt đầu từ ô tìm kiếm: **Nhập Điểm Đón (Pick-up) và Điểm Trả (Drop-off)**. Đây là "cửa ngõ" quyết định toàn bộ tỷ lệ chuyển đổi (Conversion Funnel) và trải nghiệm của khách hàng.

* **Hoàn cảnh sử dụng thực tế (Mobile on-the-go):**
  Khác với tìm kiếm trên web hay sàn thương mại điện tử (ngồi trước màn hình máy tính hoặc lướt thư thả), người dùng gọi xe thường tương tác trong trạng thái **vội vã, đứng bên lề đường, một tay xách đồ, vừa đi vừa gõ bằng một ngón tay** trên màn hình điện thoại di động nhỏ. Trong điều kiện đó, thói quen gõ tắt, gõ không dấu, nuốt từ, dính chữ số và gõ sai bộ gõ tiếng Việt (Telex/VNI) diễn ra liên tục.

* **Ba "nỗi đau" sản phẩm chí mạng (Core Product Pain Points):**

  1. **Rơi rụng khách hàng vì không ra kết quả (Search Drop-off & Lost GMV):**
     Khi khách hàng gõ vội *"bv cho ray"*, *"nga 4 hang xanh"*, *"kcn song than 1"*, hệ thống tìm kiếm nếu không hiểu sẽ trả về màn hình **"Không tìm thấy điểm đón"**. Khách hàng mất kiên nhẫn, lập tức thoát app và chuyển sang ứng dụng đối thủ (Grab, Be). Mỗi lượt tìm kiếm rỗng là một khách hàng và doanh thu bị mất trực tiếp.
  
  2. **Thảm hoạ đón sai điểm do "sửa lợn lành thành lợn què" (Over-correction Disaster):**
     Nếu mô hình tự ý suy diễn hoặc hallucinate làm đổi số nhà/ngõ (*"ngõ 325"* thành *"ngõ 35"*), đổi phường/quận (*"phường 19"* thành *"phường 1"*), hoặc tráo đổi địa danh (*"bv 115"* thành *"bv chợ rẫy"*), tài xế sẽ nhận cuốc và **di chuyển đến sai điểm đón cách xa hàng km**.
     - **Hậu quả nghiệp vụ:** Khách hàng bức xúc huỷ chuyến (Cancellation), tài xế lãng phí thời gian và pin/xăng, Xanh SM tốn chi phí phát voucher đền bù và tổn hại nghiêm trọng niềm tin thương hiệu.
     - **Nguyên tắc sản phẩm sống còn:** **Thà không sửa, chứ tuyệt đối không được sửa sai địa chỉ đúng của khách hàng.**

  3. **Khựng gián đoạn do độ trễ tìm kiếm (Perceived Latency in Search-as-you-type):**
     Trải nghiệm tìm kiếm điểm đón bắt buộc phải là **Search-as-you-type (Autocomplete tức thì sau từng phím gõ)**. Người dùng gõ đến đâu, danh sách gợi ý phải nảy ra đến đó mượt mà (< 100ms toàn trình). Bất kỳ sự khựng lại nào của giao diện cũng làm đứt mạch suy nghĩ của khách hàng.

### 1.2 Quy Mô Vấn Đề Từ Dữ Liệu Sản Phẩm (3.55 Triệu Zero-Click Logs)

Để thấu hiểu gốc rễ vấn đề, thực hiện audit trực tiếp trên **3.55 triệu log truy vấn Zero-click thực tế** đối sánh với cơ sở dữ liệu bản đồ **OpenStreetMap Việt Nam** (`vietnam-latest.osm.pbf` với gần 500.000 thực thể địa lý). Kết quả bộc lộ các rào cản đặc thù:

* **40.5% truy vấn gắn liền với chữ số:** Người dùng tìm địa chỉ nhà, số ngõ/ngách, toà chung cư, số hiệu phân khu (*"86 xo viet nghe tinh p19 binh thanh"*, *"ngõ 325 kim ngưu"*, *"chung cu s201 vinhomes ocean park"*).
* **Mật độ từ viết tắt hành chính & giao thông dày đặc:** Thói quen viết tắt cố hữu của người Việt: *"dh"* (đại học), *"bv"* (bệnh viện), *"kcn"* (khu công nghiệp), *"ubnd"* (ủy ban nhân dân), *"bx"* (bến xe), *"kđt"* (khu đô thị), *"p."* (phường), *"q."* (quận).
* **Lỗi dính phím và ranh giới từ (Token Boundary):** Gõ liền số và chữ do bộ gõ di động (*"quan1"*, *"p11"*, *"ngo325"*, *"ct1"*).
* **Đuôi dài khổng lồ (Long-tail 77.5%):** 77.5% truy vấn chỉ xuất hiện duy nhất 1 lần trong toàn bộ lịch sử hệ thống, gồm các điểm đón vi mô (micro-POI) và mô tả địa chỉ phức hợp, không có cuốn từ điển tĩnh nào bao quát hết (*"tiệm tạp hoá cô ba ngã ba cây điệp"*, *"lô b chung cư an lộc gò vấp"*, *"quán phở đối diện cổng kcn vsip 1"*).
* **73.2% query zero-click đã có dấu:** Người dùng gõ có dấu nhưng vẫn rỗng kết quả do viết tắt nửa vời, dính phím Telex hoặc lệch cấu trúc tên với dữ liệu OpenStreetMap (*"bv chợ rẫy"*, *"kcn sóng thần 1"*, *"tttm aeon mall tân phú"*, *"ngã 6 tahnhf"* — trong khi index bản đồ lưu dạng đầy đủ *"Bệnh viện Chợ Rẫy"*, *"Trung tâm thương mại AEON Mall"* khiến tìm kiếm từ khoá thất bại).

### 1.3 Vì Sao Các Giải Pháp Sản Phẩm Hiện Nay Thất Bại? (The Dilemma)

Khi đối mặt với bài toán này, các giải pháp kỹ thuật truyền thống đều rơi vào ngõ cụt:

* **Từ điển tra cứu tĩnh (Regex / Gazetteer):** Nhanh nhưng không phân biệt được ngữ cảnh (ví dụ: *"đh"* lúc là *"đại học bách khoa"*, lúc là *"đồng hồ nước"*; *"ct1"* lúc là *"chung cư toà 1"*, lúc là *"cao tốc 1"*). Hệ thống sẽ gãy đổ hoàn toàn khi gặp địa danh mới hoặc cách gõ biến thể.
* **Elasticsearch Fuzzy Search:** Thuật toán tính khoảng cách chỉnh sửa (Levenshtein distance) cực kỳ nguy hiểm trong tìm kiếm địa điểm, vì nó có xu hướng tự động tráo số nhà (*"325"* thành *"35"*) hoặc đổi tên đường tương đồng, dẫn đến thảm hoạ tài xế đón sai điểm.
* **Mô hình Seq2Seq Pretrained lớn (ViT5 226M, BARTpho 132M):** Rất giỏi hiểu ngữ cảnh và nhận diện thương hiệu, nhưng độ trễ suy luận lên tới **45 – 88 ms trên CPU** — **vi phạm ngân sách trải nghiệm thời gian thực**, và chi phí hạ tầng máy chủ GPU sẽ bùng nổ không thể kiểm soát ở quy mô hàng chục ngàn QPS.
* **Mô hình nhỏ tự huấn luyện từ đầu (Scratch Transformer 6–10M params):** Tốc độ tốt, nhưng dính **Capability Gap**: không thể tổng quát hoá các thương hiệu / địa danh mới chưa gặp trong tập train (Brand Held-out chỉ đạt ~60%), dễ hallucinate lặp từ khi gặp truy vấn dài.

### 1.4 Định Vị Module QU ReparoS Trong Kiến Trúc Sản Phẩm

Để giải quyết triệt để bài toán mà không phá vỡ hạ tầng, **ReparoS** được thiết kế như một **bộ đệm dịch thuật thông minh siêu nhẹ (Lightweight Query Understanding Buffer)**, đứng ngay phía trước Search & Geocoding Engine để "dịch" ngôn ngữ gõ vội của người dùng thành ngôn ngữ bản đồ chuẩn xác:

```
[Người dùng gõ trên App Mobile]
  (Đứng ngoài đường, gõ vội 1 tay)
             │
             ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Tầng Query Understanding (QU) - REPAROS (CPU INT8)                     │
 │ - Sửa lỗi chính tả phonetic & gõ dính phím Telex                       │
 │ - Khôi phục dấu thanh cho truy vấn địa danh / địa chỉ                  │
 │ - Mở rộng viết tắt giao thông/hành chính ("bv", "dh", "kcn", "p.", "q")│
 │ - Tách ranh giới từ dính số ("p19", "quan1", "ngõ325")                 │
 └────────────────────────────────────────────────────────────────────────┘
             │
             ▼ 
  [Truy vấn địa điểm đã chuẩn hoá]
```

**Ba ràng buộc vận hành thực tế của hạ tầng sản phẩm:**

1. **Không dùng GPU cho tiền xử lý (Zero-GPU Infrastructure):** Tầng QU đón đầu toàn bộ lượng keystroke từ hàng trăm nghìn người dùng đồng thời (hàng ngàn QPS). Việc cấp GPU riêng cho một bước chuẩn hoá chuỗi văn bản là lãng phí và làm chi phí máy chủ bùng nổ. **Mô hình bắt buộc phải chạy trên CPU pod Kubernetes tiêu chuẩn (x86_64).**
2. **Độ trễ gần như vô hình (Transparent Overhead):** Pipeline tìm kiếm địa lý phía sau (Elasticsearch + Spatial Ranking trên bản đồ) vốn đã tiêu tốn thời gian. Tầng QU chỉ là bước tiền xử lý, nên **độ trễ phải cực nhỏ (< 10.0ms)** để không làm phình tổng thời gian phản hồi của backend và tránh nghẽn hàng đợi (thread starvation) trong giờ cao điểm.
3. **Kích thước siêu nhẹ, nạp tức thì (< 50MB):** Dung lượng mô hình phải nhỏ gọn để nạp tức thì vào RAM pod, phục vụ việc tự động mở rộng cụm máy chủ (Horizontal Pod Autoscaling) trong vài giây khi lượng đặt xe tăng đột biến (mưa gió, tan tầm).

### 1.5 Phát Biểu Bài Toán Sản Phẩm & Tiêu Chí Thành Công

> **Làm thế nào để một mô hình chuẩn hoá siêu nhẹ (< 10M params, < 50MB, chạy trên CPU pod thông thường) có thể hiểu sâu ngữ cảnh để sửa đúng các biến thể địa danh, viết tắt và lỗi gõ vội trong ứng dụng gọi xe Xanh SM, đồng thời đảm bảo an toàn tuyệt đối với truy vấn đúng (No-op Retention ≥ 99.5%) mà không làm tăng độ trễ của hệ thống tìm kiếm?**

Bảng tiêu chí thành công liên kết trực tiếp giữa chỉ số kỹ thuật và tác động sản phẩm:

| Trục Đánh Giá | Chỉ Số Kỹ Thuật | Mục Tiêu Kỹ Thuật | Tác Động Sản Phẩm & Doanh Nghiệp (Product & Business Impact) |
|---|---|---|---|
| **Hiệu năng & Trải nghiệm** | Latency P50 (CPU single-query) | **< 10.0 ms** | Đảm bảo Search-as-you-type mượt mà, không giật lag giao diện |
| | Latency P99 (CPU) | **< 20.0 ms** | Giữ vững trải nghiệm ổn định cả trong giờ cao điểm mưa gió, tan tầm |
| | Kích thước Model | **< 50 MB** | Tiết kiệm RAM pod, nạp model tức thì khi scale-up hạ tầng |
| | Throughput | **> 500 QPS / core** | Tối ưu chi phí server CPU, không cần đầu tư cụm GPU đắt đỏ |
| **Năng lực Địa điểm & POI** | Plasticity (Sửa lỗi sai) | **≥ 50.0%** | Sửa đúng các lỗi gõ vội, thiếu dấu, dính từ để tìm ra điểm đón |
| | Brand & POI Held-out | **≥ 70.0%** | Nhận diện đúng thương hiệu, toà nhà, điểm đón mới xuất hiện |
| | Zero-click Resolution | Cải thiện ca lỗi thực | **Giảm tỷ lệ bỏ app (Session Abandonment), tăng số cuốc xe hoàn tất** |
| **An Toàn Vận Hành** | No-op Retention | **≥ 99.5%** | **Tuyệt đối không sửa sai địa chỉ đúng**, triệt tiêu rủi ro tài xế đón nhầm |
| | Determinism | 100% tất định | Greedy search (Beam=1), đảm bảo kết quả nhất quán với cùng một query |
| | Graceful Fallback | Pass-through input | An toàn phòng vệ: Giữ nguyên query gốc nếu mô hình có độ tự tin thấp |

Deploy một mô hình cồng kềnh phá vỡ ngân sách latency là **giải sai bài toán kinh doanh**. Ngược lại, giữ một mô hình nhỏ nhưng nông cạn, liên tục làm mất intent người dùng là **chấp nhận thất bại về mặt trải nghiệm tìm kiếm**. ReparoS được thiết kế để giải quyết điểm nghẽn này.

---

## 2. Baseline Hiện Tại & Bằng Chứng Thực Nghiệm

Tất cả các mô hình được huấn luyện trên **cùng 300.000 cặp câu sạch**, **10.000 bước huấn luyện**, đánh giá trên **cùng 5 bộ test cố định (3.100 truy vấn) + 18 ca zero-click thực tế**. Runtime suy luận CTranslate2 INT8, CPU, 4 threads (với Arm H đánh giá trên PyTorch native do tokenizer âm tiết custom).

### 2.1 Ma Trận Thiết Kế Kiến Trúc (Architecture Exploration Matrix)

Để tìm ra điểm cân bằng tối ưu giữa năng lực chuẩn hoá và độ trễ phục vụ, không gian kiến trúc được khảo sát có hệ thống từ mô hình Transformer tự sinh (Scratch) đến mô hình Pretrained quy mô lớn:

| Mô hình | Nhóm Kiến trúc | Cấu hình Enc/Dec | $d_{model}$ | $d_{ff}$ | Heads | Số Params | Triết lý Thiết kế (Design Rationale) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Arm A** *(baseline)* | Scratch Shallow | **2E / 1D** | 128 | 2048 | 4 | **6.5M** | Tối đa hoá tốc độ: Decoder 1 tầng cực nông để sinh từ tức thì |
| **Arm B** | Scratch Balanced | **4E / 4D** | 128 | 2048 | 4 | **9.6M** | Cân bằng đối xứngs|
| **Arm C** | Scratch Deep | **6E / 6D** | 128 | 2048 | 4 | **12.1M** | Thăm dò trần độ sâu: Tăng gấp 3 độ sâu Decoder để xem model có thông minh hơn không |
| **Arm D** | Scratch Asymmetric | **4E / 1D** | 128 | 2048 | 4 | **7.7M** | Bất đối xứng: Encoder sâu để hiểu ngữ cảnh, Decoder 1 tầng để giữ latency thấp |
| **Arm E** *(tốt nhất scratch)*| Scratch Balanced-Lite | **2E / 2D** | 128 | 2048 | 4 | **7.1M** | Bổ sung 1 tầng Decoder so với Arm A để khắc phục lỗi nuốt từ và lặp từ |
| **Arm F** | Scratch Wide | **2E / 1D** | **256** | 2048 | 8 | **13.4M** | Thăm dò chiều rộng: Giữ số tầng nông nhưng tăng gấp đôi kích thước hidden dimension |
| **Arm H** *(BARTpho)* | Pretrained BART | **6E / 6D** | 768 | 3072 | 12 | **132M** | Mô hình ngôn ngữ sequence-to-sequence âm tiết tiếng Việt (vinai/bartpho-syllable-base), đóng vai trò **Teacher Upper-bound** |
| **Arm G** *(ViT5-base)* | Pretrained T5 | **12E / 12D** | 768 | 2048 | 12 | **226M** | Mô hình T5 tiếng Việt lớn nhất trong thử nghiệm (VietAI/vit5-base), đóng vai trò **Teacher Upper-bound** |

### 2.2 Bảng Tổng Hợp Benchmark 5 Frozen Suites

| Mô hình | Kiến trúc | Params | P50 CPU | Mean Acc | Plasticity | Retention | User-Centric | Brand Heldout | Zero-Click (Canonical) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Arm A *(baseline)* | 2E/1D d128 | 6.5M | **2.38ms** | 63.50% | 41.7% | 70.2% | 58.3% | 56.8% | **77.8%** (14/18) |
| Arm B | 4E/4D d128 | 9.6M | 5.51ms | 67.11% | 43.7% | 74.2% | 66.7% | 61.0% | **72.2%** (13/18) |
| Arm D | 4E/1D d128 | 7.7M | 7.25ms | 65.87% | 42.4% | 71.1% | 66.7% | 58.4% | **77.8%** (14/18) |
| **Arm E** *(tốt nhất từ scratch)* | 2E/2D d128 | 7.1M | **5.71ms** | **67.33%** | 43.6% | 73.0% | 66.7% | 62.4% | **66.7%** (12/18) |
| Arm F | 2E/1D d256 | 13.4M | 3.97ms | 64.76% | 42.4% | 72.5% | 58.3% | 58.8% | **83.3%** (15/18) |
| Arm C | 6E/6D d128 | 12.1M | 17.63ms | 64.23% | 42.7% | 72.8% | 58.3% | 59.6% | **88.9%** (16/18) |
| **Arm H** *(BARTpho-syllable)* | 6E/6D BART | 132M | ~45ms | **70.68%** | 40.4% | 72.6% | 58.3% | **90.8%** 🚀 | **61.1%** (11/18) |
| **Arm G** *(ViT5-base)* | 12E/12D T5 | 226M | **88.75ms** | **73.43%** | 43.1% | 73.6% | 66.7% | **90.0%** | **66.7%** (12/18) |

### 2.3 Phân Tích Đột Phá: So Sánh Kiến Trúc & Điểm Cân Bằng Tối Ưu (The Architectural Sweet Spot)

Thay vì kết luận đơn giản rằng mô hình lớn luôn tốt hơn, việc phân tích ma trận dữ liệu bộc lộ các phát hiện kiến trúc mang tính bước ngoặt:

#### 1. Mô hình Scratch áp đảo Pretrained ở 3 tiêu chí cốt lõi:
* **Khả năng sửa lỗi ngữ âm & dấu Telex (Plasticity):** 
  Cả **Arm B (43.7%)** và **Arm E (43.6%)** đều **vượt trội hơn cả ViT5 (43.1%) và BARTpho (40.4%)**. Lý do: Mô hình nhỏ được huấn luyện tập trung 100% trọng số vào dữ liệu biến âm địa phương, trong khi mô hình Pretrained mang theo bias ngôn ngữ tổng quát nên dễ phân vân khi gặp các biến thể gõ telex lạ.
* **Xử lý các ca lỗi thực tế (Zero-Click Resolution):** 
  Các mô hình Scratch đạt từ **66.7% đến 88.9%**, áp đảo hoàn toàn **ViT5 (66.7%)** và **BARTpho (61.1%)**. Mô hình Scratch bám sát cấu trúc truy vấn, mở rộng viết tắt hành chính (`bv`, `kcn`, `ubnd`) dứt khoát và không bị **ảo giác ngữ nghĩa (semantic hallucination)** chèn thêm từ lạ (*"chung cư hà nội"*, *"song hành"*).
* **Độ trễ vận hành trên CPU (Serving Latency):** 
  Các mô hình Scratch đạt P50 từ **2.38 ms đến 5.71 ms**, nhanh gấp **15× đến 37×** so với ViT5 (88.75ms) và gấp **8× đến 19×** so với BARTpho (45ms). Đây là yếu tố sống còn để triển khai trên hệ thống search-as-you-type mà không cần đầu tư máy chủ GPU.

#### 2. Pretrained vượt trội ở duy nhất một khía cạnh: Tri Thức Thực Thể (World Knowledge)
* Khoảng cách điểm tổng thể của ViT5 (+6.10 pp) và BARTpho (+3.35 pp) so với Scratch **không đến từ thuật toán sửa dấu hay tách từ**, mà tập trung hoàn toàn ở **Brand Held-out (~90% vs ~60%)**. Nhờ dung lượng bộ nhớ lớn từ pretraining, chúng nhận diện được các tên riêng, toà nhà, địa danh chưa từng thấy trong tập train (như *Bình Quới*).

#### 3. Đánh giá nội bộ họ Scratch: Vì sao Arm E (2E / 2D) là Kiến Trúc Tối Ưu Nhất (Production Winner)?

| Kiến trúc Scratch | Cấu hình | Params | P50 CPU | Mean Acc | Zero-Click | Đánh giá & Giới hạn thực tế |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Arm A** *(baseline)* | 2E / 1D | 6.5M | **2.38ms** | 63.50% | 77.8% | Nhanh nhất nhưng **Decoder 1 tầng quá mỏng**: dễ bị nuốt từ hoặc tráo từ ở các câu địa chỉ dài (`p19` thành `thành phố`). |
| **Arm C** *(deep)* | 6E / 6D | 12.1M | 17.63ms | 64.23% | **88.9%** | Zero-click rất cao nhưng **Mean Acc tụt** và latency vọt lên 17.63ms (> 10ms SLA). Deep scaling từ scratch bị over-fitting và bão hoà. |
| **Arm F** *(wide)* | 2E / 1D (d256)| 13.4M | 3.97ms | 64.76% | 83.3% | Tăng gấp đôi chiều rộng $d_{model}$ tốn RAM gấp 2 lần nhưng không tăng accuracy tổng thể. Chiều rộng không bù đắp được độ sâu suy luận. |
| **Arm D** *(asymmetric)*| 4E / 1D | 7.7M | 7.25ms | 65.87% | 77.8% | Tăng Encoder lên 4 tầng giúp hiểu câu tốt hơn Arm A, nhưng Decoder 1 tầng vẫn là nút thắt cổ chai khi sinh từ. |
| **Arm B** *(balanced)* | 4E / 4D | 9.6M | 5.51ms | 67.11% | 72.2% | Rất tốt về accuracy (Retention 74.2%), nhưng 4 tầng Decoder làm tăng footprint tính toán không cần thiết. |
| 🏆 **Arm E** *(tối ưu)* | **2E / 2D** | **7.1M** | **5.71ms** | **67.33%** | 66.7% | **Điểm ngọt hoàn hảo (The Sweet Spot):** Mean Acc cao nhất dòng scratch, độ trễ 5.71ms cực đẹp, kích thước siêu nhẹ (~28MB INT8). |

> **Tại sao cấu hình 2E / 2D (Arm E) chiến thắng?**
> - **Cân bằng tải sinh từ:** So với Arm A (2E/1D), việc bổ sung **đúng 1 tầng Decoder (2D)** cung cấp thêm tầng Self-Attention và Cross-Attention thứ hai. Điều này cho phép Decoder duy trì ngữ cảnh autoregressive ổn định, triệt tiêu lỗi lặp từ và tráo từ ở các chuỗi địa chỉ dài mà chỉ tốn thêm **3.33ms**.
> - So với 4D hay 6D, cấu hình 2D không bị over-parameterized khi train trên 300K câu, giúp mô hình hội tụ nhanh, kiểm soát tốt tail latency P99 (< 12ms).
> - **Kết luận:** **Arm E (2E/2D d128) chính là nền tảng kiến trúc phần cứng tối ưu nhất để mang đi làm Student Model**, và chỉ cần "bơm" thêm tri thức thực thể từ Teacher (ViT5/BARTpho) là có thể giải quyết trọn vẹn bài toán.

### 2.4 Bảng Đối Soát Toàn Diện 18 Ca Zero-Click Thực Tế (Qualitative Output Inspection)

Dưới đây là bảng đối chiếu kết quả dự đoán chi tiết của cả 4 mô hình đại diện trên toàn bộ **18 ca lỗi truy vấn thực tế** trích từ log production zero-click:

| STT | Truy vấn đầu vào (Input) | Nhãn chuẩn (Canonical Target) | Arm A (Baseline 6.5M) | Arm E (Top Scratch 7.1M) | Arm G (ViT5-base 226M) | Arm H (BARTpho 132M) | Hiện tượng & Nhận xét kỹ thuật |
|:---:|:---|:---|:---|:---|:---|:---|:---|
| 1 | `cau vuot song than` | `cầu vượt sóng thần` | ✅ `cầu vượt sóng thần` | ❌ `cầu vượt sông than` | ❌ `cầu vượt sông than` | ❌ `cầu vượt sông thần` | Địa danh Sóng Thần: Arm A phục hồi chuẩn xác dấu sắc; các mô hình khác bị lệch thanh điệu |
| 2 | `cho ba chieu` | `chợ bà chiểu` | ✅ `chợ bà chiểu` | ✅ `chợ bà chiểu` | ✅ `chợ bà chiểu` | ✅ `chợ bà chiểu` | Chợ Bà Chiểu: Tất cả các mô hình phục hồi dấu thanh hoàn hảo |
| 3 | `nga 4 hang xanh` | `ngã 4 hàng xanh` | ✅ `ngã 4 hàng xanh` | ✅ `ngã 4 hàng xanh` | ❌ `ngã 4 hang xanh` | ✅ `ngã 4 hàng xanh` | Ngã 4 Hàng Xanh: ViT5 bị bảo thủ, bỏ quên dấu huyền ở chữ "hang" |
| 4 | `nga 3 vung tau` | `ngã 3 vũng tàu` | ✅ `ngã 3 vũng tàu` | ❌ `nga 3 vũng tàu` | ✅ `ngã 3 vũng tàu` | ✅ `ngã 3 vũng tàu` | Ngã 3 Vũng Tàu: Arm E quên bỏ dấu ngã ở chữ "nga" |
| 5 | `bv cho ray` | `bệnh viện chợ rẫy` | ✅ `bệnh viện chợ rẫy` | ✅ `bệnh viện chợ rẫy` | ❌ `bệnh viện chợ ray` | ❌ `bệnh viện chợ ray` | Viết tắt BV Chợ Rẫy: Cả 2 mô hình Pretrained đều thiếu dấu ngã ở "rẫy"; Scratch mở rộng và phục hồi dấu xuất sắc |
| 6 | `Bv nhi đong` | `bệnh viện nhi đồng` | ❌ `bệnh viện nhi đong` | ❌ `bệnh viện nhi đong` | ✅ `bệnh viện nhi đồng` | ❌ `ben nhi đong` | BV Nhi Đồng: ViT5 phục hồi hoàn hảo; Scratch thiếu dấu huyền; BARTpho sinh lỗi "ben" |
| 7 | `benh vien 175` | `bệnh viện 175` | ✅ `bệnh viện 175` | ✅ `bệnh viện 175` | ✅ `bệnh viện 175` | ✅ `bệnh viện 175` | Bệnh viện 175: Tất cả mô hình đều bảo toàn số hiệu 175 và phục hồi dấu thanh |
| 8 | `dh kinh te tphcm` | `đại học kinh tế thành phố hồ chí minh` | ✅ `đại học kinh tế thành phố hồ chí minh` | ✅ `đại học kinh tế thành phố hồ chí minh` | ✅ `đại học kinh tế thành phố hồ chí minh` | ✅ `đại học kinh tế thành phố hồ chí minh` | Viết tắt ĐH Kinh tế TPHCM: Cả 4 mô hình mở rộng đa token hoàn hảo |
| 9 | `86 xo viet nghe tinh p19 binh thanh` | `86 xô viết nghệ tĩnh phường 19 bình thạnh` | ❌ `... thành phố bình thạnh` | ✅ `... phường 19 bình thạnh` | ✅ `... phường 19 bình thạnh` | ✅ `... phường 19 bình thạnh` | Địa chỉ dài: Arm A bị tráo "phường 19" thành "thành phố"; Arm E, G, H mở rộng p19 chuẩn xác |
| 10 | `duong le van viet q9` | `đường lê văn việt quận 9` | ✅ `đường lê văn việt quận 9` | ✅ `đường lê văn việt quận 9` | ✅ `đường lê văn việt quận 9` | ✅ `đường lê văn việt quận 9` | Mở rộng đơn vị hành chính q9: Cả 4 mô hình đều xử lý chính xác |
| 11 | `hem 212 thoai ngoc hau phuong phu thanh` | `hẻm 212 thoại ngọc hầu phường phú thạnh` | ✅ `hẻm 212 thoại ngọc hầu phường phú thạnh` | ✅ `hẻm 212 thoại ngọc hầu phường phú thạnh` | ✅ `hẻm 212 thoại ngọc hầu phường phú thạnh` | ❌ `... phường phú thành` | Phường Phú Thạnh: BARTpho nhầm sang "phú thành"; các model khác phục hồi dấu nặng đúng |
| 12 | `kcn song than 1` | `khu công nghiệp sóng thần 1` | ❌ `... sông than 1` | ❌ `... sông thành 1` | ❌ `... song hành 1` | ❌ `... sông thần 1` | KCN Sóng Thần: ViT5 hallucinate sang "song hành"; các mô hình khác thiếu dấu sắc ở "sóng" |
| 13 | `ubnd xa phuoc thai` | `ủy ban nhân dân xã phước thái` | ✅ `ủy ban nhân dân xã phước thái` | ✅ `ủy ban nhân dân xã phước thái` | ✅ `ủy ban nhân dân xã phước thái` | ❌ `ban nhân dân xã phước thái` | Viết tắt UBND xã: Arm A, E, G mở rộng chuẩn xác; BARTpho bị nuốt mất từ "ủy" |
| 14 | `158/16 binh quew` | `158/16 bình quới` | ❌ `158/16 bình quế` | ❌ `158/16 binh quế` | ✅ `158/16 bình quới` | ✅ `158/16 bình quới` | Lỗi phím thừa 'w': Mô hình nhỏ đoán mò thành "bình quế"; chỉ Pretrained hiểu đúng địa danh "Bình Quới" |
| 15 | `chung cu ha` | `chung cư hà` | ✅ `chung cư hà` | ✅ `chung cư hà` | ❌ `chung cư hạ` | ❌ `chung cư hà nội` | Truy vấn mơ hồ: BARTpho hallucinate thêm từ "nội"; ViT5 đoán nhầm dấu nặng; Scratch giữ nguyên tốt |
| 16 | `ngã 6 tahnhf` | `ngã 6 thành` | ✅ `ngã 6 thành` | ✅ `ngã 6 thành` | ✅ `ngã 6 thành` | ✅ `ngã 6 thành` | Lỗi gõ phím Telex 'f' ở cuối: Cả 4 mô hình loại bỏ ký tự rác và phục hồi đúng |
| 17 | `tttm aeon mall tan phu` | `trung tâm thương mại aeon mall tân phú` | ✅ `trung tâm thương mại aeon mall tân phú` | ❌ `trung tâm thương mại tân phú` | ❌ `... aeon mall tan phu` | ✅ `trung tâm thương mại aeon mall tân phú` | Mất từ (Truncation): Arm E nuốt mất "aeon mall"; ViT5 quên bỏ dấu; BARTpho chuẩn xác 100% |
| 18 | `dh bach khoa ha noi` | `đại học bách khoa hà nội` | ✅ `đại học bách khoa hà nội` | ✅ `đại học bách khoa hà nội` | ✅ `đại học bách khoa hà nội` | ✅ `đại học bách khoa hà nội` | Viết tắt trường ĐH: Cả 4 mô hình mở rộng và phục hồi dấu chuẩn tuyệt đối |
| | **TỔNG KẾT ĐÚNG (ACCURACY)** | | **14 / 18 (77.8%)** | **12 / 18 (66.7%)** | **12 / 18 (66.7%)** | **11 / 18 (61.1%)** | *(Theo chuẩn nhãn Canonical không thiên vị)* |

> **Bình luận chuyên sâu từ dữ liệu thực nghiệm:**
> 1. **Mô hình nhỏ (Scratch) không hề thua kém ở các tác vụ thông thường:** Với các từ viết tắt phổ biến (`bv`, `dh`, `q9`) và lỗi gõ dính phím Telex (`tahnhf` $\rightarrow$ `thành`), mô hình nhỏ đạt độ chính xác gần như tuyệt đối, thậm chí Arm A còn đạt **14/18 ca (77.8%)**, vượt cả ViT5 và BARTpho.
> 2. **Điểm yếu cốt tử của mô hình nhỏ là thiếu Tri Thức Thực Thể (World Knowledge):** Khi gặp địa danh hiếm gặp như *Bình Quới* trong `158/16 binh quew`, cả Arm A và Arm E đều tự động đoán mò thành *bình quế*. Chỉ có ViT5 và BARTpho nhờ pretraining trên kho ngữ liệu khổng lồ mới nhận diện đúng thực thể này.
> 3. **Ảo giác ngữ nghĩa (Semantic Hallucination) của Pretrained:** Mô hình lớn đôi khi tự động suy diễn quá mức: BARTpho tự ý biến `chung cu ha` thành *"chung cư hà nội"*, ViT5 biến `kcn song than` thành *"khu công nghiệp song hành"*. Điều này khẳng định không thể tin tưởng tuyệt đối vào Pretrained Teacher mà phải có cơ chế lọc (filtering) và kiểm soát an toàn trước khi chuyển giao tri thức.
---

## 3. Phân Tích Gốc Rễ: Bản Chất Khoảng Cách Năng Lực (Capability Gap)

Khoảng cách tổng thể giữa mô hình thực chiến tối ưu **Arm E (67.33%)** và Teacher Upper-bound **Arm G (73.43%)** là **6.10 pp**. Thay vì phỏng đoán cảm tính, dữ liệu từ 5 Benchmark Suites và 18 ca kiểm chứng thực tế đã bóc tách chính xác bản chất của khoảng cách này:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            BẢN ĐỒ PHÂN HÓA NĂNG LỰC: ARM E vs ARM G                         │
├───────────────────────────────┬──────────────┬──────────────┬───────────────────────────────┤
│ Nhóm bài toán (Benchmark)    │ Arm E (7.1M) │ Arm G (226M) │ Đánh giá thực nghiệm          │
├───────────────────────────────┼──────────────┼──────────────┼───────────────────────────────┤
│ 1. Plasticity (Sửa lỗi gõ)    │    43.6%     │    43.1%     │ Arm E THẮNG (+0.5 pp)         │
│ 2. Retention (Giữ từ đúng)    │    73.0%     │    73.6%     │ Tương đương (-0.6 pp)         │
│ 3. User-Centric (Hành vi)     │    66.7%     │    66.7%     │ HÒA TUYỆT ĐỐI (0.0 pp)        │
│ 4. Protection Seen (Từ quen)  │    91.0%     │    93.8%     │ Thu hẹp sát nút (-2.8 pp)     │
│ 5. Protection Held-out (Mới)  │    62.4%     │    90.0%     │ ARM G ÁP ĐẢO (-27.6 pp)       │
└───────────────────────────────┴──────────────┴──────────────┴───────────────────────────────┘
```

Từ bảng đối soát trên, 3 kết luận gốc rễ được xác lập:

1. **Khoảng cách 6.10 pp thuần túy là "Out-of-Distribution Entity Gap" (+27.6 pp ở Held-out):**
   - Ở toàn bộ các tác vụ biến đổi quy luật (gõ nhầm dấu Telex, dính phím, viết tắt phổ thông, giữ nguyên từ chuẩn), mô hình nhỏ 7.1M tham số đạt độ chính xác ngang ngửa hoặc vượt trội mô hình 226M tham số.
   - Mô hình nhỏ **chỉ thất bại khi gặp các thực thể tên riêng chưa từng xuất hiện trong tập huấn luyện 300K** (như tên chung cư, toà nhà, đường phố ngoại thành, thương hiệu mới). Điều này chứng minh mô hình nhỏ không hề thiếu năng lực suy luận ngôn ngữ (reasoning capacity) mà chỉ **thiếu tri thức sự kiện (World Knowledge)**.

2. **Tri thức địa lý & POI là "Không Gian Đóng" (Closed-World Knowledge):**
   - ViT5 hay BARTpho nhận diện đúng các thực thể lạ vì chúng đã được nạp hàng chục GB văn bản tổng quát (Wikipedia, báo chí) trong giai đoạn pretraining.
   - Tuy nhiên, trong nghiệp vụ gọi xe và giao vận công nghệ, thế giới thực thể không phải là vô tận: danh sách đơn vị hành chính (tỉnh/thành, quận/huyện, phường/xã), danh mục POI trọng điểm (sân bay, bệnh viện, trường học, TTTM, KCN) và hệ thống tên đường tại Việt Nam là **hữu hạn, có cấu trúc và sẵn có trong cơ sở dữ liệu bản đồ (Gazetteer)**. Do đó, việc duy trì một mô hình 226M tham số chỉ để "nhớ hộ" các tên riêng này là cực kỳ lãng phí tài nguyên.

3. **Trade-off Nguy Cơ: Ảo Giác Ngữ Nghĩa (Hallucination) vs Bảo Thủ (Conservative):**
   - Pretrained model có xu hướng "tự biên tự diễn" khi gặp truy vấn cụt hoặc mơ hồ (như biến `chung cu ha` thành *"chung cư hà nội"*, `kcn song than` thành *"khu công nghiệp song hành"*). Nguy cơ này có thể gây điều hướng cuốc xe sai hàng chục cây số.
   - Ngược lại, mô hình nhỏ học từ đầu (Scratch) có hành vi bảo thủ và trung thực với ngữ cảnh đầu vào hơn, rất hiếm khi tự bịa thêm các token không liên quan.

---

## 4. Ràng Buộc Thiết Kế & Trade-off Hạ Tầng Production

Để một mô hình QU được đưa vào phục vụ trực tiếp hàng triệu người dùng thời gian thực, độ chính xác (Accuracy) không phải là tiêu chí duy nhất. Các ràng buộc kỹ thuật dưới đây là **bắt buộc và không thể nhân nhượng**:

| Tiêu chuẩn vận hành | Ngưỡng cam kết (SLA) | Ý nghĩa nghiệp vụ & kỹ thuật |
|:---|:---|:---|
| **Latency P50** | **< 10.0 ms** (CPU INT8, 1 vCPU) | Chạy đồng bộ (in-line) trước bước Retrieval; không gây giật lag trải nghiệm gõ phím. |
| **Latency P99** | **< 20.0 ms** (CPU INT8) | Kiểm soát đuôi trễ (tail latency) vào giờ cao điểm, bảo vệ bộ đệm server. |
| **Kích thước mô hình** | **< 50 MB** trên đĩa | Triển khai gọn nhẹ trên các container Pods, scale-out tức thì không nghẽn I/O. |
| **Throughput** | **> 500 QPS / core** | Tiết kiệm chi phí cụm server khi traffic đột biến. |
| **Phương thức giải mã** | **Greedy Decoding (Beam=1)** | Đảm bảo tính tất định (deterministic), loại bỏ chi phí tính toán nhiều nhánh. |
| **Bảo toàn No-op** | **Tỷ lệ giữ nguyên ≥ 99.5%** | Tuyệt đối không làm biến dạng các truy vấn người dùng đã gõ đúng. |
| **Cơ chế Fallback** | **Bypass về Raw Query** | Nếu module timeout (>20ms) hoặc crash, trả ngay input gốc cho search engine. |
| **Chi phí huấn luyện** | **≤ 2 engineer-weeks** | Khả năng retrain liên tục định kỳ tuần/tháng theo luồng dữ liệu mới. |

### Đánh Giá Trade-off Thực Tế: Vì Sao Không Thể Deploy Pretrained?

Đối chiếu các ứng viên với bảng tiêu chuẩn SLA hạ tầng:

```
┌───────────────────────────┬──────────────┬──────────────┬──────────────┬────────────────────────┐
│ Mô hình                   │ P50 Latency  │ Kích thước   │ Vốn phần cứng│ Kết luận triển khai    │
├───────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────────────┤
│ Arm G (ViT5-base)         │   88.75 ms   │    ~900 MB   │ GPU / Multi-C│ VI PHẠM SLA (Trễ 9×)   │
│ Arm H (BARTpho-syllable)  │   ~45.00 ms  │    ~530 MB   │ High CPU     │ VI PHẠM SLA (Trễ 4.5×) │
│ Arm A (2E/1D d128)        │    2.38 ms   │     15.8 MB  │ Low CPU (1C) │ Đạt chuẩn xuất sắc     │
│ Arm E (2E/2D d128)        │    5.71 ms   │     27.1 MB  │ Low CPU (1C) │ SWEET SPOT TOÀN DIỆN   │
└───────────────────────────┴──────────────┴──────────────┴────────────────────────┘
```

> **Quyết định kiến trúc:**
> 1. **Loại bỏ triển khai trực tiếp ViT5-base và BARTpho:** Dù đạt độ chính xác cao hơn 6.10 pp, việc làm tăng độ trễ thêm ~80ms là cái giá không thể chấp nhận đối với hệ thống tìm kiếm thời gian thực.
> 2. **Xác lập Arm E làm Kiến Trúc Nền Tảng (Production Base):** Với P50 chỉ **5.71 ms** (đạt 57% ngưỡng trần SLA) và dung lượng chỉ **27.1 MB**, Arm E là khung sườn hoàn hảo.
> 3. **Chiến lược lấp đầy khoảng cách 6.10 pp:** Không giải quyết bằng cách tăng model size hay nhồi thêm layers, mà giải quyết tập trung bằng **Targeted Data Augmentation** (bơm tri thức Gazetteer vào trainset) và **Distillation có chọn lọc** từ Teacher Pretrained.

---

## 5. Giả Thuyết Khoa Học & Không Gian Giải Pháp (Core Hypotheses)

> **Chiến lược cốt lõi:** Thay vì đánh đổi chi phí hạ tầng để triển khai các mô hình Pretrained khổng lồ (vi phạm SLA trễ gấp 9 lần), chúng tôi đề xuất giữ vững kiến trúc tối ưu siêu nhẹ **Arm E (2E/2D d128)** và chia tách chiến lược xử lý dữ liệu làm 2 giai đoạn: **(1) Huấn luyện nền tảng bằng Dữ liệu Tổng hợp 4M (Synthetic Data Engine)** và **(2) Khai thác 3.55M log zero-click thực tế thông qua Teacher Pseudo-Labeling hoặc Huấn luyện Không giám sát (Unsupervised DAE)**.

Không gian giải pháp được xây dựng trên 3 giả thuyết khoa học có khả năng kiểm chứng độc lập:

1. **Giả thuyết Nền Tảng Tổng Hợp (Synthetic Base Hypothesis):**
   - *Luận điểm:* Mô hình nhỏ 7.1M tham số hoàn toàn có thể nắm vững quy luật ngữ âm tiếng Việt và cấu trúc hành chính nếu được cung cấp tập dữ liệu tổng hợp bao phủ toàn diện 63 tỉnh thành kết hợp bộ giải lập gõ phím.
   - *Kiểm chứng:* Tập dữ liệu 4,000,000 cặp câu tổng hợp (Synthetic Data Recipe) giúp mô hình đạt độ chính xác >91% trên từ vựng quen thuộc (Seen) và duy trì tỷ lệ bảo tồn câu sạch $\ge 99.5\%$.

2. **Giả thuyết Điểm Cân Bằng Kiến Trúc (Sweet Spot Architecture Hypothesis):**
   - *Luận điểm:* Cấu hình **Arm E (2 Encoder / 2 Decoder, $d_{model}=128$)** là điểm cân bằng hoàn hảo giữa năng lực biểu diễn và độ trễ. 2 lớp Decoder cung cấp vừa đủ năng lực tự hồi quy (autoregression) để giải quyết các lỗi tổ hợp đa tầng (vừa dính phím telex, vừa viết tắt tiền tố), vượt trội cấu hình 1 Decoder (+1.68 pp) nhưng vẫn duy trì độ trễ CPU P50 = **5.71 ms** (đáp ứng trần SLA < 10 ms).

3. **Giả thuyết Khai Thác Log Chưa Gán Nhãn (Unlabeled In-the-Wild Log Hypothesis):**
   - *Luận điểm:* 3.55M log zero-click là dữ liệu thô không có nhãn ground-truth chuẩn. Việc đưa trực tiếp vào tập train supervised là bất khả thi.
   - *Kiểm chứng:* Kho log này sẽ được khai thác ở giai đoạn tiếp theo theo 2 hướng:
     - **Hướng A (Teacher Pseudo-labeling):** Dùng Teacher (ViT5-base) để tự động gán nhãn mục tiêu cho log zero-click, sau đó lọc kiểm duyệt (confidence filtering) nhằm loại bỏ hallucination trước khi chuyển giao cho Arm E.
     - **Hướng B (Unsupervised DAE):** Huấn luyện tái tạo nhiễu trực tiếp trên phân phối truy vấn thực tế của 3.55M log mà không cần nhãn con người, giúp mô hình thích ứng mạnh mẽ với phương ngữ và thói quen gõ tắt ngoài thực địa.

---

## 6. Kiến Trúc Giải Pháp & Quy Trình Huấn Luyện (Training & Serving Pipeline)

### 6.1. Tích Hợp Vào Pipeline Tìm Kiếm Sản Phẩm

Trong hệ thống định vị và đặt xe của Xanh SM, **ReparoS** đóng vai trò là **tầng tiền xử lý đầu tiên (First-pass Query Understanding & Autocorrect)**. Vai trò sống còn của nó là chuẩn bị dữ liệu đầu vào sạch sẽ cho bộ bóc tách thực thể ngữ nghĩa (Semantic Parser):

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               PIPELINE TÌM KIẾM THỜI GIAN THỰC                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   Truy vấn người dùng (Raw Query): "86 xo viet nghe tinh p19", "bv cho ray"            │
│      │                                                                                  │
│      ▼                                                                                  │
│  ┌───────────────────────────────────────────────────────────┐                          │
│  │ 1. Tiền xử lý: Unicode NFC + Lowercase (Chống vỡ mã byte) │                          │
│  └───────────────────────────┬───────────────────────────────┘                          │
│                              ▼                                                          │
│  ┌───────────────────────────────────────────────────────────┐                          │
│  │ 2. ReparoS Serving Engine (Arm E: 2E/2D d128 - CTranslate2│  ← TRỌNG TÂM ĐỀ ÁN       │
│  │    • Kích thước: 7.1M params (INT8: 27.1 MB trên đĩa)     │  • Latency P50: 5.71 ms  │
│  │    • Giải mã: Greedy (Beam=1), Repetition Penalty = 1.15  │  • Latency P99: 11.2 ms  │
│  │    • Output: "86 xô viết nghệ tĩnh phường 19",            │  • Throughput: > 600 QPS │
│  │              "bệnh viện chợ rẫy"                          │                          │
│  └───────────────────────────┬───────────────────────────────┘                          │
│                              ▼                                                          │
│   Truy vấn đã chuẩn hoá tiền tố & dấu thanh (Normalized Query)                          │
│      │                                                                                  │
│      ▼                                                                                  │
│  ┌───────────────────────────────────────────────────────────┐                          │
│  │ 3. Semantic Parser (NER / Slot-Filling / Entity Extractor)│  ← Downstream kế thừa    │
│  │    • Bóc tách chuẩn xác:                                  │                          │
│  │      [HOUSE_NUM: 86] [STREET: xô viết nghệ tĩnh]          │                          │
│  │      [WARD: phường 19] [POI: bệnh viện chợ rẫy]           │                          │
│  │    *(Nếu không chuẩn hoá 'p19' → 'phường 19' hay 'bv',    │                          │
│  │      Semantic Parser sẽ trượt slot và gãy toàn bộ search)*│                          │
│  └───────────────────────────┬───────────────────────────────┘                          │
│                              ▼                                                          │
│  ┌───────────────────────────────────────────────────────────┐                          │
│  │ 4. Geocoding & Spatial Search Engine (OSM / Map Index)    │  → Toạ độ ghim đón khách │
│  └───────────────────────────────────────────────────────────┘                          │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

* **Phạm vi tác động (Blast Radius):** Độc lập hoàn toàn, chỉ thay đổi artifact mô hình trong module QU. Giữ nguyên 100% downstream Semantic Parser và hạ tầng serving.

---

### 6.2. Chiến Lược Dữ Liệu Huấn Luyện (Data Engineering & Recipe)

#### 6.2.1. Phân Phối Dữ Liệu Cơ Sở 4M: Ánh Xạ Trực Tiếp Từ 3.55M Log Thực Tế
Phân phối dữ liệu huấn luyện không thể thiết lập tùy tiện, mà bắt buộc phải **phản ánh trung thực các tỷ lệ lỗi phát hiện từ 3.55 triệu log zero-click** (Mục 1.2):

* *Hiện thực log:* **40.5%** query gắn liền với số và đơn vị hành chính; **73.2%** query đã có dấu nhưng gõ tắt hoặc dính phím Telex; **77.5%** là Long-tail; và yêu cầu sống còn là **Clean Preservation $\ge 99.5\%$**.
* *Công thức phối trộn 4 Tầng (4-Tier Synthetic Data Recipe quy mô 4,000,000 cặp câu):*

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│              CƠ CẤU PHỐI TRỘN DỮ LIỆU HUẤN LUYỆN NỀN TẢNG (4,000,000 CẶP CÂU)           │
├──────────────────────┬─────────┬──────────────────────────────────┬─────────────────────┤
│ Tầng dữ liệu         │ Tỷ lệ   │ Ánh xạ từ thực tế 3.55M Log      │ Bản chất bài toán   │
├──────────────────────┼─────────┼──────────────────────────────────┼─────────────────────┤
│ Layer 1: Geographic  │   30%   │ Khớp với 40.5% log dính số &     │ Mở rộng tiền tố,    │
│ & Address Symbols    │ (1.20M) │ thói quen viết tắt hành chính    │ tách ranh giới số   │
├──────────────────────┼─────────┼──────────────────────────────────┼─────────────────────┤
│ Layer 2: Mobile Typo │   30%   │ Khớp với 73.2% log có dấu nhưng  │ Sửa dính phím telex,│
│ & IME Telex / VNI    │ (1.20M) │ dính lỗi gõ phím di động vội vã  │ nhảy phím lân cận   │
├──────────────────────┼─────────┼──────────────────────────────────┼─────────────────────┤
│ Layer 3: Replay      │   30%   │ Neo giữ tỷ lệ không sửa bậy      │ Chống over-correct, │
│ Anchor & Clean No-op │ (1.20M) │ Clean Preservation ≥ 99.5%       │ giải quyết bập bênh │
├──────────────────────┼─────────┼──────────────────────────────────┼─────────────────────┤
│ Layer 4: DAE Noise   │   10%   │ Khớp với 77.5% Long-tail         │ Tăng tính dẻo dai   │
│ (Denoising Auto-Enc) │ (0.40M) │ các biến thể micro-POI lạ hoắc   │ (plasticity)        │
└──────────────────────┴─────────┴──────────────────────────────────┴─────────────────────┘
```

#### Chi tiết kỹ thuật và bài học từ thực nghiệm:
1. **Layer 1 (30% - Geographic & Address Symbols):**
   - Bơm toàn bộ danh mục cây địa giới 63 tỉnh thành từ OpenStreetMap.
   - Chuẩn hoá triệt để tiền tố hành chính và giao thông: `đ / dg` $\rightarrow$ `đường`, `p` $\rightarrow$ `phường`, `q` $\rightarrow$ `quận`, `tp` $\rightarrow$ `thành phố`, `ubnd`, `kcn`, `bv`, `dh`, `bx`, `tttm`.
   - **Tách ranh giới từ dính số xác định (Determined Number Boundary):** Xử lý dính số cơ học `p19` $\rightarrow$ `phường 19`, `q.1 / q1` $\rightarrow$ `quận 1`, `ngõ325` $\rightarrow$ `ngõ 325`, `toà ct1` $\rightarrow$ `toà ct1`.
   - *Nguyên tắc loại trừ ảo giác (Guardrail):* Tuyệt đối **không đưa vào các ca nén số nhà thiếu thông tin** (như ép `32322` $\rightarrow$ `323/22`). Thực nghiệm chứng minh đây là bài toán thiếu thông tin trầm trọng (ill-posed), ép mô hình học thuộc lòng sẽ gây ảo giác biến dạng số nhà chuẩn (`16 lê lợi` bị sửa nhầm thành `1/6 lê lợi`).
2. **Layer 2 (30% - Mobile Typo & IME Synthesizer):**
   - Giải lập bộ gõ di động: dính phím Telex (`tahnhf` $\rightarrow$ `thành`, `hoangf` $\rightarrow$ `hoàng`), lỗi trượt phím lân cận QWERTY (`qura` $\rightarrow$ `qua`), mất dấu thanh toàn phần hoặc bán phần do người dùng gõ ngắt quãng.
3. **Layer 3 (30% - Replay Anchor & Clean No-op):**
   - Tỷ lệ 30% (1.2M cặp `X → X`) là bắt buộc để giải quyết **Hiện tượng bập bênh Gradient (Seesaw Effect)**: Nếu tỷ lệ này dưới 25%, mỗi khi tăng khả năng sửa lỗi, tỷ lệ bảo toàn câu sạch lại bị kéo tụt xuống dưới 90% (False Correction Rate bùng nổ >10%).
   - Tầng này hoạt động như một "mỏ neo", dạy mô hình cơ chế copy tuyệt đối các truy vấn người dùng đã gõ đúng.
4. **Layer 4 (10% - Denoising Auto-Encoder):**
   - Masking 15% token, xóa ký tự ngẫu nhiên nhằm rèn luyện Encoder hiểu sâu ngữ cảnh của 77.5% truy vấn Long-tail.

---

#### 6.2.2. Lộ Trình Khai Thác 3.55M Log Zero-Click (In-the-Wild Unlabeled Data)
3.55M truy vấn zero-click là dữ liệu thô, phản ánh chính xác hành vi người dùng thật nhưng **hoàn toàn chưa có nhãn chuẩn hoá**. Ở giai đoạn tiếp theo, kho dữ liệu này sẽ được khai thác thông qua hai kỹ thuật:

```
                                  ┌──────────────────────────────┐
                                  │   3.55M Log Zero-Click Thô   │
                                  │    (Unlabeled Real Queries)  │
                                  └──────────────┬───────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
           [HƯỚNG 1: TEACHER PSEUDO-LABELING]                [HƯỚNG 2: UNSUPERVISED DAE]
           • ViT5-base dự đoán nhãn mục tiêu                 • Thêm nhiễu ngẫu nhiên vào log
           • Bộ lọc tin cậy (Loại bỏ hallucination)          • Huấn luyện mô hình khôi phục log gốc
           • Chuyển giao nhãn sạch cho Arm E                 • Nắm bắt phân phối từ vựng thực địa
```

1. **Hướng 1 — Teacher Pseudo-Labeling & Selective Distillation (Bán giám sát):**
   - Sử dụng Teacher lớn (ViT5-base / BARTpho) chạy offline để gán nhãn mục tiêu cho các truy vấn zero-click.
   - **Bộ lọc kiểm duyệt (Safety & Consistency Filter):** Loại bỏ ngay các ca mà Teacher sinh ảo giác (hallucination, tự ý bịa thêm từ như `chung cu ha` $\to$ `chung cư hà nội`). Chỉ giữ lại các cặp câu có độ tin cậy cao để fine-tune tiếp cho học sinh Arm E.
2. **Hướng 2 — Unsupervised Denoising (Tự giám sát / Không cần nhãn):**
   - Đưa trực tiếp raw queries vào quy trình huấn luyện không giám sát: làm méo nhẹ truy vấn (dropout, mask ký tự) và ép mô hình giải mã tái tạo lại đúng truy vấn gốc. Kỹ thuật này giúp mô hình thích nghi sâu với phân phối ngôn ngữ gõ vội của người dùng thực tế mà không tốn công sức gán nhãn thủ công.

---

### 6.3. Quy Trình & Tham Số Huấn Luyện (Training Regime & Hyperparameters)

Quá trình huấn luyện mô hình được thiết kế chặt chẽ nhằm triệt tiêu các lỗi vận hành đã được kiểm chứng qua thực nghiệm:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        THÔNG SỐ HUẤN LUYỆN CHUẨN HOÁ (TRAINING REGIME)                  │
├──────────────────────────┬────────────────────────────┬─────────────────────────────────┤
│ Tham số / Cấu hình       │ Giá trị thiết lập          │ Rationale kỹ thuật kiểm chứng   │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ **Bộ Tokenizer**         │ SentencePiece Unigram      │ Tối ưu biểu diễn âm tiết tiếng  │
│                          │ Kích thước Vocab: 12,000   │ Việt; bắt buộc Lowercase trước  │
│                          │ (Lowercase input)          │ khi tokenize để chống vỡ byte   │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ **Hàm mất mát (Loss)**   │ Label Smoothed Cross-Ent   │ Triệt tiêu over-confidence,     │
│                          │ $\epsilon = 0.1$           │ tăng tính khái quát hoá         │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ **Bộ tối ưu (Optimizer)**│ AdamW                      │ $\beta_1 = 0.9, \beta_2 = 0.98$ │
│                          │ Weight Decay = 0.01        │ Gradient Clipping = 1.0         │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ **Learning Rate**        │ Peak: $5 \times 10^{-4}$   │ Warmup 4,000 steps              │
│                          │ Min: $1 \times 10^{-5}$    │ Cosine Annealing Decay          │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ **Kích thước Batch**     │ 256 sequences / batch      │ Gradient Accumulation = 2       │
├──────────────────────────┼────────────────────────────┼─────────────────────────────────┤
│ **Serving Repetition**   │ Repetition Penalty = 1.15  │ Triệt tiêu 100% lỗi lặp từ      │
│                          │ Beam Size = 1 (Greedy)     │ (`vinfast vinfast` → `vinfast`) │
└──────────────────────────┴────────────────────────────┴─────────────────────────────────┘
```

#### Phương Pháp Huấn Luyện Bền Vững (Curriculum Learning with Dynamic Replay Anchor)

Huấn luyện một mạng Seq2Seq nhỏ (7.1M tham số) trên bài toán chuẩn hoá truy vấn đối mặt với một thách thức toán học kinh điển: **Xung đột Gradient (Gradient Conflict) và Hiện tượng Quên thảm hoạ (Catastrophic Forgetting)**. 
- Nếu đưa toàn bộ dữ liệu tổ hợp đa lỗi vào ngay từ đầu, gradient hỗn loạn sẽ phá vỡ cơ chế sao chép (copy mechanism), khiến mô hình trở nên "hung hăng" và sửa sai cả những câu người dùng đã gõ đúng.
- Nếu chia giai đoạn tuần tự mà thay thế hoàn toàn tập dữ liệu, mô hình sẽ quên sạch các quy tắc đơn giản đã học ở giai đoạn trước, kéo tỷ lệ Clean Preservation tụt dốc thảm hại.

Để giải quyết triệt để, chúng tôi áp dụng chiến lược **Curriculum Learning 3 Giai Đoạn kết hợp Bộ Đệm Neo Giữ Động (Dynamic Replay Anchor Buffer)** trên tổng số 50,000 steps (~18 giờ huấn luyện):

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       TIẾN TRÌNH HUẤN LUYỆN CURRICULUM WITH DYNAMIC REPLAY ANCHOR                            │
├──────────────────────────┬─────────────────────────────┬────────────────────────────────────────────────────┤
│ Giai đoạn huấn luyện     │ Tỷ lệ phối trộn Batch (Mix) │ Trọng tâm kỹ thuật & Mục tiêu hội tụ               │
├──────────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────┤
│ **Giai đoạn 1 (Phase I)** │ • 50% Layer 3 (Clean No-op) │ **Thiết lập "Hàm Đồng Nhất" $f(x) = x$ & Sửa Đơn**│
│ Step 0 – 20,000          │ • 35% Layer 2 (Typo Telex)  │ • Học cơ chế copy tuyệt đối cho câu đúng           │
│ (Nền tảng & Sửa lỗi đơn) │ • 15% Layer 1 (Prefix cơ bản│ • Sửa lỗi gõ 1 ký tự (dính telex, rụng dấu thanh)  │
│                          │                             │ • Cổng kiểm soát (Gate 1): Clean Preserv ≥ 99.2%   │
├──────────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────┤
│ **Giai đoạn 2 (Phase II)**│ • 35% Replay Anchor (L2, L3)│ **Mở Rộng Cấu Trúc Địa Giới & Ranh Giới Số**      │
│ Step 20,000 – 40,000     │ • 45% Layer 1 (Address/Num) │ • Neo 35% dữ liệu cũ chống quên thảm họa           │
│ (Cấu trúc & Tách số)     │ • 20% Layer 4 (DAE nhẹ)     │ • Nạp tiền tố hành chính (`đ/p/q`) & tách số `p19` │
│                          │                             │ • Cổng kiểm soát (Gate 2): Không hallucinate số    │
├──────────────────────────┼─────────────────────────────┼────────────────────────────────────────────────────┤
│ **Giai đoạn 3 (Phase III)│ • 30% Replay Anchor Buffer  │ **Tôi Luyện Tổ Hợp Đa Lỗi & Annealing Hội Tụ**     │
│ Step 40,000 – 50,000     │ • 50% Compositional Errors  │ • Giải quyết truy vấn "siêu lỗi" đa tầng           │
│ (Tổ hợp đa tầng & Đuôi)  │ • 20% Long-tail DAE Noise   │ • Cosine decay learning rate về $1 \times 10^{-5}$  │
│                          │                             │ • Cổng nghiệm thu (Final): Mean ≥ 67%, Clean ≥ 99.5%│
└──────────────────────────┴─────────────────────────────┴────────────────────────────────────────────────────┘
```

#### Chi tiết cơ chế vận hành từng giai đoạn:

1. **Giai đoạn 1 (Step 0 – 20,000 | Warm-up & Foundation Anchor):**
   - *Mục tiêu toán học:* Ép hàm mất mát Cross-Entropy tối ưu hoá việc học trọng số Attention sao cho các ma trận chéo (diagonal self-attention) có trọng số cao nhất khi gặp các token từ điển. Điều này biến mô hình thành một bộ lọc sao chép bảo thủ (conservative identity mapping).
   - *Dữ liệu:* Phối trộn 50% cặp đồng nhất `(X → X)` và 35% lỗi gõ phím 1 biến thể đơn lẻ (`tahnhf` $\rightarrow$ `thành`).
   - *Điều kiện chuyển giai đoạn (Checkpoint Gate 1):* Đánh giá trên tập validation: Tỷ lệ bảo tồn câu sạch (Clean Preservation) bắt buộc phải đạt **$\ge 99.2\%$** và độ chính xác sửa lỗi telex đơn đạt **$\ge 92\%$**. Nếu chưa đạt, tiếp tục train warm-up thêm 5,000 steps.

2. **Giai đoạn 2 (Step 20,000 – 40,000 | Structured Complexity & Number Separation):**
   - *Cơ chế Replay Buffer:* Tuyệt đối không thay thế dữ liệu! Hệ thống giữ cố định **35% Replay Buffer** (gồm dữ liệu sạch Layer 3 và lỗi phím Layer 2) trong mỗi mini-batch. Dữ liệu này đóng vai trò như một "lực cản gradient" (gradient anchor) ngăn chặn các cập nhật trọng số làm biến dạng các quy tắc nền tảng đã học.
   - *Nạp bài toán cấu trúc:* 45% batch được cấp cho Layer 1: chuẩn hoá cấu trúc địa chỉ OpenStreetMap (`đ` $\rightarrow$ `đường`, `p` $\rightarrow$ `phường`, `ubnd`, `bv`) và xử lý dính phím số xác định (`p19` $\rightarrow$ `phường 19`, `q.1` $\rightarrow$ `quận 1`, `ngõ325` $\rightarrow$ `ngõ 325`).
   - *Chốt chặn kiểm soát (Safety Gate 2):* Kiểm tra chéo trên tập test số nhà. Nếu phát hiện mô hình bắt đầu tự ý chèn dấu xuyệt vào số nhà không liên quan (`16 lê lợi` $\rightarrow$ `1/6 lê lợi`), ngay lập tức kích hoạt bộ lọc loại bỏ toàn bộ các mẫu nén số ill-posed khỏi batch.

3. **Giai đoạn 3 (Step 40,000 – 50,000 | Compositional Annealing & Long-tail Adaptation):**
   - *Thách thức tổ hợp đa lỗi:* Người dùng thực tế không bao giờ gõ một loại lỗi đơn lẻ. Họ gõ tắt tiền tố, kèm số nhà, dính phím telex và sai thanh điệu cùng một lúc (*"86 xo viet nghe tinh p19 binh thanh"*).
   - *Phối trộn dữ liệu khó:* Nạp 50% mẫu tổ hợp (Compositional Errors) kết hợp 20% DAE Masking/Permutation ngẫu nhiên trên các địa danh Long-tail.
   - *Chiến lược suy giảm Learning Rate (Cosine Annealing):* Giảm dần learning rate từ $5 \times 10^{-4}$ về $1 \times 10^{-5}$ trong 10,000 steps cuối cùng. Việc giảm learning rate trong khi nạp dữ liệu khó giúp mô hình "đóng băng nhẹ" các trọng số biểu diễn cốt lõi, chỉ tinh chỉnh các head suy luận cao tầng mà không làm xáo trộn nền móng.

#### Hệ Thống Giám Sát Thoái Lui Trực Tuyến (Online Regression Monitor):
Trong suốt tiến trình 50,000 steps, một tiến trình đánh giá độc lập (Evaluator Worker) chạy ngầm kiểm tra mô hình sau mỗi 2,000 steps trên tập cố định **1,000 câu truy vấn sạch (Clean Test Suite)** và **5 Frozen Suites**:
* **Quy tắc Rollback tự động:** Nếu số ca sửa sai trên truy vấn sạch vượt quá **5 ca / 1,000 câu** (tức False Correction Rate > 0.5% hay Clean Preservation < 99.5%), checkpoint đó bị đánh dấu fail. Hệ thống tự động kích hoạt cơ chế tăng tỷ trọng Replay Anchor lên **45%** ở các step tiếp theo để kéo mô hình về trạng thái an toàn.
* **Quy tắc Early Stopping:** Huấn luyện sẽ dừng sớm nếu sau 3 lần kiểm tra liên tiếp (6,000 steps) chỉ số Mean Accuracy trên 5 suites không cải thiện vượt ngưỡng 0.1 pp và tỷ lệ Clean không suy giảm.

---

## 7. Kế Hoạch Thực Nghiệm & Lộ Trình Triển Khai (Engineering Roadmap)

Lộ trình triển khai giải pháp được cấu trúc thành các giai đoạn nối tiếp có cổng kiểm soát chất lượng (Quality Gate) nghiêm ngặt:

| Giai đoạn | Trọng tâm kỹ thuật | Đầu ra bàn giao (Deliverables) | Tiêu chí nghiệm thu (Quality Gate) |
|---|---|---|---|
| **Phase 0: Chẩn đoán & Probing** *(ĐÃ HOÀN THÀNH)* | Thực nghiệm Scaling Law trên 6 Arms Scratch (A–F) và probe 2 Pretrained (G, H) trên 5 Frozen Suites. | Báo cáo thực nghiệm chuyên sâu; xác lập **Arm E (2E/2D d128)** là Sweet Spot; chỉ rõ gap Held-out +27.6 pp. | Phân tích rõ nguyên nhân gốc rễ và xác định không gian giải pháp khả thi. |
| **Phase 1: Serving Runtime & Hạ Tầng** | Đóng gói Arm E sang định dạng CTranslate2 INT8; tích hợp tiền xử lý Lowercase + NFC và Repetition Penalty = 1.15. | Module ReparoS Serving Engine độc lập, sẵn sàng cắm trước Semantic Parser. | **P50 < 6.0 ms, P99 < 12.0 ms** trên CPU; kích thước < 30 MB; triệt tiêu 100% lỗi lặp từ và vỡ mã byte. |
| **Phase 2: Nền Tảng Dữ Liệu Tổng Hợp 4M** | Huấn luyện Arm E trên 4-Tier Synthetic Recipe (30% Address, 30% Typo, 30% Replay Anchor, 10% DAE) theo Curriculum 3 giai đoạn. | Checkpoint Arm E Base V3+ 50,000 steps (~18h huấn luyện GPU). | **Clean Preservation $\ge 99.5\%$**; Mean Accuracy 5 suites $\ge 67.5\%$; bảo toàn quy luật số nhà chuẩn. |
| **Phase 3: Khai Thác 3.55M Log Zero-Click** | Khai thác kho log thô qua 2 hướng: (1) Teacher Pseudo-labeling (lọc an toàn) và (2) Unsupervised DAE pretraining. | Tập dữ liệu bán giám sát chuẩn hóa; Checkpoint Arm E Domain-Adapted. | **Protection Held-out Accuracy nâng từ 62.4% lên $\ge 70.0\%$**; Zero-click Canonical $\ge 72\%$. |
| **Phase 4: Tầng Lai Ghép Gazetteer & Cache** | Thiết lập bộ đệm tri thức tĩnh (Static Gazetteer Cache) cho top-500 POI / thương hiệu chuỗi hiếm gặp ngoài đời. | Sub-module Fast Cache Lookup (độ trễ < 0.2ms) trước khi gọi nơ-ron. | Độ chính xác trên danh mục P0 POI / Chuỗi thương hiệu đạt **$\ge 99.0\%$**. |
| **Phase 5: Kiểm Thử Cổng Quyết Định & Distillation** | Đánh giá đối soát toàn diện trên 5 Benchmark Suites và 18 ca Canonical Zero-Click so với ViT5-base. | Quyết định đóng băng mô hình hoặc kích hoạt Knowledge Distillation có chọn lọc. | **Cổng quyết định:** Nếu gap < 2.0 pp $\rightarrow$ **DEPLOY PRODUCTION**; nếu gap $\ge 2.0$ pp $\rightarrow$ Kích hoạt Selective Distillation. |

---

## 8. Đánh Giá & Tiêu Chí Thành Công

Để đảm bảo giải pháp giải quyết tận gốc 3 nỗi đau sản phẩm nêu tại Mục 1, hệ thống tiêu chí thành công được phân rã theo 3 trụ cột kỹ thuật và 1 trụ cột tác động sản phẩm trực tuyến:

### 8.1 Trụ Cột 1 — Năng Lực Chuẩn Hóa & Giảm Drop-Off (Accuracy Metrics)

| Chỉ số kỹ thuật | Ý nghĩa & Phương pháp đo | Hiện trạng Arm E | Mục tiêu hoàn thành | Ý nghĩa sản phẩm |
|---|---|:---:|:---:|---|
| **Mean Accuracy (5 Suites)** | Tỷ lệ Exact Match trên toàn bộ 3,100 câu kiểm thử frozen | 67.33% | **$\ge 70.0\%$** | Nâng cao chất lượng hiểu ngôn ngữ toàn diện |
| **Brand Held-out Accuracy** | Tỷ lệ nhận diện đúng tên riêng / thương hiệu chưa từng thấy (500 câu) | 62.40% | **$\ge 70.0\%$** *(ViT5: 90.0%)* | Giảm tỷ lệ người dùng bỏ tìm kiếm khi gõ địa điểm mới |
| **Mở rộng viết tắt hành chính** | Mở rộng chuẩn xác các tiền tố `đ/p/q`, `ubnd`, `bv`, `dh`, `kcn` | ~70.0% | **$\ge 75.0\%$** | Mở khóa 100% năng lực bóc tách cho Semantic Parser |
| **Zero-Click Accuracy (Canonical)** | Độ chính xác trên 18 ca lỗi người dùng thực tế từ search logs | 66.67% *(12/18)* | **$\ge 72.2\%$** *(≥ 13/18)* | Trực tiếp xử lý các truy vấn từng làm khách hàng thất vọng |
| **Correction Precision** | $TP / (TP + FP)$ — Khi đã sửa thì phải sửa đúng | ~94.0% | **$\ge 97.0\%$** | Tránh gây ức chế vì sửa sai ý định người dùng |

### 8.2 Trụ Cột 2 — An Toàn Trải Nghiệm & Chống Thoái Lui (Safety Bounds)

| Chỉ số an toàn | Định nghĩa & Ngưỡng kiểm soát | Hiện trạng | Mục tiêu cam kết | Rủi ro nếu vi phạm |
|---|---|:---:|:---:|---|
| **Tỷ lệ bảo tồn câu sạch (No-op)** | Tỷ lệ giữ nguyên 100% khi người dùng gõ câu đúng chuẩn | 90.2% | **$\ge 99.5\%$** *(False Corr < 0.5%)* | Nguy cơ "sửa lợn lành thành lợn què" phá hỏng truy vấn chuẩn |
| **Số ca Regression so với Arm A** | Số câu Arm A đoán đúng mà mô hình mới bị đoán sai | 12 ca / 1.000 câu | **$\le 5$ ca / 1.000 câu** | Thoái lui các tính năng cơ bản đã chạy tốt |
| **Số nhà & Ranh giới số** | Tuyệt đối không tự ý chèn dấu xuyệt vào số nhà thông thường | Kẹt ở 26% (ill-posed) | **Triệt tiêu 100% ảo giác số** | Sai số nhà $\rightarrow$ tài xế đón nhầm điểm $\rightarrow$ hủy cuốc |

### 8.3 Trụ Cột 3 — Hiệu Năng & Cam Kết Hạ Tầng (Infrastructure SLA)

| Chỉ số vận hành | Môi trường đo đạc | Trần cam kết (SLA) | Kết quả thực tế Arm E | Đánh giá |
|---|---|:---:|:---:|:---:|
| **Latency P50** | Single query, CPU 1-core (x86_64), CTranslate2 INT8 | **< 10.0 ms** | **5.71 ms** | **ĐẠT (Dư 43% ngân sách)** |
| **Latency P90** | Cùng môi trường đo lường | **< 15.0 ms** | **8.42 ms** | **ĐẠT** |
| **Latency P99** | Cùng môi trường đo lường (kiểm soát đuôi trễ giờ cao điểm) | **< 20.0 ms** | **11.20 ms** | **ĐẠT (An toàn tuyệt đối)** |
| **Dung lượng file mô hình** | Kích thước file nhị phân `model.bin` trên ổ đĩa | **< 50.0 MB** | **27.1 MB** | **ĐẠT (Bằng 54% giới hạn)** |
| **Throughput xử lý** | Batch inference trên 1 core CPU | **> 500 QPS** | **> 600 QPS** | **ĐẠT (Tiết kiệm chi phí Pod)** |

### 8.4 Trụ Cột 4 — Tác Động Sản Phẩm Trực Tuyến (Online Business Impact)

Khi đưa vào thử nghiệm A/B Testing trực tiếp trên luồng người dùng thật của ứng dụng Xanh SM:
* **Tỷ lệ Zero-Result Search:** Giảm ít nhất **15% – 20%** lượng truy vấn trả về 0 kết quả trên toàn hệ thống.
* **Tỷ lệ Chuyển đổi Đặt xe (Search-to-Ride Conversion Rate):** Tăng **0.8% – 1.2%** nhờ người dùng tìm thấy điểm đón ngay từ lần gõ đầu tiên.
* **Thời gian hoàn tất tìm kiếm (Time-to-First-Result):** Giảm trung bình **1.8 giây/phiên** do giảm thiểu số lần người dùng phải xoá đi gõ lại.

---

## Phụ lục A — Bằng Chứng Thực Nghiệm Lịch Sử: Tiến Hóa Từ Base V1 $\rightarrow$ Base V3/V3.1 FT Đến Arm E & Đối Soát Pretrained

Nhằm cung cấp cái nhìn toàn diện về lịch sử nghiên cứu và phát triển hệ thống ReparoS, phụ lục này ghi nhận đầy đủ kết quả đo đạc thực tế qua 4 thế hệ mô hình sản xuất (Base V1 $\rightarrow$ Base V3.1 FT) trước khi xác lập kiến trúc tối ưu **Arm E** và đối soát với trần năng lực Pretrained (**ViT5, BARTpho**).

---

### A.1. Quá Trình Tiến Hóa 4 Thế Hệ Base Trên Dữ Liệu Sản Xuất

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                       THÔNG SỐ VÀ CẤU HÌNH CÁC THẾ HỆ MÔ HÌNH NỘI BỘ (BASE V1 → V3.1 FT)                    │
├──────────────┬──────────────────┬──────────────┬───────────────────────────────┬────────────────────────────┤
│ Thế hệ       │ Cấu hình Enc/Dec │ Tham số/Dung │ Quy mô Dữ liệu Huấn luyện     │ Trọng tâm thử nghiệm       │
├──────────────┼──────────────────┼──────────────┼───────────────────────────────┼────────────────────────────┤
│ **Base V1**  │ 1E / 1D, d=128   │ 4.9M (4.8MB) │ 50,000 steps trên tập cơ sở   │ Baseline ban đầu           │
│ **Base V2**  │ 1E / 1D, d=128   │ 4.9M (4.8MB) │ 32,000 steps, Curriculum V2   │ Tăng cường lỗi tổ hợp      │
│ **Base V3**  │ 2E / 1D, d=128   │ 6.5M (6.5MB) │ 50,000 steps trên 4.0M cặp câu│ Mở rộng Vocab 12k, phủ 63T │
│ **Base V3.1**│ 2E / 1D (FT)     │ 6.5M (6.5MB) │ 6,000 steps FT trên 1.0M cặp  │ Vá điểm nghẽn địa chỉ/viết │
└──────────────┴──────────────────┴──────────────┴───────────────────────────────┴────────────────────────────┘
```

#### Bảng Đo Đạc Thực Tế Trên 3 Bộ Benchmark Lịch Sử:

1. **Benchmark `USER-CENTRIC-V2` (88 câu truy vấn người dùng thực tế):**
   * **Exact Match (Recall@1):** Base V1: `32.95%` $\rightarrow$ Base V2: `47.73%` $\rightarrow$ Base V3: `50.00%` $\rightarrow$ **Base V3.1 FT: `55.68%`** *(+22.73 pp so với V1)*.
   * **Case-Insensitive Recall@1:** Base V1: `42.05%` $\rightarrow$ Base V2: `55.68%` $\rightarrow$ Base V3: `57.95%` $\rightarrow$ **Base V3.1 FT: `63.64%`** *(+21.59 pp)*.
   * **Recall@10:** Base V1: `43.18%` $\rightarrow$ Base V3: `59.09%` $\rightarrow$ **Base V3.1 FT: `67.05%`** *(+23.87 pp)*.
   * **Autocorrect F1-Score:** Base V1: `38.16%` $\rightarrow$ Base V3: `56.41%` $\rightarrow$ **Base V3.1 FT: `60.87%`** *(+22.71 pp)*.
   * **CER Net Reduction (Mức độ giảm khoảng cách lỗi):** Tăng từ `29.71%` (V1) lên **`63.64%`** (V3.1 FT).

2. **Benchmark `COMPOSITIONAL-4K` (4,000 câu lỗi tổ hợp đa tầng):**
   * **Exact Match (Recall@1):** Tăng vọt từ `43.18%` (V1) lên **`62.95%`** (V3) và `62.72%` (V3.1 FT).
   * **Tỷ lệ gây thoái lui (Regression Rate):** Giảm mạnh từ `11.88%` (V1) xuống còn **`4.58%`** (V3) và `5.17%` (V3.1 FT), chứng minh tính an toàn vượt bậc khi nâng cấp lên tập train 4M và 2 Encoder.

3. **Benchmark `DIAGNOSTIC-10K` (10,000 câu phân rã lỗi chuyên sâu của Base V3/V3.1 FT):**
   * **Lỗi gõ nhầm VNI (`vni_leak`):** Đạt đỉnh cao **`93.38%`** (Base V3).
   * **Lỗi gõ nhầm Telex (`telex_leak`):** Duy trì xuất sắc **`93.00%`** (Base V3.1 FT).
   * **Khôi phục dấu thanh (`missing_diacritics`):** Đạt **`80.00%`**.
   * **Ranh giới từ dính/tách (`word_boundary`):** Đạt **`80.38%`**.

> **Bài học cốt tử dẫn tới Đề án TechJam:** Dù Base V3 và Base V3.1 FT đã cải thiện vượt bậc so với V1 (+22.7% User-Centric), mô hình vẫn chạm "trần năng lực" của cấu trúc 1 Decoder khi xử lý các chuỗi dài và địa chỉ tổ hợp sâu. Đây chính là động lực để chuyển dịch sang **Arm E (2 Encoder / 2 Decoder, 7.1M params)**.

---

### A.2. Bảng Đối Soát Toàn Diện: Arm E (Sweet Spot) vs. Pretrained Upper-Bound (ViT5, BARTpho)

Trên bộ khung 5 Frozen Suites (3,100 câu chuẩn hóa độc lập) và 18 ca kiểm chứng Canonical Zero-Click:

| Bộ kiểm thử (Số lượng mẫu) | Base V3 (2E/1D) | Arm E (2E/2D Scratch) | Arm G (ViT5 226M) | Arm H (BARTpho 132M) | Chênh lệch (G vs E) |
|:---|:---:|:---:|:---:|:---:|:---:|
| Plasticity (700 pairs) | 41.2% | **43.6%** | 43.1% | 40.4% | -0.5 pp *(E thắng)* |
| Retention (1,200 pairs) | 71.5% | **73.0%** | 73.6% | 72.6% | +0.6 pp |
| User-Centric (300 pairs) | 50.0% | **66.7%** 🚀 | 66.7% | 58.3% | 0.0 pp *(Hòa tuyệt đối)*|
| Protection Seen (400 pairs) | 88.5% | **91.0%** | 93.8% | 91.3% | +2.8 pp |
| Protection Held-out (500 pairs)| 58.0% | **62.4%** | 90.0% | **90.8%** 🚀 | **+27.6 pp** *(Trọng tâm gap)*|
| **Mean (5 suites - 3,100 câu)** | 62.84% | **67.33%** | **73.43%** | **70.68%** | **+6.10 pp** |
| Zero-Click (18 ca Canonical) | 11/18 (61.1%) | **12/18 (66.7%)** | 12/18 (66.7%) | 11/18 (61.1%) | 0.0 pp |
| **P50 Latency (CPU INT8)** | **3.80 ms** | **5.71 ms** | **88.75 ms** | **~45.00 ms** | **Nhanh hơn 15.5×** |

**Phân tích bản chất chênh lệch:**
1. **Sự vượt trội của Arm E so với Base V3 cũ:** Nâng cấp từ 1 Decoder (Base V3) lên 2 Decoder (Arm E) giúp chỉ số User-Centric nhảy vọt từ **50.0% lên 66.7% (+16.7 pp)** và Mean tăng từ **62.84% lên 67.33% (+4.49 pp)** trong khi độ trễ CPU vẫn chỉ 5.71 ms.
2. **Khoảng cách +6.10 pp giữa Pretrained (ViT5) và Scratch (Arm E):** Hoàn toàn không đến từ việc sửa dấu hay sửa lỗi Telex (Plasticity, Retention, User-Centric bám sát nhau từ -0.5 pp đến +0.6 pp). Thay vào đó, chênh lệch tập trung gần như 100% tại **Protection Held-out (+27.6 pp, từ 62.4% lên 90.0% ở ViT5 và 90.8% ở BARTpho)**.
3. **Định hướng giải pháp dứt điểm:** Không cần mô hình khổng lồ để học cú pháp tiếng Việt, mà cần tri thức thực thể (Entity / Brand Knowledge) được bơm vào tập huấn luyện của Arm E thông qua Data Augmentation và khai thác 3.55M log zero-click.

---

## Phụ lục B — Taxonomy Năng Lực Để Gán Nhãn Lỗi (Chuẩn Hoá Canonical)

| Nhóm | Ví dụ đầu vào | Output mong đợi (Canonical) | Thách thức chính |
|:---|:---|:---|:---|
| Viết tắt thương hiệu | `bv cho ray` | `bệnh viện chợ rẫy` | Mở rộng viết tắt + khôi phục dấu thanh |
| Viết tắt trường đại học | `dh bach khoa ha noi` | `đại học bách khoa hà nội` | Mở rộng đa token |
| Khu công nghiệp | `kcn song than 1` | `khu công nghiệp sóng thần 1` | Mở rộng viết tắt + sửa thanh điệu |
| Cơ quan hành chính | `ubnd xa phuoc thai` | `ủy ban nhân dân xã phước thái` | Chuẩn hoá đơn vị hành chính cấp xã |
| Địa chỉ phức hợp | `86 xo viet nghe tinh p19 binh thanh` | `86 xô viết nghệ tĩnh phường 19 bình thạnh` | Đa lỗi, địa chỉ dài, mở rộng phường |
| Phường/quận viết tắt | `duong le van viet q9` | `đường lê văn việt quận 9` | Mở rộng viết tắt đơn vị hành chính |
| Thiếu dấu Telex | `cau vuot song than` | `cầu vượt sóng thần` | Phục hồi dấu thanh theo ngữ cảnh địa danh |
| Copy / no-op | `bệnh viện 175` | `bệnh viện 175` | Bảo toàn truy vấn sạch (no-op retention) |
| Lỗi chính tả / gõ nhầm phím | `158/16 binh quew` | `158/16 bình quới` | Khôi phục ký tự gõ vội cuối từ |

---

## Phụ lục C — Ma Trận Kiểm Soát Rủi Ro Vận Hành (Production Guardrails Matrix)

Đúc kết từ các sự cố kỹ thuật thực tế trong quá trình thử nghiệm, hệ thống thiết lập 4 chốt chặn an toàn bắt buộc trước khi phục vụ người dùng thật:

| Rủi ro vận hành | Triệu chứng lỗi thực tế | Nguyên nhân kỹ thuật | Chốt chặn an toàn (Guardrail) đã kiểm chứng |
|---|---|---|---|
| **1. Vỡ mã Byte khi gặp chữ hoa** | `VinFast` $\rightarrow$ sinh ra rác `ahn jiin homestay` | Tokenizer Unigram bị vỡ thành byte fallback `<0x56>`, `<0x46>` khi gặp chữ hoa lạ. | **Tiền xử lý bắt buộc:** Chuẩn hoá Unicode NFC và tự động Lowercase toàn bộ input trước khi đưa vào Tokenizer. |
| **2. Sinh lặp từ vô tận (Repetition)** | `VinFast` $\rightarrow$ `vinfast vinfast`<br>`sanbay noibai` $\rightarrow$ sinh `sân bay nội bài nội` | CTranslate2 chạy giải mã mặc định `penalty = 1.0` kết hợp nhãn train cũ bị dính `đường đường`. | **Cấu hình Runtime:** Bật **`repetition_penalty = 1.15`** kết hợp làm sạch toàn diện nhãn lặp từ trong dataset. |
| **3. Đoán mò số nhà thiếu thông tin** | `16 lê lợi` bị sửa nhầm thành `1/6 lê lợi` | Dữ liệu train chứa các mẫu nhân tạo ill-posed (`32322` ép thành `323/22`). | **Lọc dữ liệu:** Loại bỏ 100% bài toán chèn dấu xuyệt số nhà thiếu căn cứ. Chỉ giữ bài toán tách ranh giới từ dính số xác định (`p19` $\rightarrow$ `phường 19`). |
| **4. Nghẽn luồng / Sập dịch vụ (Latency Spike)** | Timeout khi gặp chuỗi truy vấn rác siêu dài | Tắc nghẽn hàng đợi (thread starvation) vào giờ cao điểm mưa gió. | **Circuit Breaker:** Đặt hard-timeout 20ms. Nếu module QU vượt quá 20ms hoặc gặp exception, tự động **fallback trả về raw query gốc** cho Semantic Parser. |

