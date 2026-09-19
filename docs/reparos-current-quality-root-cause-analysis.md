# Phân tích nguyên nhân chất lượng ReparoS hiện tại

**Bài toán:** Query Understanding và sửa truy vấn địa điểm tiếng Việt cho ứng dụng đặt xe Xanh SM  
**Phạm vi:** ReparoS Base và ReparoS Curriculum V2 Final  
**Ngày cập nhật:** 19/09/2026  
**Trạng thái:** Phân tích nguyên nhân trước khi quyết định sinh thêm dữ liệu, train lâu hơn, tăng capacity hoặc bổ sung reranker

## 1. Mục tiêu của báo cáo

Báo cáo này ghi lại các nguyên nhân có thể khiến mô hình hiện tại chưa đạt mức production, dựa trên:

- lineage và phân phối dữ liệu hiện tại;
- cấu hình huấn luyện ReparoS Base;
- kết quả Diagnostic 10K, Composition 4K và User-centric 88;
- so sánh Base với Curriculum V2 Final;
- hiện tượng quan sát trực tiếp trong demo;
- thiết kế và kết quả được công bố trong bài báo ReparoS 2023.

Mục tiêu là phân biệt rõ ba loại kết luận:

1. **Đã có bằng chứng:** số liệu hiện tại trực tiếp xác nhận vấn đề.
2. **Giả thuyết mạnh:** phù hợp với nhiều quan sát nhưng chưa có thí nghiệm tách biệt.
3. **Chưa đủ bằng chứng:** cần audit hoặc ablation trước khi kết luận.

Báo cáo không mặc định rằng tăng kích thước mô hình sẽ giải quyết vấn đề. ReparoS 1 encoder–1 decoder vẫn là kiến trúc production candidate do yêu cầu latency CPU thấp hơn 20 ms.

## 2. Tóm tắt kết luận

Chất lượng hiện tại nhiều khả năng không đến từ một nguyên nhân duy nhất. Mẫu hình kết quả cho thấy bốn vấn đề chính:

1. **Base chưa có đủ độ phủ và độ ổn định để làm nền cho curriculum.**
2. **Dữ liệu synthetic còn mỏng và chưa chắc phản ánh đúng phân phối lỗi người dùng thực.**
3. **Curriculum tạo ra distribution shift và catastrophic forgetting.**
4. **Tần suất target/entity và thiếu hard negative khiến model over-correct hoặc suy diễn quá xa.**

Giới hạn capacity 1/1, tokenizer và decoding vẫn có thể đóng góp, nhưng hiện chưa phải các nguyên nhân có bằng chứng mạnh nhất.

Biểu hiện quan trọng nhất là Final không thất bại đồng đều. Nó học rất tốt những capability được tăng cường trong curriculum, nhưng giảm mạnh ở các capability ít được replay. Điều này nghiêng về vấn đề dữ liệu và chiến lược fine-tune hơn là mô hình hoàn toàn không đủ khả năng học bài toán.

## 3. Hiện trạng dữ liệu và cấu hình Base

### 3.1. Lineage dữ liệu

Dữ liệu gốc được lấy từ `data/osm/vietnam-latest.osm.pbf`, kích thước khoảng 328,7 MB.

Manifest `prepared-v3` ghi nhận:

- 496.643 entities;
- 575.198 terms/clean queries;
- 2.300.792 pairs;
- 575.198 clean pairs;
- 1.725.594 noisy pairs;
- ba noisy variants và một clean control trên mỗi term theo cấu hình sinh dữ liệu.

Sau khi chia leak-free theo normalized clean-query group, tập ReparoS Base có:

| Split | Pairs | Unique clean-query groups |
|---|---:|---:|
| Train | 1.838.912 | 251.241 |
| Validation | 234.424 | 31.322 |
| Test | 227.456 | 31.435 |

Các biến thể noisy trỏ về cùng một clean query được giữ trong cùng split. Overlap group giữa train, validation và test bằng 0.

Riêng tập train:

- 1.378.560 noisy rows;
- 460.352 clean rows;
- clean fraction khoảng 25,03%;
- 624 synthetic no-op đã có sẵn trong dữ liệu và được phân loại lại, không phải dòng được thêm mới.

### 3.2. Phân phối lỗi trong Base train

Một số nhóm chính:

| Nhóm | Số dòng xấp xỉ |
|---|---:|
| keyboard_edit | 388K |
| missing_diacritics_full | 239K |
| wrong_diacritic | 193K |
| missing_diacritics_partial | 184K |
| word_boundary | 107K |
| telex_leak | 104K |
| vni_leak | 71K |
| address_abbreviation | 33K |
| address_symbol | 8K |

Phân phối này cho thấy nhóm abbreviation, symbol và các tổ hợp khó có lượng dữ liệu thấp hơn đáng kể so với keyboard hoặc mất dấu. Số lượng dòng cũng chưa phản ánh độ khó: abbreviation phụ thuộc ngữ cảnh và cần nhiều hard negative hơn một phép bỏ dấu gần deterministic.

### 3.3. Cấu hình mô hình Base

- Transformer encoder-decoder;
- 1 encoder layer và 1 decoder layer;
- hidden size 128;
- embedding 128;
- 8 attention heads;
- FFN 512;
- SentencePiece, target vocabulary thực tế 7.008 token;
- token batch 16.384;
- FP16;
- 25.000 train steps;
- validation và checkpoint mỗi 1.250 steps;
- Adam + Noam, warm-up 1.000 steps;
- beam width 10 khi decode.

Trên Kaggle T4, lần train Base thực tế mất khoảng ba giờ. Khoảng thời gian này bao gồm 20 lần validation trên tập 234.424 câu và 20 lần lưu checkpoint, nên không thể coi toàn bộ ba giờ là thời gian GPU training thuần.

## 4. Kết quả chất lượng hiện tại

### 4.1. ReparoS Base

| Bộ đánh giá | Exact/Recall@1 | Recall@10 | Improvement | Regression | Clean preservation |
|---|---:|---:|---:|---:|---:|
| Diagnostic 10K | 77,75% overall; 75,62% noisy | 90,58% | 71,60% | 7,05% | 90,72% |
| Composition 4K | 43,35% | 67,55% | 74,65% | 12,35% | — |
| User-centric 88 | 32,95% overall; 36,25% noisy | 45,45% | 60,23% | 18,18% | 0% ở clean-protected |

Base mạnh ở Telex, VNI, mất dấu và boundary đơn giản trên dữ liệu diagnostic. Base yếu ở composition phức tạp, abbreviation/acronym thực tế và bảo vệ brand/query sạch.

Một số điểm đáng chú ý:

- Composition `all_joined`: 2,8% exact.
- User-centric abbreviation: 0%.
- User-centric abbreviation + boundary: 0%.
- User-centric location acronym: 0%.
- User-centric clean protected: 0%.

### 4.2. Curriculum V2 Final so với Base

| Bộ đánh giá | Base | Final | Thay đổi |
|---|---:|---:|---:|
| User-centric exact noisy | 36,25% | 68,75% | +32,50 điểm % |
| Composition exact | 43,35% | 58,90% | +15,55 điểm % |
| Composition Recall@10 | 67,55% | 83,18% | +15,63 điểm % |
| Diagnostic exact noisy | 75,62% | 61,67% | -13,95 điểm % |
| Diagnostic Recall@10 | 90,58% | 84,40% | -6,18 điểm % |

Các capability Final cải thiện rõ:

- location acronym: 0% → 100%;
- diacritics + boundary: 12,5% → 100%;
- abbreviation + boundary: 0% → 75%;
- abbreviation: 0% → 62,5%;
- mixed long: 0% → 50%;
- Composition all-joined: 2,8% → 43,4%.

Các capability Final giảm:

- VNI diagnostic: 88,38% → 48,13%;
- address abbreviation diagnostic: 91,13% → 39,63%;
- wrong diacritic diagnostic: 86,00% → 62,25%;
- short ambiguous user-centric: 100% → 75%;
- VNI/input-method user-centric: 100% → 37,5%;
- clean protected vẫn 0%.

Kết quả này là bằng chứng trực tiếp rằng curriculum đã tạo ra capability mới, nhưng đồng thời làm suy giảm khả năng tổng quát đã có ở Base.

## 5. Phân tích từng nhóm nguyên nhân

### 5.1. Base chưa hội tụ hoặc chưa ổn định

**Trạng thái:** Giả thuyết mạnh, chưa được kiểm chứng trực tiếp.

Base được train 25K steps, nhưng hiện chưa có báo cáo tổng hợp metric của toàn bộ các checkpoint cuối. Chưa biết:

- validation loss còn giảm ở 25K hay không;
- exact và Recall@10 đã bão hòa chưa;
- checkpoint 20K, 22,5K và 25K có dao động mạnh không;
- seed khác có tạo kết quả tương tự không.

Nếu checkpoint Base chưa ổn định, fine-tune từ một checkpoint đơn lẻ sẽ làm kết quả curriculum nhạy với trạng thái ngẫu nhiên của Base.

**Cách kiểm chứng:**

1. Chấm các checkpoint 15K–25K trên cả ba bộ.
2. Vẽ loss, exact, Recall@10, regression và clean preservation theo step.
3. Train tiếp từ 25K lên 35K/50K với dữ liệu Base không đổi.
4. Nếu có ngân sách, chạy thêm ít nhất một seed.

**Dấu hiệu xác nhận:** metric cuối vẫn tăng, dao động lớn giữa checkpoint, hoặc continued training cải thiện đồng thời nhiều bộ test.

### 5.2. Độ phủ noisy variants của Base quá thấp

**Trạng thái:** Giả thuyết mạnh.

Bài báo ReparoS tạo khoảng 270 triệu synthetic pairs từ 300 nghìn clean seed queries, tương đương trung bình khoảng 900 pairs/seed. Đây không có nghĩa mỗi seed bắt buộc có 900 lỗi độc nhất, nhưng cho thấy không gian corruption trong paper dày hơn dữ liệu hiện tại rất nhiều.

Dữ liệu hiện tại được cấu hình khoảng ba noisy variants trên mỗi term ở giai đoạn tạo `prepared-v3`. Vì mỗi query có thể có nhiều loại lỗi, vị trí lỗi và tổ hợp lỗi, ba biến thể khó bao phủ đầy đủ không gian:

```text
bệnh viện bạch mai
├── benh vien bach mai
├── beenhj vieenj bachj mai
├── be65nh5 vie65n6 bach mai
├── bệnh viện bạchmai
├── benh vienbach mai
├── bv bạch mai
├── bvbạchmai
├── benh vien bm
└── các tổ hợp keyboard + diacritics + boundary khác
```

Các kết quả Base phù hợp với giả thuyết này: primitive đơn lẻ tốt hơn nhiều so với composition và abbreviation + boundary.

Tuy nhiên, tăng số dòng một cách máy móc chưa chắc hiệu quả. Cần đo:

- số noisy string độc nhất trên mỗi target;
- tỷ lệ duplicate/no-op;
- coverage theo loại lỗi và vị trí lỗi;
- coverage tổ hợp hai hoặc ba lỗi;
- mức tự nhiên của biến thể so với query người dùng.

### 5.3. Synthetic noise chưa khớp lỗi người dùng thật

**Trạng thái:** Giả thuyết mạnh.

Diagnostic 10K chủ yếu đo phân phối synthetic rộng, trong khi User-centric 88 mô phỏng intent search thực tế. Khoảng cách lớn giữa hai bộ cho thấy model học taxonomy synthetic tốt hơn hành vi người dùng.

Paper ReparoS không chỉ dùng edit ngẫu nhiên. Họ sử dụng:

- Brill–Moore error probability;
- phonetic/transliteration candidates;
- query reformulation chain;
- edit/phonetic kết hợp compounding;
- weak supervision từ click feedback;
- clean query được chọn bằng frequency và CTR.

Dữ liệu hiện tại chưa có nguồn query log/click feedback tương đương. Vì vậy, tăng từ 3 lên 25 hay 100 variants chỉ có ý nghĩa nếu generator sinh lỗi có cấu trúc và giống người dùng, không phải tăng số lượng biến thể rác.

### 5.4. Mất cân bằng giữa các nhóm lỗi

**Trạng thái:** Có bằng chứng từ manifest.

Các nhóm dễ như keyboard edit và missing diacritics chiếm tỷ lệ lớn, trong khi abbreviation, symbol và composition khó ít hơn nhiều. Model tối ưu cross-entropy trên toàn tập nên nhóm nhiều mẫu có ảnh hưởng lớn hơn.

Ngoài số dòng, cần tính cả độ đa dạng:

- 100K dòng sinh từ ít pattern lặp lại không tương đương 100K lỗi độc lập;
- một abbreviation cần cả positive expansion và negative preservation;
- một boundary corruption có nhiều vị trí nối/tách khác nhau.

Do đó, phân phối cần được thiết kế theo capability và độ khó, không chỉ theo tỷ lệ phần trăm tổng thể.

### 5.5. Thiếu clean hard negative và protected query

**Trạng thái:** Có bằng chứng rõ từ User-centric và demo.

Clean fraction hiện tại khoảng 25%, nhưng clean ngẫu nhiên không đủ để dạy model tránh sửa các chuỗi có hình thức giống abbreviation hoặc typo.

Các failure điển hình:

```text
BV Bank           → bệnh viện bank
Starbucks Reserve → starbucks resort
Q1 Tower          → q1 tower hoặc output sai khác
thcs              → trung học cơ sở trung học cơ sở
```

Model cần hard negative có chủ đích:

- brand và tên riêng;
- chuỗi viết hoa;
- chữ + số;
- mã đường/quốc lộ;
- abbreviation hợp lệ cần giữ nguyên;
- query sạch ngắn và mơ hồ.

Đây không chỉ là việc tăng clean fraction. Clean data phải nằm sát decision boundary của các rule correction.

### 5.6. Policy target không nhất quán và source ambiguity

**Trạng thái:** Chưa audit.

Một input có thể có nhiều output hợp lệ tùy policy:

```text
thcs → giữ nguyên
thcs → trung học cơ sở
thcs → một trường cụ thể nếu có ngữ cảnh khác
```

```text
bv → bệnh viện
bv → giữ nguyên trong BV Bank
bv → bảo vệ trong ngữ cảnh khác
```

Nếu cùng hoặc gần cùng source được gán nhiều target, mô hình one-to-one nhận tín hiệu xung đột và có xu hướng chọn target phổ biến nhất.

**Cần audit:**

- số normalized sources có nhiều target;
- entropy target theo source;
- policy expansion abbreviation;
- khác biệt target casing/punctuation;
- cùng alias có trỏ đến nhiều entity hay không.

### 5.7. Target và entity frequency bias

**Trạng thái:** Đã có bằng chứng ở Stage 3.

Target `đại học tài chính marketing` xuất hiện khoảng 1.011 lần trong dữ liệu lexical Stage 3. Khi user nhập `truongwf`, model từng sinh chính entity này thay vì correction gần nhất là `trường`.

Đây là biểu hiện của target prior quá mạnh:

\[
P(\text{target}\mid\text{input})
\]

Nếu một entity xuất hiện qua nhiều alias và biến thể, decoder học xác suất sinh entity đó cao. Khi source ngắn hoặc mơ hồ, prior có thể lấn át bằng chứng từ input.

**Cần audit:**

- frequency top 100 và top 1.000 targets;
- tỷ lệ dữ liệu do các target phổ biến nhất chiếm;
- số alias/variant trên mỗi entity;
- cap per target/entity;
- correlation giữa target frequency và hallucination.

### 5.8. Curriculum gây catastrophic forgetting

**Trạng thái:** Đã có bằng chứng rõ.

Final tăng mạnh ở User-centric và Composition nhưng giảm trên Diagnostic, VNI và một số nhóm tổng quát. Đây là mẫu hình điển hình của fine-tune distribution shift.

Nguyên nhân có thể gồm:

- tỷ lệ Base replay thấp;
- stage data quá tập trung;
- learning rate fine-tune chưa phù hợp;
- mỗi stage train quá lâu;
- target/entity lặp nhiều;
- gate chỉ kiểm tra capability của stage mà chưa kiểm tra đầy đủ global regression.

Gate hiện tại có thể pass dù một nhóm ngoài gate giảm mạnh. Ví dụ Stage 3 có thể vượt abbreviation và lexical typo nhưng không bị chặn khi VNI tổng quát giảm.

**Cần bổ sung global gate:**

- Diagnostic exact và Recall@10;
- VNI và Telex tổng quát;
- clean preservation;
- short ambiguous;
- brand/alphanumeric protection;
- global regression;
- repetition rate.

### 5.9. Curriculum đang phải vá capability đáng lẽ Base cần có

**Trạng thái:** Giả thuyết mạnh.

Stage 1–3 hiện vừa học primitive, composition, lexical mapping, abbreviation, acronym và preservation. Phạm vi này rộng hơn vai trò refinement thông thường.

Trong paper, ReparoS-Base đã là một baseline mạnh trước khi curriculum. C1 tập trung vào lỗi edit/phonetic + compounding khó; C2 dùng weak supervision để khôi phục và nâng regression. Nếu Base hiện tại chưa đủ rộng, mỗi stage của mình trở thành một đợt thay distribution thay vì bổ sung nhẹ capability khó.

### 5.10. Tokenizer fragmentation

**Trạng thái:** Chưa đủ bằng chứng.

SentencePiece hiện có 7.008 target labels. Các input như:

```text
bvbạchmai
truongwf
ubndp
d9u7o7ng2
be65nh5
```

có thể bị chia thành nhiều subword hiếm. Điều này làm encoder khó học representation ổn định. Ngược lại, entity phổ biến có thể được biểu diễn bằng ít token và dễ được decoder ưu tiên.

**Cần audit:**

- số subword trung bình theo error type;
- phân phối độ dài noisy so với clean;
- tỷ lệ `<unk>`;
- token frequency;
- exact/Recall@10 theo số subword;
- fragmentation của failure so với success.

### 5.11. Top-1 ranking khác generation coverage

**Trạng thái:** Có bằng chứng một phần.

Khoảng cách giữa Recall@1 và Recall@10 cho thấy nhiều trường hợp gold đã nằm trong beam nhưng chưa được chọn Top-1:

- Base Diagnostic: 77,75% → 90,58%;
- Base Composition: 43,35% → 67,55%;
- Base User-centric: 32,95% → 45,45%;
- Final Composition: 58,90% → 83,18%.

Cần chia mỗi lỗi thành:

1. **Generation failure:** gold không có trong Top-10.
2. **Ranking failure:** gold có trong Top-10 nhưng Top-1 sai.

Reranker chỉ xử lý nhóm thứ hai. Vì Base và Final vẫn có nhiều generation failure, chưa nên coi reranker là giải pháp thay cho việc củng cố Base. Tuy vậy, Recall@10 cao cho thấy reranker có giá trị sau khi generation coverage được nâng thêm.

### 5.12. Decoding và sai lệch OpenNMT/CTranslate2

**Trạng thái:** Có ảnh hưởng cục bộ, không phải nguyên nhân chính.

OpenNMT và CTranslate2 đôi khi chọn Top-1 khác nhau dù beam tương tự, ví dụ `ho guom`. Absolute score của hai backend không cùng scale nên không được so trực tiếp.

Sweep trên Diagnostic 10K cho thấy CT2 với length penalty 0 và không dùng `no_repeat_ngram_size` vẫn có chất lượng tổng thể tốt nhất trong các cấu hình đã thử. Bật no-repeat giảm repetition nhưng đồng thời giảm exact và Recall@10.

Do đó decoding giải thích một số lỗi Top-1 và repetition, nhưng không giải thích được mức giảm lớn của VNI hoặc Diagnostic sau curriculum.

### 5.13. Capacity của kiến trúc 1/1

**Trạng thái:** Có thể tồn tại, nhưng chưa phải giả thuyết ưu tiên.

Paper cho thấy model sâu hơn cải thiện chất lượng, nhưng cuối cùng vẫn chọn 1 encoder–1 decoder cho production vì latency CPU. Số latency của paper được đo trên hạ tầng riêng của họ, không thể áp trực tiếp cho laptop, Kaggle hay production của Xanh SM.

Kết quả Final của mình cho thấy model 1/1 vẫn học được những capability mới rất mạnh. Điều này chống lại kết luận rằng model hoàn toàn thiếu khả năng biểu diễn. Mẫu hình `học nhóm mới nhưng quên nhóm cũ` phù hợp với data distribution và fine-tuning hơn.

Capacity sweep 2/2 hoặc 4/4 chỉ nên là thí nghiệm chẩn đoán sau khi:

- Base convergence được kiểm tra;
- dữ liệu được audit;
- target bias được sửa;
- replay/global gate được thiết kế lại.

Kiến trúc production vẫn phải được chọn theo Pareto quality–latency với CPU p95 dưới 20 ms trên đúng hạ tầng triển khai.

### 5.14. Bộ đánh giá chưa đại diện hoàn toàn cho production

**Trạng thái:** Khoảng trống đã biết.

- Diagnostic 10K rộng nhưng chủ yếu synthetic.
- Composition 4K tập trung vào lỗi ghép capability.
- User-centric 88 chỉ có tám mẫu mỗi nhóm; một câu làm metric nhóm thay đổi 12,5 điểm %.
- Chưa có bộ human-labelled từ traffic thực và chưa có frequency weighting.
- Exact correction chưa phản ánh đầy đủ retrieval outcome của search.

Cần xây dựng thêm:

- regression set chủ yếu là clean/head queries;
- improvement set chủ yếu là noisy/tail queries;
- brand/protected set;
- traffic-weighted set;
- retrieval evaluation: corrected query có đưa đúng POI vào kết quả hay không.

## 6. Xếp hạng nguyên nhân theo bằng chứng hiện tại

| Ưu tiên | Nguyên nhân | Đánh giá hiện tại |
|---:|---|---|
| 1 | Curriculum imbalance và catastrophic forgetting | Đã có bằng chứng trực tiếp |
| 2 | Target/entity frequency bias | Đã có bằng chứng trực tiếp |
| 3 | Thiếu clean hard negative/protected data | Đã có bằng chứng trực tiếp |
| 4 | Độ phủ noisy variants của Base còn mỏng | Giả thuyết mạnh |
| 5 | Synthetic noise chưa giống lỗi người dùng | Giả thuyết mạnh |
| 6 | Base chưa hội tụ hoặc chưa ổn định | Cần checkpoint audit |
| 7 | Policy target/source ambiguity | Cần data audit |
| 8 | Top-1 ranking chưa tốt | Có bằng chứng từ khoảng cách Recall@1–10 |
| 9 | Tokenizer fragmentation | Chưa audit |
| 10 | Decoding/backend mismatch | Có ảnh hưởng cục bộ |
| 11 | Capacity 1/1 không đủ | Chưa đủ bằng chứng, không ưu tiên trước |
| 12 | Evaluation chưa sát production | Khoảng trống đo lường chắc chắn |

## 7. Kế hoạch điều tra trước khi train lại

### Phase A — Audit Base checkpoint

Chấm tất cả checkpoint được giữ lại và, nếu còn artifact, các checkpoint 15K–25K:

- Diagnostic 10K;
- Composition 4K;
- User-centric 88;
- clean preservation;
- regression;
- Recall@1 và Recall@10;
- repetition.

**Quyết định:** Nếu metric vẫn tăng ở checkpoint cuối, continued training Base là thí nghiệm đầu tiên. Nếu metric dao động, cần xem seed, learning rate và checkpoint selection.

### Phase B — Audit dữ liệu Base

Xuất báo cáo:

- unique noisy variants per clean group;
- duplicate và no-op rate;
- error type và composition matrix;
- top targets/entities;
- aliases per entity;
- normalized source có nhiều target;
- clean protected coverage;
- độ dài subword theo nhóm lỗi.

**Quyết định:** Chỉ tăng dữ liệu sau khi biết thiếu coverage nào và generator nào đang sinh dữ liệu giá trị thấp.

### Phase C — Phân rã lỗi beam

Với từng bộ test, chia lỗi thành:

- gold không có trong Top-10;
- gold có trong Top-10 nhưng không đứng đầu;
- output sửa quá mức;
- output lặp;
- output đúng correction nhưng sai casing/punctuation;
- ambiguous/policy mismatch.

**Quyết định:** Nếu ranking failure chiếm đa số, lightweight reranker đáng làm. Nếu generation failure chiếm đa số, ưu tiên dữ liệu/Base.

### Phase D — Củng cố Base

Thử theo thứ tự:

1. train tiếp Base hiện tại để kiểm tra convergence;
2. bổ sung hard negative/protected pairs;
3. cap target/entity frequency;
4. tăng variants có kiểm soát lên Base-10;
5. nếu còn tăng rõ, thử Base-25;
6. chỉ thử Base-50/100 nếu scaling curve chưa bão hòa.

Không nên nhân thẳng số step theo số dòng. Dataset lớn và đa dạng hơn có thể cần ít epoch hơn. Chọn checkpoint theo validation convergence và global gate.

### Phase E — Thiết kế lại curriculum

Sau khi Base đạt gate:

- tăng Base/general replay ở mọi stage;
- giới hạn mẫu trên target/entity;
- dùng hard negative trong Stage 3;
- giảm learning rate hoặc số step nếu xuất hiện forgetting;
- áp global regression gate bên cạnh stage-specific gate;
- không cho một checkpoint pass nếu VNI, clean protection hoặc Diagnostic giảm vượt ngân sách cho phép.

### Phase F — Capacity và reranker

Chỉ sau khi các phase trên hoàn thành:

- chạy 2/2 như ablation capacity;
- benchmark CPU p50/p95/p99 trên hạ tầng production thật;
- giữ 1/1 nếu gain không đủ bù latency;
- thử reranker khi Recall@10 đủ cao và ranking failure chiếm tỷ lệ lớn.

## 8. Gate đề xuất cho Base ổn định

Các ngưỡng dưới đây chưa phải ngưỡng production cuối cùng; chúng là điều kiện để Base đủ ổn định trước curriculum:

| Nhóm | Điều kiện đề xuất |
|---|---|
| Diagnostic exact noisy | Không thấp hơn Base hiện tại và ổn định qua checkpoint |
| Diagnostic Recall@10 | ≥ 90% |
| Clean preservation | ≥ 92%, sau đó tăng dần |
| Telex | Không regression đáng kể so với Base hiện tại |
| VNI | Không regression đáng kể so với Base hiện tại |
| Composition | Có cải thiện ở boundary + diacritics, không chỉ primitive |
| Clean protected | Phải lớn hơn 0% rõ rệt |
| Regression | Không tăng khi exact tăng |
| Checkpoint stability | Các checkpoint cuối không dao động mạnh |

Ngưỡng phải được hiệu chỉnh sau khi có production-labelled set.

## 9. Những kết luận chưa được phép khẳng định

Cho đến khi hoàn thành audit, không nên khẳng định:

- model tệ chủ yếu vì capacity 1/1;
- tăng lên 4/4 chắc chắn vẫn đạt CPU latency dưới 20 ms;
- 25K steps đã đủ hội tụ;
- sinh 100 variants luôn tốt hơn 10 hoặc 25;
- 270M pairs của paper đồng nghĩa đúng 900 biến thể độc nhất trên mỗi query;
- reranker sẽ giải quyết lỗi nếu gold chưa xuất hiện trong beam;
- Diagnostic 10K đại diện cho traffic production;
- pass stage gate đồng nghĩa model tổng thể tốt hơn.

## 10. Kết luận cuối

Giả thuyết làm việc hiện tại là:

> ReparoS Base chưa được chứng minh là đã hội tụ và có độ phủ noise còn mỏng. Dữ liệu synthetic chưa phản ánh đầy đủ query thực tế, đặc biệt abbreviation, protected query và composition. Curriculum V2 sau đó học mạnh các nhóm mục tiêu nhưng làm lệch phân phối, khuếch đại target prior và gây catastrophic forgetting. Capacity 1/1 có thể giới hạn trần chất lượng, nhưng chưa phải nguyên nhân ưu tiên khi các vấn đề dữ liệu và training chưa được loại trừ.

Quyết định tiếp theo không phải lập tức tăng model hay tạo hàng chục triệu dòng. Bước đúng là audit Base checkpoint, audit coverage/conflict của dữ liệu và phân rã generation–ranking failure. Sau đó mới chạy scaling Base-10/Base-25 và thiết kế lại curriculum có replay cùng global regression gate.
