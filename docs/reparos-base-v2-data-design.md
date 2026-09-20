# Thiết kế dữ liệu ReparoS Base V2

**Bài toán:** Sửa truy vấn địa điểm tiếng Việt cho Xanh SM  
**Ngày:** 19/09/2026  
**Mục tiêu:** Xây dựng một Base tổng quát và ổn định trước khi áp dụng curriculum  
**Phạm vi:** Thiết kế dữ liệu; chưa tối ưu thời gian train và chưa thay đổi kiến trúc 1 encoder–1 decoder

## 1. Quyết định thiết kế

Base V2 không tiếp tục cách sinh `N` biến thể bằng cách bốc ngẫu nhiên từ một bảng weight duy nhất. Thay vào đó, dữ liệu được thiết kế theo **capability quota**:

- mỗi nhóm clean query được tiếp xúc với nhiều họ lỗi xác định;
- lỗi đơn, lỗi tổ hợp và clean/protected examples đều có quota riêng;
- chỉ sinh loại lỗi phù hợp với query;
- loại duplicate và no-op trước khi ghi;
- giới hạn số seed/variant trên mỗi target và entity;
- bare abbreviation/acronym không được tự động bung nếu chưa có policy hoặc mapping đã duyệt;
- train, validation và test được chia theo normalized target group trước khi sinh noise;
- các phiên bản scaling phải nested để so sánh công bằng.

Mục tiêu đầu tiên không phải tái tạo đúng 270 triệu pairs của paper. Mục tiêu là tạo một tập Base có coverage đo được, noise hợp lý và không bị target prior chi phối.

## 2. Vấn đề của generator hiện tại

Generator hiện tại trong `src/webspell/osm/noise.py`:

1. Chọn loại lỗi ngẫu nhiên theo `DEFAULT_NOISE_WEIGHTS`.
2. Thử sinh tối đa 20 lần nếu loại lỗi không áp dụng được.
3. Nếu vẫn thất bại, fallback về `keyboard_edit` hoặc nối thêm ký tự `a`.
4. Trả đúng số trial nhưng không bảo đảm mỗi family có mặt.
5. Không loại duplicate/no-op ngay trong `location_query_variants`.
6. `combined` chỉ lấy hai operation ngẫu nhiên, không bảo đảm coverage tổ hợp quan trọng.
7. Abbreviation dictionary còn hẹp và không có negative policy đi kèm.

Hệ quả: tăng `--noisy-variants` không đồng nghĩa tăng coverage đều; query không phù hợp với abbreviation/symbol có thể tiếp tục tạo keyboard edit; composition quan trọng xuất hiện ít; target có nhiều alias có thể đóng góp quá nhiều dòng.

## 3. Đơn vị dữ liệu

### 3.1. Đơn vị split

Đơn vị split là `normalized correct-query group`. Toàn bộ alias và noisy variants trỏ tới cùng normalized target phải nằm trong cùng một split. Không chia ngẫu nhiên theo dòng sau khi sinh noise.

### 3.2. Đơn vị sampling

Đơn vị sampling chính là **target group**, không phải OSM row hay alias row:

```text
target_group
├── canonical target
├── reviewed aliases
├── entity metadata
├── protected attributes
└── generated noisy sources
```

Điều này ngăn một entity có nhiều OSM objects hoặc aliases chiếm phần lớn dữ liệu.

### 3.3. Chọn clean seed surface

Cho mỗi target group:

- luôn giữ canonical target;
- chọn tối đa hai alias có giá trị;
- loại alias trùng sau normalize;
- loại alias chỉ khác casing/punctuation nhưng không thêm thông tin;
- đánh dấu nguồn canonical, OSM alias, curated alias hoặc observed query.

Giới hạn mặc định: `max_seed_surfaces_per_target = 3`.

## 4. Policy output của mô hình

### 4.1. Correction mặc định

Base được phép khôi phục dấu, sửa Telex/VNI, lỗi bàn phím và lexical typo hợp lý, sửa dính/tách từ, chuẩn hóa spacing/address symbol và bung abbreviation khi mapping cùng ngữ cảnh đủ rõ.

### 4.2. Những việc Base không được tự suy diễn

Base không được biến một từ chung thành entity cụ thể, bung acronym mơ hồ, đổi brand thành danh từ phổ thông, thêm địa chỉ không có trong input hoặc thay đổi số/mã đường nếu không có policy.

| Input | Target policy |
|---|---|
| `truongwf` | `trường`, không phải một trường đại học cụ thể |
| `BV Bank` | giữ nguyên |
| `Q1 Tower` | giữ nguyên |
| `thcs` | giữ nguyên nếu chưa có policy expansion thống nhất |
| `bv bạch mai` | có thể bung thành `bệnh viện bạch mai` |
| `đhbáchkhoa hn` | có thể bung khi mapping đã được duyệt |

### 4.3. Curated expansion

Mọi lexical/acronym expansion phải nằm trong registry có version gồm source pattern, expected expansion, required/blocked context, confidence class, provenance và review status. Không tự động khai thác bare acronym từ initial letters của OSM name rồi coi đó là ground truth.

## 5. Taxonomy Base V2

### Family A — Clean và protected

- clean canonical và alias;
- brand/operator;
- alphanumeric như `Q1 Tower`, `D2`, `1A`, `ATM`;
- acronym hợp lệ cần giữ nguyên;
- tên nước ngoài;
- query ngắn/mơ hồ;
- punctuation hợp lệ;
- clean query gần giống typo nhưng là entity thật.

Clean phải có `protection_reason`, không chỉ `src == tgt`.

### Family B — Primitive orthographic/input method

- missing diacritics full/partial;
- wrong tone và wrong vowel shape;
- Telex canonical/malformed;
- VNI canonical/malformed;
- keyboard deletion/insertion/neighbor substitution/transposition;
- repeated character.

### Family C — Boundary và spacing

- nối một hoặc nhiều boundary;
- all-joined;
- tách sai token dài;
- thừa space;
- thiếu space quanh số/symbol;
- boundary ở prefix loại địa điểm và administrative unit.

### Family D — Composition

Sinh theo ma trận cố định:

- missing diacritics + boundary;
- Telex/malformed Telex/VNI + boundary;
- keyboard + missing diacritics;
- keyboard + boundary;
- abbreviation + boundary;
- abbreviation + missing diacritics;
- number/symbol + missing diacritics;
- controlled three-operation;
- all-joined + missing diacritics.

### Family E — Domain lexical/address

- contextual address abbreviation;
- curated hospital/school/university abbreviation;
- location acronym có context;
- district/ward/city shorthand;
- address symbol, house number và route code;
- POI category abbreviation;
- lexical typo từ confusion registry.

Family này phải có positive và negative examples cân xứng: `bv bạch mai` là positive expansion, `BV Bank` là protected negative.

### Family F — Observed/weakly supervised

- reformulation chain;
- accepted correction/click-through;
- repeated failed query rồi successful query;
- production confusion counts;
- human-reviewed difficult pairs.

Khi chưa có log, để trống family này thay vì tạo synthetic rồi gắn nhãn như lỗi thật.

## 6. Quota đề xuất cho Base V2-32

`V2-32` là tối đa 32 noisy trials có giá trị trên mỗi seed surface. Không bắt buộc ghi đủ nếu query không đủ điều kiện.

| Macro-family | Trials/seed | Tỷ lệ noisy |
|---|---:|---:|
| Primitive orthographic/input | 10 | 31,25% |
| Boundary/spacing | 5 | 15,63% |
| Composition | 10 | 31,25% |
| Domain lexical/address | 5 | 15,63% |
| Observed/hard confusion reserve | 2 | 6,25% |
| **Tổng noisy tối đa** | **32** | **100%** |

Quota mặc định:

```text
Primitive (10)
  1 missing_diacritics_full
  1 missing_diacritics_partial
  1 wrong_diacritic_or_shape
  1 telex_canonical
  1 telex_malformed
  1 vni_canonical
  1 vni_malformed
  3 keyboard operations khác nhau

Boundary (5)
  2 single-boundary ở vị trí khác nhau
  1 multi-boundary
  1 all-joined nếu đủ điều kiện
  1 wrong-split hoặc number/symbol spacing

Composition (10)
  2 missing_diacritics + boundary
  1 telex + boundary
  1 malformed_telex + boundary
  1 vni + boundary
  1 keyboard + missing_diacritics
  1 keyboard + boundary
  1 abbreviation + boundary nếu đủ điều kiện
  1 abbreviation + missing_diacritics nếu đủ điều kiện
  1 controlled_three_operation

Domain lexical/address (5)
  contextual abbreviation/acronym/symbol/lexical confusion
  chỉ sinh nếu target đủ điều kiện và mapping hợp lệ

Observed reserve (2)
  observed confusion hoặc reviewed hard pair
  không có dữ liệu thì không synthetic-fill bằng keyboard edit
```

Nếu một slot không áp dụng được, generator thử biến thể khác cùng family rồi bỏ slot; không chuyển thành keyboard edit. `ineligible_reason` phải xuất hiện trong audit summary.

## 7. Clean ratio và hard negative

Mục tiêu khởi đầu là 25% clean/protected và 75% noisy/correction.

| Clean subtype | Tỷ lệ phần clean |
|---|---:|
| canonical/alias thông thường | 40% |
| brand/proper-name protected | 20% |
| alphanumeric/number/symbol | 15% |
| short ambiguous/acronym | 15% |
| adversarial near-correction | 10% |

Không duplicate cùng clean row nhiều lần chỉ để đủ tỷ lệ. Nếu protected pool nhỏ, cần curate thêm target.

## 8. Cap chống target/entity bias

```text
max_seed_surfaces_per_target = 3
max_noisy_rows_per_seed_surface = profile quota
max_identical_source_target_pair = 1
max_identical_source = 3 targets để audit ambiguity
max_curated_mapping_repetitions = 1 trước oversampling
```

Nếu cần oversample capability hiếm, thực hiện ở DataLoader/corpus weight thay vì ghi hàng nghìn dòng giống nhau. Manifest phải có top target/entity frequencies, alias count, duplicate counts và tỷ lệ rows thuộc top 100/1.000 targets.

## 9. Deduplication và conflict policy

Khóa exact dedup:

```text
(normalized_source, normalized_target, error_family, operations)
```

Nếu một normalized source trỏ tới nhiều target, phải xuất conflict report và phân loại `valid_ambiguous`, `bad_alias` hoặc `policy_conflict`. Bare ambiguous source phải được giữ nguyên hoặc yêu cầu context tùy policy.

No-op không được ghi thành noisy row. Có thể chuyển sang protected pool nếu có lý do hợp lệ, đồng thời ghi no-op count theo generator.

## 10. Scaling profiles phục vụ ablation

Các profile phải nested: V2-8 là tập con của V2-16; V2-16 là tập con của V2-32. Cùng `variant_id` luôn tạo cùng source.

| Profile | Noisy quota/seed | Nội dung |
|---|---:|---|
| V2-8 | 8 | primitive, boundary cơ bản và hai composition chủ lực |
| V2-16 | 16 | thêm malformed IME, multi-boundary và composition |
| V2-32 | 32 | coverage đầy đủ cùng domain/hard examples |
| V2-64 | 64 | chỉ dùng nếu V2-32 còn tăng; mở rộng vị trí/observed confusions |

Không dùng Base-10/Base-25 kiểu mỗi lần random tạo tập khác. Nested profiles giúp chênh lệch chất lượng phản ánh coverage bổ sung.

## 11. Ước tính quy mô

Train có 251.241 normalized clean-query groups. Với trung bình 1,5 seed surfaces/group:

| Profile | Noisy thô tối đa | Noisy sau dedup ước tính | Tổng rows nếu clean 25% |
|---|---:|---:|---:|
| V2-8 | 3,01M | 2,4–2,8M | 3,2–3,7M |
| V2-16 | 6,03M | 4,8–5,6M | 6,4–7,5M |
| V2-32 | 12,06M | 9,0–11,0M | 12,0–14,7M |
| V2-64 | 24,12M | chưa nên giả định | quyết định sau V2-32 |

Generator phải dry-run và xuất eligibility/dedup report trước khi materialize toàn bộ.

## 12. Validation và test

- split target group trước generation;
- corruption seed validation/test độc lập với train;
- không đưa curated mapping giống hệt benchmark vào validation nếu đã train;
- dùng validation-dev 10K–20K thường xuyên;
- validation-full chỉ chạy ở checkpoint quan trọng;
- giữ Diagnostic 10K, Composition 4K, User-centric 88 và production-labelled set như frozen external benchmarks.

Validation-dev phải stratify theo primitive, composition, protected, target head/tail, độ dài query, ambiguity và brand/alphanumeric.

## 13. Metadata bắt buộc

Mỗi row cần có source/target, target group, entity, seed surface/role, macro/error family, operations và parameters, variant ID, clean/protection reason, provenance, generator version và policy version.

Ví dụ:

```json
{
  "source": "bvbạchmai",
  "target": "bệnh viện bạch mai",
  "target_group": "...",
  "seed_role": "canonical",
  "macro_family": "composition",
  "error_family": "abbreviation_boundary",
  "operations": ["abbreviation", "boundary"],
  "variant_id": "composition/abbr-boundary/0",
  "is_clean": false,
  "provenance": "synthetic",
  "generator_version": "reparos-base-v2",
  "policy_version": "correction-policy-v1"
}
```

## 14. Quality gates trước khi train

### Integrity

- group overlap bằng 0;
- exact duplicate bằng 0 sau dedup;
- noisy no-op bằng 0;
- metadata/checksum/manifest đầy đủ.

### Distribution

- quota family nằm trong tolerance;
- không fallback ngầm sang keyboard;
- target/entity không vượt cap;
- clean/protected đạt tỷ lệ;
- composition matrix đạt coverage tối thiểu.

### Plausibility

- sample ít nhất 200 rows/family;
- human review lexical/acronym/abbreviation;
- đo reject rate;
- không materialize full nếu reject rate vượt ngưỡng.

### Difficulty

- source khác target thật sự;
- query vẫn có khả năng được người dùng nhập;
- không thêm semantic information nếu mapping chưa duyệt;
- fragmentation/edit distribution không quá cực đoan.

## 15. Thí nghiệm dữ liệu đầu tiên

1. Implement dry-run/audit cho Base V2 generator.
2. Sinh V2-8 và train Base từ scratch.
3. Sinh V2-16 nested và train cùng config/model.
4. Sinh V2-32 nested và train cùng config/model.
5. So scaling curve trên cùng frozen benchmarks.
6. Chỉ sau đó quyết định V2-64, continued training hay curriculum.

Mỗi run phải chấm Exact/Recall@1, Recall@10, Improvement, Regression, clean preservation, protected accuracy, metric theo family, generation/ranking failure, repetition và checkpoint stability.

## 16. Tiêu chí chọn Base mới

Không chọn chỉ theo Diagnostic exact. Base mới phải:

- không giảm Telex/VNI so với Base hiện tại;
- tăng Composition và User-centric;
- tăng clean/protected preservation;
- giảm regression;
- giữ Recall@10 cao;
- không có target hallucination nổi bật;
- ổn định ở nhiều checkpoint cuối;
- sau đó mới đủ điều kiện làm initial checkpoint cho curriculum.

## 17. Kết luận

RTX PRO 6000 cho phép chạy scaling experiment lớn, nhưng compute không thay thế thiết kế dữ liệu. Base V2 cần chuyển từ `random weighted variants` sang `capability quota + eligibility + dedup + target cap + protected negatives + explicit correction policy`.

Mốc đầu tiên là V2-8, V2-16 và V2-32 theo cấu trúc nested. V2-32 dự kiến tạo khoảng 9–11 triệu noisy rows sau dedup, đủ lớn để kiểm tra giả thuyết data coverage mà chưa cần nhảy thẳng tới hàng trăm triệu mẫu. Chỉ khi scaling curve V2-32 chưa bão hòa mới mở rộng V2-64 hoặc tích hợp weak supervision từ query log.
