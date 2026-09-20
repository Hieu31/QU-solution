# Audit dữ liệu zero-click

**Nguồn:** `data/zero_click.csv`  
**Ngày audit:** 19/09/2026  
**Phạm vi:** Phân tích read-only; chưa sửa, lọc hoặc xuất lại dữ liệu gốc

## 1. Kết luận chính

`zero_click.csv` là log truy vấn không có kết quả hoặc không phát sinh click, kèm vị trí người dùng. Đây là nguồn dữ liệu production rất có giá trị để hiểu query distribution, xây benchmark thực tế và khai thác weak supervision.

Tuy nhiên, file **không phải tập correction có nhãn**. Nó không có:

- expected/corrected query;
- kết quả search đã hiển thị;
- POI người dùng cuối cùng chọn;
- timestamp;
- session/user ID;
- query tiếp theo trong reformulation chain;
- lý do zero-click.

Vì vậy không được đưa trực tiếp `keyword → OSM entity gần nhất` vào train. Zero-click có thể xảy ra do typo, POI chưa có trong index, query quá chung, người dùng đang gõ dở, sai vùng, vấn đề ranking hoặc đơn giản là người dùng không click.

## 2. Schema và quy mô

| Thuộc tính | Giá trị |
|---|---:|
| File size | 204.535.047 bytes, khoảng 195,06 MiB |
| Rows | 3.549.549 |
| Columns | `keyword`, `user_lat`, `user_lng` |
| Blank keyword | 0 |
| Invalid/missing coordinate | 0 |
| Unique keyword nguyên bản | 1.660.687 |
| Unique keyword sau NFC + casefold + collapse-space | 1.563.508 |
| Unique coordinates chính xác | 1.173.687 |

File đọc đúng bằng UTF-8. Việc PowerShell từng hiển thị `khÃ¡ch` là lỗi hiển thị console, không phải toàn bộ file đã bị double-encoded. Chỉ 332 rows chứa các marker giống mojibake; cần review riêng thay vì sửa encoding hàng loạt.

## 3. Tần suất và duplication

| Metric | Giá trị |
|---|---:|
| Unique normalized queries | 1.563.508 |
| Chỉ xuất hiện một lần | 1.212.396 |
| Xuất hiện ít nhất 2 lần | 351.112 |
| Xuất hiện ít nhất 5 lần | 75.340 |
| Xuất hiện ít nhất 10 lần | 29.001 |
| Xuất hiện ít nhất 100 lần | 1.878 |

Khoảng 77,5% unique queries chỉ xuất hiện một lần. Đây là long-tail rất lớn.

Traffic concentration:

| Head queries | Tỷ lệ events được bao phủ |
|---:|---:|
| Top 10 | 2,86% |
| Top 100 | 8,61% |
| Top 1.000 | 18,17% |
| Top 10.000 | 31,10% |
| Top 100.000 | 49,65% |

Các query phổ biến gồm `bệnh viện`, `trường`, `bệnh`, `ngõ`, `nhà`, `chợ`, `bến`, `bến xe`, `sân bay`, `cầu`, `cổng`, `chung cư`, `quán`, `vincom`, `đại học` và nhiều query ngắn khác.

Điều này cho thấy file chứa cả:

- query hoàn chỉnh;
- category query;
- prefix/partial query trong lúc gõ;
- query không đủ định danh một POI;
- typo hoặc query thiếu dấu;
- địa chỉ có số;
- nội dung chat/paste dài.

Không được mặc định mọi row là lỗi chính tả.

## 4. Độ dài và cấu trúc query

| Nhóm | Events | Tỷ lệ xấp xỉ |
|---|---:|---:|
| Một token | 788.587 | 22,22% |
| Dài không quá 3 ký tự | 381.866 | 10,76% |
| Có ít nhất một chữ số | 1.437.437 | 40,50% |
| Chỉ gồm chữ số | 111.173 | 3,13% |
| Dài từ 100 ký tự | 14.474 | 0,41% |
| Chỉ punctuation | 14 | rất nhỏ |

Token-count distribution:

| Tokens | Events |
|---:|---:|
| 1 | 788.587 |
| 2 | 831.904 |
| 3 | 610.905 |
| 4 | 474.121 |
| 5 | 281.138 |
| 6 | 162.158 |
| 7 | 109.239 |
| 8 | 72.990 |
| 9 | 50.589 |
| 10+ | 167.918 |

Một ví dụ query dài là toàn bộ câu chat bán hàng kèm địa chỉ. Nhóm này cần được lọc khỏi spell-correction training hoặc chuyển sang lane riêng cho address extraction.

## 5. Dấu tiếng Việt và input form

| Nhóm | Events | Tỷ lệ xấp xỉ |
|---|---:|---:|
| Có dấu tiếng Việt | 2.599.375 | 73,23% |
| ASCII/không dấu | 950.174 | 26,77% |

Tỷ lệ có dấu cao cho thấy zero-click không đồng nghĩa missing diacritics. Nhiều query sạch vẫn zero-click vì index/ranking/coverage hoặc query intent quá chung.

File có thể dùng để ước lượng production mix của query có dấu/không dấu, nhưng chưa đủ để phân loại Telex, VNI, typo hay boundary nếu không có target đúng.

## 6. Mức overlap với clean OSM queries

Clean targets từ `prepared-v4-leakfree`:

- 313.998 normalized targets;
- 309.220 targets sau bỏ dấu.

Zero-click overlap:

| Kiểu match | Events | Tỷ lệ events | Unique zero-click queries |
|---|---:|---:|---:|
| Exact normalized OSM target | 661.297 | 18,63% | 37.907 |
| Match sau bỏ dấu | 817.751 | 23,04% | 62.539 |

Ý nghĩa:

1. Ít nhất 18,63% events đã giống hệt một clean OSM target nhưng vẫn zero-click. Đây không phải lỗi spelling theo định nghĩa hiện tại.
2. Khoảng 23,04% events có thể khớp một OSM target nếu bỏ qua dấu, nhưng match bỏ dấu có thể one-to-many và chưa đủ làm ground truth.
3. Phần còn lại không nhất thiết là typo; có thể là POI ngoài OSM, địa chỉ chi tiết, query đang gõ dở, business name mới, số điện thoại hoặc nội dung ngoài domain.

Nhóm exact-clean zero-click là hard negative/diagnostic quan trọng: spell model không nên tự ý sửa chúng chỉ vì search trả zero result.

## 7. Tọa độ và rủi ro riêng tư

| Thuộc tính | Giá trị |
|---|---:|
| Rows trong bounding box Việt Nam rộng | 3.548.135 |
| Rows ngoài bounding box | 1.414 |
| Latitude range | -8,8281 đến 51,0281 |
| Longitude range | -118,2439 đến 125,6309 |
| Unique exact coordinates | 1.173.687 |
| Số rows lớn nhất tại cùng một coordinate | 1.395 |

Tọa độ có độ chính xác cao, nhiều giá trị có hơn sáu chữ số thập phân. Đây có thể là vị trí thiết bị và là dữ liệu nhạy cảm. Ngoài ra phát hiện:

- 859 rows có chuỗi giống số điện thoại;
- 2 rows có chuỗi giống email;
- 14.474 rows dài từ 100 ký tự, có thể chứa nội dung paste hoặc thông tin cá nhân.

Yêu cầu xử lý:

- không commit bản raw lên Git;
- không đưa latitude/longitude raw vào model;
- không xuất ví dụ chứa số điện thoại/email/địa chỉ nhà riêng vào báo cáo;
- nếu dùng vị trí để weak-label, xử lý tạm thời rồi chỉ lưu coarse grid/geohash hoặc khoảng cách;
- áp dụng retention/access policy của doanh nghiệp;
- chạy PII filtering trước khi tạo corpus nghiên cứu.

## 8. File này dùng được cho việc gì?

### 8.1. Production-weighted benchmark sampling

Lấy mẫu theo frequency/head-tail và human-label:

- query có cần correction không;
- expected correction;
- giữ nguyên hay sửa;
- intent/category;
- expected POI nếu đủ thông tin;
- nguyên nhân zero-click.

Đây là ứng dụng có giá trị cao nhất và ít giả định nhất.

### 8.2. Hard-negative mining

Các zero-click query exact-match clean OSM hoặc được human xác nhận sạch có thể bổ sung vào protected set. Chúng dạy model rằng zero result không phải lý do tự động sửa query.

### 8.3. Traffic prior và evaluation weighting

Tần suất keyword có thể dùng để:

- chia head/tail;
- weight benchmark;
- ưu tiên annotation;
- đo coverage của taxonomy Base V2;
- so sánh capability-balanced training distribution với production distribution.

Không dùng traffic frequency để lặp thô cùng target hàng nghìn lần trong training.

### 8.4. Weak label bằng OSM và GPS

Có thể tạo candidate labels với pipeline:

```text
zero-click keyword
→ PII/filtering
→ candidate retrieval trong bán kính quanh coarse location
→ lexical/phonetic/boundary similarity
→ ambiguity checks
→ confidence tiers
→ human review
→ accepted weak pairs
```

Chỉ auto-accept khi candidate gần như duy nhất và vượt ngưỡng mạnh. GPS gần POI không chứng minh POI đó là intent; cần tránh nearest-POI labeling ngây thơ.

### 8.5. Noise realism study

Sau khi một subset được label, có thể đo production distribution thật của:

- missing diacritics;
- Telex/VNI leakage;
- keyboard typo;
- boundary;
- abbreviation/acronym;
- number/symbol;
- clean/protected;
- out-of-domain/incomplete query.

Kết quả này dùng để reweight Base V2 hoặc curriculum, thay vì thay taxonomy dựa trên phỏng đoán.

## 9. File này chưa dùng được cho việc gì?

Không được dùng trực tiếp cho:

- `keyword → nearest OSM entity` supervised pairs;
- tự coi mọi zero-click là misspelled;
- train correction model khi chưa có target;
- tạo query reformulation chain vì không có timestamp/session ID;
- đánh giá correction accuracy vì không có expected output;
- đưa raw GPS vào feature/model;
- công bố ví dụ raw chưa lọc PII.

## 10. Annotation proposal

Tạo tập review đầu tiên khoảng 5.000–10.000 unique queries, stratified:

| Stratum | Mục tiêu |
|---|---:|
| Head frequent | 20% |
| Random tail | 20% |
| Exact OSM match | 15% |
| Plain/diacritic OSM match | 15% |
| Short ambiguous | 10% |
| Number/address | 10% |
| High-confidence fuzzy nearby candidate | 10% |

Label schema:

```text
action: keep | correct | incomplete | out_of_domain | unknown
expected_query: nullable
intent_type
error_families: multi-label
candidate_entity_id: nullable
annotator_confidence
contains_pii
notes
```

Hai annotator nên review nhóm ambiguous/weak-label quan trọng; disagreement được adjudicate.

## 11. Quyết định

Giữ `zero_click.csv`. Đây có thể là nguồn production evidence quan trọng nhất hiện có, nhưng phải xem nó là **unlabelled sensitive query log**, không phải correction pairs.

Thứ tự khai thác phù hợp:

```text
PII/privacy audit
→ normalize và deduplicate
→ traffic/coverage report
→ stratified annotation
→ build production-weighted benchmark
→ weak-label candidate mining
→ reviewed hard negatives/pairs
→ reweight Base V2 hoặc curriculum
```
