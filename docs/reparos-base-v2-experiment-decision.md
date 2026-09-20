# Quyết định thí nghiệm ReparoS Base V2

**Ngày:** 19/09/2026  
**Tài liệu liên quan:** `reparos-current-quality-root-cause-analysis.md`, `reparos-base-v2-data-design.md`  
**Trạng thái:** Addendum có độ ưu tiên cao hơn thứ tự execution trong hai tài liệu trước

## 1. Các kết luận được xác nhận

Root-cause hiện tại chỉ đúng nút thắt chính: bằng chứng đang nghiêng về data/training strategy hơn capacity.

- Base đạt Diagnostic tương đối cao nhưng yếu hơn nhiều trên Composition và User-centric.
- Base all-joined chỉ đạt 2,8%; abbreviation, location acronym và clean-protected trong User-centric đều 0%.
- Curriculum làm capability được nhắm tới tăng mạnh: location acronym 0→100%, abbreviation+boundary 0→75%, all-joined 2,8→43,4%.
- Đồng thời VNI giảm 88,38→48,13% và Diagnostic noisy giảm gần 14 điểm phần trăm.

Mẫu hình này xác nhận model 1/1 vẫn học được capability mới, nhưng phân phối fine-tune làm quên capability cũ. Capacity có thể đặt trần về sau nhưng chưa phải thứ đầu tiên cần sửa.

## 2. Vai trò của Base V2

Base V2 giải quyết trực tiếp các root cause bằng cách thay:

```text
N random variants/query
```

bằng:

```text
capability quota
+ eligibility
+ dedup
+ target/entity cap
+ protected hard negatives
+ explicit correction policy
```

Đơn vị sampling chuyển sang `target_group`, tối đa ba seed surfaces trên mỗi target. Đây là thay đổi kiến trúc dữ liệu quan trọng vì nó hạn chế target prior do OSM object/alias lặp. Split vẫn phải thực hiện theo normalized target group trước khi sinh noise.

Các profile V2 phải nested:

```text
V2-8 ⊂ V2-16 ⊂ V2-32
```

Cùng một `variant_id` phải sinh cùng source. Nhờ vậy scaling curve phản ánh block coverage được thêm vào thay vì khác biệt ngẫu nhiên giữa hai dataset.

## 3. Giới hạn cần ghi rõ: quota là hand-designed hypothesis

Quota V2-32 hiện tại — 10 primitive, 5 boundary, 10 composition, 5 domain và 2 observed/hard reserve — là giả thuyết để cân bằng capability. Nó **không phải** phân phối production đã được đo.

Cần giữ tách biệt:

```text
training coverage distribution != production frequency distribution
```

- Training coverage distribution bảo đảm Base học đủ capability, đặc biệt nhóm khó hoặc hiếm.
- Production frequency distribution mô tả traffic thật, chỉ có thể xác định đáng tin cậy từ query log, reformulation, click-through và human-labelled samples.

Không cần thay taxonomy Base khi có traffic. Production data sẽ được dùng để reweight, calibrate hoặc làm curriculum/weak supervision sau này.

Mọi manifest V2 phải ghi:

```json
{
  "distribution_objective": "capability_balanced",
  "production_distribution_claimed": false,
  "quota_status": "design_hypothesis"
}
```

## 4. Không materialize V2-32 ngay

Trước khi tạo 9–11 triệu noisy rows, phải đóng câu hỏi Base hiện tại đã converge chưa.

Hai thí nghiệm trả lời hai câu khác nhau:

```text
Current Base continued training
→ Dữ liệu hiện tại có bị undertrained không?

V2 scaling
→ Thiết kế coverage mới có nâng trần quality không?
```

Nếu bỏ continued-training control, khi V2 tốt hơn sẽ không biết gain đến từ data design hay chỉ do V2 được train với budget hiệu quả hơn.

## 5. Decision tree và thứ tự execution bắt buộc

### Phase 1 — Audit Base hiện tại

Chấm checkpoint 15K→25K trên:

- Diagnostic 10K;
- Composition 4K;
- User-centric 88;
- Recall@1 và Recall@10;
- clean preservation;
- regression;
- repetition.

Sau đó continue-train chính Base 25K lên 35K/50K mà không đổi dữ liệu. Đây là convergence control.

### Phase 2 — Audit generator/data hiện tại

Đo:

- duplicate và no-op;
- normalized source→multiple targets;
- top target/entity concentration;
- aliases và seed surfaces trên mỗi target;
- unique variants/group;
- capability coverage matrix;
- tỷ lệ fallback sang keyboard edit;
- số operation ineligible theo family.

### Phase 3 — Implement V2 dry-run

Chưa ghi hàng triệu pair. Dry-run phải xuất:

- eligible/ineligible counts;
- rejected/no-op counts;
- duplicate rate;
- achieved quota theo family;
- target/entity concentration;
- source conflict report;
- projected row count/disk size;
- sample tối thiểu 200 rows/family cho human review.

### Phase 4 — Train scaling ablation

Chỉ materialize khi dry-run pass gate:

1. V2-8 train from scratch.
2. V2-16 nested, cùng model/protocol.
3. V2-32 nested nếu V2-16 tạo gain có ý nghĩa.
4. V2-64 hoặc observed-query data chỉ khi curve chưa bão hòa.
5. Curriculum/replay chỉ quay lại sau khi Base mới ổn định.

## 6. Ma trận đối chứng tối thiểu

| Run | Dataset | Câu hỏi |
|---|---|---|
| B15–B25 | Base hiện tại, retained checkpoints | Base đã ổn định chưa? |
| B35/B50 | Base hiện tại, continued training | Base có bị undertrained không? |
| V2-8 | Capability-balanced tối thiểu | Thiết kế V2 có giúp không? |
| V2-16 | Nested extension của V2-8 | Coverage bổ sung có gain không? |
| V2-32 | Nested extension của V2-16 | Scaling đã bão hòa chưa? |

Các run phải so theo hai chế độ:

1. **Cùng training budget:** cùng optimizer steps hoặc training tokens để đo data efficiency.
2. **Best converged checkpoint:** đo trần quality của từng dataset.

Không được so V2 train lâu hơn với Base 25K rồi quy toàn bộ gain cho data design.

## 7. Tiêu chí quyết định

### Base hiện tại bị undertrained

Kết luận này chỉ đúng nếu B35/B50 cải thiện ổn định trên nhiều bộ test mà không làm regression/clean preservation xấu đi đáng kể.

### Base V2 tốt hơn do thiết kế dữ liệu

Kết luận này cần:

- V2 tốt hơn ở cùng training budget;
- gain xuất hiện đúng family được bổ sung;
- không chỉ do target memorization;
- protected accuracy và regression cùng tốt hơn;
- checkpoint cuối ổn định.

### Cần mở rộng V2-32/V2-64

Chỉ mở rộng nếu V2-8→V2-16→V2-32 còn tạo marginal gain rõ ràng. Nếu curve bão hòa, cần cải thiện realism/observed data hoặc objective thay vì tiếp tục sinh synthetic.

## 8. Quyết định cuối

Thứ tự hiện tại được chốt là:

```text
audit Base checkpoints
→ continued-training control
→ audit generator/data cũ
→ V2 dry-run
→ V2-8
→ V2-16
→ V2-32 nếu cần
→ chọn Base ổn định
→ curriculum/replay
```

RTX PRO 6000 loại bỏ phần lớn rào cản compute trong giai đoạn research, nhưng không thay đổi yêu cầu về causal experiment design. Chưa materialize V2-32 cho tới khi Base convergence control và V2 dry-run audit hoàn tất.
