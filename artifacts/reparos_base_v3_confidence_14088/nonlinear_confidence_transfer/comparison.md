# Nonlinear confidence comparison

Queries: 14088; development: 11021; test: 3067.

| Model | CV Brier all | Test Brier all | Test Brier top1 | Test ECE top1 | Test top1 mean p | Test top1 exact |
|---|---:|---:|---:|---:|---:|---:|
| logistic_full_all_extras | 0.03130 | 0.03182 | 0.16905 | 0.06712 | 0.63738 | 0.67754 |
| hgb_shallow | 0.02878 | 0.02954 | 0.14680 | 0.04829 | 0.63468 | 0.67754 |
| hgb_deep | 0.02844 | 0.02889 | 0.14252 | 0.06117 | 0.63241 | 0.67754 |

Same strict-gold labels and expected-text grouped split. The test was previously inspected; treat this comparison as exploratory.

## Leave-one-suite-out transfer

| Held-out suite | Model | All Brier | Top1 Brier | Top1 ECE |
|---|---|---:|---:|---:|
| reparos-compositional-4k | logistic_full_all_extras | 0.03822 | 0.16964 | 0.08898 |
| reparos-compositional-4k | hgb_shallow | 0.03806 | 0.16118 | 0.02935 |
| reparos-compositional-4k | hgb_deep | 0.03813 | 0.16007 | 0.04227 |
| reparos-diagnostic-10k | logistic_full_all_extras | 0.03827 | 0.23547 | 0.20167 |
| reparos-diagnostic-10k | hgb_shallow | 0.03953 | 0.23772 | 0.20981 |
| reparos-diagnostic-10k | hgb_deep | 0.03928 | 0.23146 | 0.19791 |
| reparos-user-centric-v2 | logistic_full_all_extras | 0.03704 | 0.22529 | 0.17346 |
| reparos-user-centric-v2 | hgb_shallow | 0.03799 | 0.23317 | 0.21388 |
| reparos-user-centric-v2 | hgb_deep | 0.03754 | 0.22812 | 0.21282 |
