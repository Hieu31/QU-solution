# Nonlinear confidence comparison

Queries: 14088; development: 11021; test: 3067.

| Model | CV Brier all | Test Brier all | Test Brier top1 | Test ECE top1 | Test top1 mean p | Test top1 exact |
|---|---:|---:|---:|---:|---:|---:|
| logistic_full_all_extras | 0.03130 | 0.03182 | 0.16905 | 0.06712 | 0.63738 | 0.67754 |
| hgb_shallow | 0.02878 | 0.02954 | 0.14680 | 0.04829 | 0.63468 | 0.67754 |
| hgb_deep | 0.02844 | 0.02889 | 0.14252 | 0.06117 | 0.63241 | 0.67754 |
| random_forest | 0.02988 | 0.02972 | 0.14705 | 0.05727 | 0.62830 | 0.67754 |

Same strict-gold labels and expected-text grouped split. The test was previously inspected; treat this comparison as exploratory.
