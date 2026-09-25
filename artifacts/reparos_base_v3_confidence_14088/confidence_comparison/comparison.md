# ReparoS candidate confidence comparison

Queries: 14088; development: 11021; final test: 3067.

| Method | CV Brier | Test Brier | Test top1 Brier | Test log loss | Test ECE10 |
|---|---:|---:|---:|---:|---:|
| full_all_extras | 0.03130 | 0.03182 | 0.16905 | 0.11531 | 0.00900 |
| full_plus_accent_fold | 0.03197 | 0.03229 | 0.17031 | 0.11666 | 0.01169 |
| full_plus_interactions | 0.03238 | 0.03276 | 0.17298 | 0.11932 | 0.00829 |
| full_plus_local_gap | 0.03266 | 0.03322 | 0.17677 | 0.12093 | 0.01279 |
| full_plus_nonlinear | 0.03296 | 0.03279 | 0.17187 | 0.11977 | 0.01134 |
| full_with_margin | 0.03309 | 0.03319 | 0.17491 | 0.12090 | 0.01111 |
| score_rank_margin | 0.03352 | 0.03412 | 0.18088 | 0.12465 | 0.01621 |
| score_rank_gap_margin | 0.03356 | 0.03419 | 0.18186 | 0.12492 | 0.01615 |
| full_without_margin | 0.03710 | 0.03541 | 0.19861 | 0.12579 | 0.01059 |
| score_rank_gap | 0.03808 | 0.03659 | 0.20722 | 0.12956 | 0.01198 |
| score_rank | 0.03830 | 0.03675 | 0.20889 | 0.12991 | 0.01299 |
| score_only | 0.03958 | 0.03866 | 0.21424 | 0.14065 | 0.03501 |
| rank_prior | 0.04019 | 0.03870 | 0.21871 | 0.13900 | 0.00237 |
| rank_only | 0.04019 | 0.03872 | 0.21884 | 0.13912 | 0.00308 |

Selected on CV: **full_all_extras**.

Scores measure strict gold agreement on diagnostic data, not semantic correctness on production traffic.
