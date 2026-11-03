# annstat report (classification, n_eval=171, hidden neurons=16)

## Performance
- **wilson**: 0.9708  [0.9334, 0.9874]
- **clopper-pearson**: 0.9708  [0.9331, 0.9904]
- **bootstrap-percentile**: 0.9708  [0.9474, 0.9942]

## Neuron-response association (BH-adjusted)
|   neuron |   statistic |   p_value |   effect | dead   |   p_adj | significant   |
|---------:|------------:|----------:|---------:|:-------|--------:|:--------------|
|        0 |    105.021  |         0 |   0.6178 | False  |       0 | True          |
|        1 |    143.299  |         0 |   0.8429 | False  |       0 | True          |
|        2 |    115.682  |         0 |   0.6805 | False  |       0 | True          |
|        3 |    112.172  |         0 |   0.6598 | False  |       0 | True          |
|        4 |    126.337  |         0 |   0.7432 | False  |       0 | True          |
|        5 |     52.7813 |         0 |   0.3105 | False  |       0 | True          |
|        6 |    122.086  |         0 |   0.7182 | False  |       0 | True          |
|        7 |    120.896  |         0 |   0.7112 | False  |       0 | True          |
|        8 |    122.245  |         0 |   0.7191 | False  |       0 | True          |
|        9 |    120.296  |         0 |   0.7076 | False  |       0 | True          |
|       10 |     84.3984 |         0 |   0.4965 | False  |       0 | True          |
|       11 |    100.203  |         0 |   0.5894 | False  |       0 | True          |
|       12 |     95.3975 |         0 |   0.5612 | False  |       0 | True          |
|       13 |    130.079  |         0 |   0.7652 | False  |       0 | True          |
|       14 |    113.438  |         0 |   0.6673 | False  |       0 | True          |
|       15 |     57.9303 |         0 |   0.3408 | False  |       0 | True          |

## Ablation importance (paired bootstrap, BH-adjusted)
|   neuron |   loss_increase |   ci_low |   ci_high |   p_value |   accuracy_drop |   p_adj | significant   |
|---------:|----------------:|---------:|----------:|----------:|----------------:|--------:|:--------------|
|        0 |          0      |  -0      |    0.0001 |    0.1868 |          0      |  0.4176 | False         |
|        1 |          0.0027 |  -0.0031 |    0.0092 |    0.2088 |         -0.0058 |  0.4176 | False         |
|        2 |          0.0032 |  -0.0029 |    0.0096 |    0.1698 |         -0.0058 |  0.4176 | False         |
|        3 |          0.0134 |   0.0061 |    0.0216 |    0.001  |          0.0058 |  0.008  | True          |
|        4 |          0.0001 |  -0.004  |    0.0032 |    0.4456 |          0      |  0.7921 | False         |
|        5 |         -0.0003 |  -0.0013 |    0.0008 |    0.7562 |          0      |  0.9545 | False         |
|        6 |          0.0158 |   0.0086 |    0.0229 |    0.001  |          0      |  0.008  | True          |
|        7 |          0.0127 |   0.0036 |    0.0204 |    0.008  |          0      |  0.0256 | True          |
|        8 |         -0.0016 |  -0.0092 |    0.0055 |    0.6743 |         -0.0117 |  0.9545 | False         |
|        9 |          0.0163 |   0.0034 |    0.0264 |    0.004  |          0      |  0.0213 | True          |
|       10 |          0.0068 |   0.002  |    0.0117 |    0.006  |          0      |  0.024  | False         |
|       11 |         -0.0024 |  -0.0081 |    0.0012 |    0.8352 |          0      |  0.9545 | False         |
|       12 |         -0.0017 |  -0.0044 |    0.0002 |    0.957  |          0      |  0.981  | False         |
|       13 |         -0.0027 |  -0.0142 |    0.007  |    0.6873 |         -0.0117 |  0.9545 | False         |
|       14 |         -0.0014 |  -0.0055 |    0.0016 |    0.8092 |         -0.0058 |  0.9545 | False         |
|       15 |         -0.0022 |  -0.0047 |   -0.0001 |    0.981  |         -0.0058 |  0.981  | False         |

## Redundancy
- effective rank 2.50 of 16 (participation ratio 1.70); dead neurons 0; highly correlated pairs 3

## Input permutation tests
| feature                 |   loss_increase |   p_value |   p_adj | significant   |
|:------------------------|----------------:|----------:|--------:|:--------------|
| mean radius             |          0.0118 |    0.1542 |  0.2722 | False         |
| mean texture            |          0.0122 |    0.0498 |  0.1628 | False         |
| mean perimeter          |          0.0026 |    0.3085 |  0.3701 | False         |
| mean area               |          0.0223 |    0.0398 |  0.1628 | False         |
| mean smoothness         |          0.0056 |    0.1194 |  0.2559 | False         |
| mean compactness        |         -0.0001 |    0.5274 |  0.565  | False         |
| mean concavity          |          0.0125 |    0.0448 |  0.1628 | False         |
| mean concave points     |          0.015  |    0.0597 |  0.1628 | False         |
| mean symmetry           |          0.002  |    0.3234 |  0.3731 | False         |
| mean fractal dimension  |          0.0086 |    0.1343 |  0.2687 | False         |
| radius error            |          0.0652 |    0.005  |  0.0498 | True          |
| texture error           |          0.0019 |    0.3582 |  0.398  | False         |
| perimeter error         |          0.025  |    0.01   |  0.0597 | False         |
| area error              |          0.1434 |    0.005  |  0.0498 | True          |
| smoothness error        |         -0.0009 |    0.7413 |  0.7669 | False         |
| compactness error       |          0.0031 |    0.2836 |  0.3669 | False         |
| concavity error         |          0.0014 |    0.0547 |  0.1628 | False         |
| concave points error    |          0.0019 |    0.1542 |  0.2722 | False         |
| symmetry error          |          0.0025 |    0.2687 |  0.3669 | False         |
| fractal dimension error |          0.0031 |    0.2736 |  0.3669 | False         |
| worst radius            |          0.015  |    0.0846 |  0.2114 | False         |
| worst texture           |          0.0635 |    0.005  |  0.0498 | True          |
| worst perimeter         |          0.0034 |    0.204  |  0.34   | False         |
| worst area              |          0.0387 |    0.0199 |  0.0995 | False         |
| worst smoothness        |          0.0277 |    0.01   |  0.0597 | False         |
| worst compactness       |          0.0034 |    0.2637 |  0.3669 | False         |
| worst concavity         |          0.008  |    0.0945 |  0.2181 | False         |
| worst concave points    |          0.0043 |    0.2587 |  0.3669 | False         |
| worst symmetry          |          0.0053 |    0.2935 |  0.3669 | False         |
| worst fractal dimension |         -0.0014 |    0.7861 |  0.7861 | False         |

## Logistic surrogate on hidden neurons
- McFadden R2 0.974, AIC 47.8, BIC 115.6, LR-test p = 1.23e-98; fidelity to network: {'agreement': 0.9707602339181286, 'surrogate_accuracy': 0.9532163742690059, 'network_accuracy': 0.9707602339181286}

## Reduction path
| method         |   k |   k_requested |   metric |   agreement |   n_neurons_needed |   n_head_params |   n_hidden_layer_params_removed |
|:---------------|----:|--------------:|---------:|------------:|-------------------:|----------------:|--------------------------------:|
| full           |  16 |            16 |   0.9708 |      1      |                 16 |              34 |                               0 |
| lsc_qr         |   2 |             2 |   0.9532 |      0.9708 |                  2 |               6 |                              14 |
| lsc_importance |   2 |             2 |   0.9474 |      0.9649 |                  2 |               6 |                              14 |
| cluster_mean   |   2 |             2 |   0.9766 |      0.9942 |                 16 |               6 |                               0 |
| pca            |   2 |             2 |   0.9766 |      0.9942 |                 16 |               6 |                               0 |
| lsc_qr         |   3 |             3 |   0.9825 |      0.9766 |                  3 |               8 |                              13 |
| lsc_importance |   3 |             3 |   0.9532 |      0.9825 |                  3 |               8 |                              13 |
| cluster_mean   |   3 |             3 |   0.9766 |      0.9942 |                 16 |               8 |                               0 |
| pca            |   3 |             3 |   0.9766 |      0.9942 |                 16 |               8 |                               0 |
| lsc_qr         |   4 |             4 |   0.9708 |      1      |                  4 |              10 |                              12 |
| lsc_importance |   4 |             4 |   0.9474 |      0.9766 |                  4 |              10 |                              12 |
| cluster_mean   |   4 |             4 |   0.9766 |      0.9942 |                 16 |              10 |                               0 |
| pca            |   4 |             4 |   0.9591 |      0.9883 |                 16 |              10 |                               0 |
| lsc_qr         |   6 |             6 |   0.9708 |      1      |                  6 |              14 |                              10 |
| lsc_importance |   6 |             6 |   0.9591 |      0.9883 |                  6 |              14 |                              10 |
| cluster_mean   |   6 |             6 |   0.9766 |      0.9825 |                 16 |              14 |                               0 |
| pca            |   6 |             6 |   0.9532 |      0.9825 |                 16 |              14 |                               0 |
| lsc_qr         |   8 |             8 |   0.9708 |      1      |                  8 |              18 |                               8 |
| lsc_importance |   8 |             8 |   0.9708 |      1      |                  8 |              18 |                               8 |
| cluster_mean   |   8 |             8 |   0.9649 |      0.9825 |                 16 |              18 |                               0 |
| pca            |   8 |             8 |   0.9474 |      0.9766 |                 16 |              18 |                               0 |

## Suggestions
- neurons whose individual ablation does not significantly raise the loss (pruning candidates): [0, 1, 2, 4, 5, 8, 10, 11, 12, 13, 14, 15]
- dead neurons: []