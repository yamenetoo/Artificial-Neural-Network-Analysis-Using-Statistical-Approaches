# annstat report (classification, n_eval=30, hidden neurons=10)

## Performance
- **wilson**: 0.9667  [0.8333, 0.9941]
- **clopper-pearson**: 0.9667  [0.8278, 0.9992]
- **bootstrap-percentile**: 0.9667  [0.9000, 1.0000]

## Neuron-response association (BH-adjusted)
|   neuron |   statistic |   p_value |   effect | dead   |   p_adj | significant   |
|---------:|------------:|----------:|---------:|:-------|--------:|:--------------|
|        0 |     24.4843 |    0      |   0.8443 | False  |  0      | True          |
|        1 |     26.2648 |    0      |   0.9057 | False  |  0      | True          |
|        2 |     26.0179 |    0      |   0.8972 | False  |  0      | True          |
|        3 |     23.8477 |    0      |   0.8223 | False  |  0      | True          |
|        4 |     24.3572 |    0      |   0.8399 | False  |  0      | True          |
|        5 |     18.7954 |    0.0001 |   0.6481 | False  |  0.0001 | True          |
|        6 |     24.1188 |    0      |   0.8317 | False  |  0      | True          |
|        7 |     20.2881 |    0      |   0.6996 | False  |  0      | True          |
|        8 |     23.8655 |    0      |   0.8229 | False  |  0      | True          |
|        9 |     25.0028 |    0      |   0.8622 | False  |  0      | True          |

## Ablation importance (paired bootstrap, BH-adjusted)
|   neuron |   loss_increase |   ci_low |   ci_high |   p_value |   accuracy_drop |   p_adj | significant   |
|---------:|----------------:|---------:|----------:|----------:|----------------:|--------:|:--------------|
|        0 |          0.0557 |  -0.0234 |    0.1395 |    0.0789 |          0.0333 |  0.0987 | False         |
|        1 |          0.0322 |  -0.0221 |    0.0871 |    0.1139 |          0      |  0.1265 | False         |
|        2 |          0.0031 |   0.0012 |    0.0055 |    0.001  |          0      |  0.005  | False         |
|        3 |          0.0001 |   0      |    0.0003 |    0.001  |          0      |  0.005  | False         |
|        4 |          0.0122 |   0.0013 |    0.0239 |    0.016  |          0      |  0.0266 | True          |
|        5 |          0.2292 |   0.064  |    0.4313 |    0.003  |          0.1    |  0.0075 | True          |
|        6 |          0.1999 |   0.0604 |    0.3421 |    0.006  |          0.1333 |  0.012  | True          |
|        7 |          0.0114 |  -0.0058 |    0.0347 |    0.1329 |          0      |  0.1329 | False         |
|        8 |          0.0316 |  -0.0138 |    0.0721 |    0.0779 |          0      |  0.0987 | False         |
|        9 |          0.0054 |   0.0016 |    0.0097 |    0.003  |          0      |  0.0075 | False         |

## Redundancy
- effective rank 2.02 of 10 (participation ratio 1.52); dead neurons 0; highly correlated pairs 4

## Contribution tests - omnibus
|   neuron | test     |   statistic |   p_value |   p_adj | significant   |
|---------:|:---------|------------:|----------:|--------:|:--------------|
|        0 | friedman |          36 |    0      |  0      | True          |
|        1 | friedman |          40 |    0      |  0      | True          |
|        2 | friedman |          22 |    0      |  0      | True          |
|        3 | friedman |          18 |    0.0001 |  0.0001 | True          |
|        4 | friedman |          26 |    0      |  0      | True          |
|        5 | friedman |          52 |    0      |  0      | True          |
|        6 | friedman |          32 |    0      |  0      | True          |
|        7 | friedman |          34 |    0      |  0      | True          |
|        8 | friedman |          34 |    0      |  0      | True          |
|        9 | friedman |          24 |    0      |  0      | True          |

## Input permutation tests
| feature           |   loss_increase |   p_value |   p_adj | significant   |
|:------------------|----------------:|----------:|--------:|:--------------|
| sepal length (cm) |          0.0297 |     0.005 |   0.005 | True          |
| sepal width (cm)  |          0.1486 |     0.005 |   0.005 | True          |
| petal length (cm) |          0.4258 |     0.005 |   0.005 | True          |
| petal width (cm)  |          0.6561 |     0.005 |   0.005 | True          |

## Logistic surrogate on hidden neurons
- McFadden R2 0.979, AIC 49.5, BIC 110.9, LR-test p = 2.6e-43; fidelity to network: {'agreement': 0.9333333333333333, 'surrogate_accuracy': 0.9, 'network_accuracy': 0.9666666666666667}

## Reduction path
| method         |   k |   k_requested |   metric |   agreement |   n_neurons_needed |   n_head_params |   n_hidden_layer_params_removed |
|:---------------|----:|--------------:|---------:|------------:|-------------------:|----------------:|--------------------------------:|
| full           |  10 |            10 |   0.9667 |      1      |                 10 |              33 |                               0 |
| lsc_qr         |   2 |             2 |   0.7667 |      0.8    |                  2 |               9 |                               8 |
| lsc_importance |   2 |             2 |   0.8333 |      0.8667 |                  2 |               9 |                               8 |
| cluster_mean   |   2 |             2 |   0.9    |      0.9333 |                 10 |               9 |                               0 |
| pca            |   2 |             2 |   0.9    |      0.9333 |                 10 |               9 |                               0 |
| lsc_qr         |   3 |             3 |   0.7667 |      0.8    |                  3 |              12 |                               7 |
| lsc_importance |   3 |             3 |   0.9333 |      0.9667 |                  3 |              12 |                               7 |
| cluster_mean   |   3 |             3 |   0.9    |      0.9333 |                 10 |              12 |                               0 |
| pca            |   3 |             3 |   0.9333 |      0.9667 |                 10 |              12 |                               0 |
| lsc_qr         |   4 |             4 |   0.9667 |      1      |                  4 |              15 |                               6 |
| lsc_importance |   4 |             4 |   0.9667 |      1      |                  4 |              15 |                               6 |
| cluster_mean   |   4 |             4 |   0.9667 |      1      |                 10 |              15 |                               0 |
| pca            |   4 |             4 |   0.9333 |      0.9667 |                 10 |              15 |                               0 |
| lsc_qr         |   6 |             6 |   0.9667 |      1      |                  6 |              21 |                               4 |
| lsc_importance |   6 |             6 |   0.9667 |      1      |                  6 |              21 |                               4 |
| cluster_mean   |   6 |             6 |   0.9667 |      1      |                 10 |              21 |                               0 |
| pca            |   6 |             6 |   0.9333 |      0.9667 |                 10 |              21 |                               0 |
| lsc_qr         |   8 |             8 |   0.9667 |      1      |                  8 |              27 |                               2 |
| lsc_importance |   8 |             8 |   0.9667 |      1      |                  8 |              27 |                               2 |
| cluster_mean   |   8 |             8 |   0.9667 |      1      |                 10 |              27 |                               0 |
| pca            |   8 |             8 |   0.9667 |      1      |                 10 |              27 |                               0 |

## Suggestions
- neurons whose individual ablation does not significantly raise the loss (pruning candidates): [0, 1, 2, 3, 7, 8, 9]
- dead neurons: []