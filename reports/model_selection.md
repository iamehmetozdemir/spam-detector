| Model | Features | Setting | CV F1 | ± | CV precision | CV recall |
|---|---|---|---|---|---|---|
| Logistic Regression | words + chars | C=10 | 0.9955 | 0.0017 | 0.9965 | 0.9945 |
| Logistic Regression | words + chars | C=100 | 0.9947 | 0.0017 | 0.9958 | 0.9936 |
| Logistic Regression | words + chars | C=1 | 0.9937 | 0.0014 | 0.9949 | 0.9926 |
| Logistic Regression | words | C=10 | 0.9900 | 0.0023 | 0.9907 | 0.9894 |
| Logistic Regression | words | C=100 | 0.9888 | 0.0025 | 0.9888 | 0.9888 |
| Logistic Regression | words | C=1 | 0.9856 | 0.0027 | 0.9853 | 0.9859 |
| Naive Bayes | words | alpha=0.01 | 0.9727 | 0.0035 | 0.9688 | 0.9766 |
| Naive Bayes | words | alpha=0.1 | 0.9710 | 0.0038 | 0.9751 | 0.9669 |
| Naive Bayes | words + chars | alpha=0.01 | 0.9395 | 0.0042 | 0.9596 | 0.9204 |
| Naive Bayes | words | alpha=1.0 | 0.9264 | 0.0086 | 0.9838 | 0.8754 |
| Naive Bayes | words + chars | alpha=0.1 | 0.9146 | 0.0063 | 0.9512 | 0.8809 |
| Naive Bayes | words + chars | alpha=1.0 | 0.8463 | 0.0114 | 0.9597 | 0.7570 |
