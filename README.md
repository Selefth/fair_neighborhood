
# Optimizing Neighborhoods for Fair Top-N Recommendation

## Abstract
This paper addresses demographic bias in *neighborhood-learning* models for collaborative filtering recommendations. Despite their superior ranking performance, these methods can learn neighborhoods that inadvertently foster discriminatory patterns. There is limited research in this area, highlighting an important research gap. A notable yet solitary effort, *Balanced Neighborhood Sparse Linear Method (BNSLIM)*, aims at balancing neighborhood influence across different demographic groups. However, BNSLIM is hampered by computational inefficiency, and its rigid balancing approach often impacts accuracy. In that vein, we introduce two novel algorithms.

The first, an enhancement of BNSLIM, incorporates the *Alternating Direction Method of Multipliers (ADMM)* to optimize all similarities concurrently, greatly reducing training time.

The second, *Fairly Sparse Linear Regression (FSLR)*, induces controlled sparsity in neighborhoods to reveal correlations among different demographic groups, achieving comparable efficiency while being more accurate.

Their performance is evaluated using standard exposure metrics alongside a new metric for user coverage disparities.

Our experiments cover various applications, including a novel exploration of bias in course recommendations by teachers' country development status.

Our results show the effectiveness of our algorithms in imposing fairness compared to BNSLIM and other well-known fairness approaches.

## Repository Content
This repository contains all the code used for our experiments. The `src` folder includes helper functions, covering data processing functions and metrics, and all the models used in our experiments. Additionally, the `notebooks` folder contains a notebook for each experiment, along with a notebook that summarizes the outcomes of all experiments.

Recommendation models:
- **EASE** (Steck, Harald, WWW 2019)
- **SLIM ADMM** (Steck, Harald, et al., WSDM 2020)
- **BNSLIM** (Burke, Robin, Sonboli, Nasim, Ordonez-Gauger, Aldo, FAccT 2018)
- **MF with Non-Parity Regularizer** (Yao, Sirui, Huang, Bert, NeurIPS 2017)
- **FDA BPR** (Chen, Lei, et al., WWW 2023)
- **FSLR** (Eleftherakis, Stavroula, Koutrika, Georgia, Amer-Yahia, Sihem, UMAP 2024)
- **BNSLIM ADMM** (Eleftherakis, Stavroula, Koutrika, Georgia, Amer-Yahia, Sihem, UMAP 2024)

Accuracy metrics:
- **Recall** (Liang, Dawen, et al., WWW 2018)
- **NDCG** (Liang, Dawen, et al., WWW 2018)

Coverage metrics:
- **Coverage** (Herlocker, Jonathan L., et al., ACM TOIS 2004)
- **APCR** (Liu, Weiwen, Burke, Robin, FATREC Workshop on Responsible Recommendation 2018)
- **u-Parity** (Eleftherakis, Stavroula, Koutrika, Georgia, Amer-Yahia, Sihem, UMAP 2024)

Exposure metrics:
- **c-Equity** (Burke, Robin, Sonboli, Nasim, Ordonez-Gauger, Aldo, FAccT 2018)
- **p-Equity** (Burke, Robin, Sonboli, Nasim, Ordonez-Gauger, Aldo, FAccT 2018)
- **REO** (Zhu, Ziwei, Wang, Jianling, Caverlee, James, SIGIR 2020)
- **RSP** (Zhu, Ziwei, Wang, Jianling, Caverlee, James, SIGIR 2020)
- **DP** (Chen, Lei, et al., WWW 2023)
- **BDV** (Eleftherakis, Stavroula, Koutrika, Georgia, Amer-Yahia, Sihem, UMAP 2024)

## Datasets
For access to the following datasets, please refer to their respective links:
- [Movielens 1M](https://grouplens.org/datasets/movielens/)
- [Lastfm 1K](http://ocelma.net/MusicRecommendationDataset/) (for the music genre of the artists, refer to [this](https://www.sciencedirect.com/science/article/abs/pii/S0306457322003090) paper)
- [COCO](https://link.springer.com/chapter/10.1007/978-3-319-77712-2_133)
- [Goodreads - Young Adult](https://mengtingwan.github.io/data/goodreads.html#news)

## Installation

To set up the project environment and install the necessary dependencies, run the following command:

```bash
poetry install --no-root

## Acknowledgements
If you utilize any part of this code for your research, please consider giving a star to this repository and citing our work:

Stavroula Eleftherakis, Georgia Koutrika, and Sihem Amer-Yahia. 2024. *Optimizing Neighborhoods for Fair Top-N Recommendation*. In UMAP ’24: Proceedings of the 32nd International Conference on User Modeling, Adaptation, and Personalization, July 1–4, 2024, Cagliari, Sardinia, Italy.

## Contact Information
For any questions or feedback, please contact me at seleftheraki [at] athenarc [dot] gr.
