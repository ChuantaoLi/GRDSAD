# Dataset

The experiment is structured into two phases: the first phase employs cardiovascular disease datasets for validation, the second phase utilizes datasets from the KEEL database. 

The cardiovascular disease datasets comprise four distinct sources: Survival7Y and Survival10Y are real-world longitudinal follow-up datasets for ischemic heart disease, provided by the CNR Institute of Clinical Physiology in Italy, sourced from: https://github.com/orientino/ml4cad. 

The SPECTF Heart dataset is obtained from the UCI Machine Learning Repository, accessible at https://archive.ics.uci.edu/dataset/96/spectf+heart. 

The Framingham dataset is derived from Kaggle, available at https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset. 

From the KEEL database, eight binary classification datasets were selected, accessible at https://sci2s.ugr.es/keel/imbalanced.php?order=name#sub10. 

An overview of these datasets is presented in the subsequent table.

|   Dataset    | Instance size | Feature size | Imbalance ratio |
| :----------: | :-----------: | :----------: | :-------------: |
|  Survival7Y  |     3987      |      18      |      5.34       |
| Survival10Y  |     3987      |      18      |      4.51       |
|    SPECTF    |      267      |      22      |      3.85       |
|  Framingham  |     4240      |      15      |      5.58       |
| ecoli-0_vs_1 |      220      |      7       |      1.86       |
|    ecoli1    |      336      |      7       |      3.36       |
|    ecoli2    |      336      |      7       |      5.46       |
|    ecoli3    |      336      |      7       |       8.6       |
|   haberman   |      306      |      3       |      2.78       |
| newthyroid1  |      215      |      5       |      5.14       |
| newthyroid2  |      215      |      5       |      5.14       |
|    yeast3    |     1484      |      8       |       8.1       |

# Code

The DSAD.py script implements GRDSAD without graph regularization. 

The Model_test.py script conducts the comparative analysis, evaluating seven machine learning models.

The AdaBoost.py script reproduces the traditional AdaBoost implementation for baseline experimental comparison.

The GRDSAD.py script contains the proposed methodology outlined in this study. 

The Parameter_analysis.py script executes the parameter sensitivity analysis.

# Parameterization

The experiments in the parameter sensitivity analysis section address four parameters: $\eta$, $\rho$, $\gamma$ and $d'$.

$\eta$ denotes the degree of smoothing of the sigmoid function in the instance weight update.

$\rho$ denotes the bandwidth of the heat kernel used to construct the similarity graph.

$\gamma$ denotes the degree of regularization of the graph.

$d'$ denotes the $d'$ percent of components retained by the dimensionality reduction. 

The parameter settings for the 12 datasets used in the experiment are shown in the table below.

|   Dataset    | $\eta$ | $\rho$ | $\gamma$ | $d'$ |
| :----------: | :----: | :----: | :------: | :--: |
|  Survival7Y  |  0.1   |   1    |   0.01   | 30%  |
| Survival10Y  |  0.1   |   1    |   1.5    | 30%  |
|    SPECTF    |  0.1   |   1    |   0.01   | 40%  |
|  Framingham  |  0.4   |  1.1   |    3     | 50%  |
| ecoli-0_vs_1 |  0.1   |   1    |   0.01   | 80%  |
|    ecoli1    |  0.1   |   1    |   0.01   | 80%  |
|    ecoli2    |  0.5   |  1.2   |   0.1    | 50%  |
|    ecoli3    |  0.1   |   1    |   0.1    | 60%  |
|   haberman   |  0.4   |  1.4   |   0.1    | 40%  |
| newthyroid1  |  0.1   |   1    |   0.01   | 80%  |
| newthyroid2  |  0.1   |   1    |   0.05   | 80%  |
|    yeast3    |  0.5   |   2    |   0.5    | 30%  |

# Results

Within the Results directory, 7Ydata.xlsx, 10Ydata.xlsx, SPECTF.xlsx, and Framingham.xlsx contain the outcomes of Experiment 1, encompassing both ablation and comparative results. 

KEEL.xlsx documents the experimental findings for the eight test datasets, including performance comparisons against seven state-of-the-art Boosting variants: AdaM1, AdaM2, AdaNC9, LexiBoostM1, LexiBoostM2, DualLexiBoost, and JanEnsemble. The origins of these models are detailed in the accompanying table. 

Additionally, Heart parameter.xlsx and KEEL parameter.xlsx archive the results of the parameter sensitivity analysis.

|                 Models                  |                            Papers                            |
| :-------------------------------------: | :----------------------------------------------------------: |
|              AdaM1, AdaM2               | Y. Freund and R. E. Schapire, “Experiments with a new boosting  algorithm,” in Proc. Int. Conf. Mach. Learn., 1996, pp. 148–156. |
|                 AdaNC9                  | S. Wang and X. Yao, “Multiclass imbalance problems: Analysis and  potential solutions,” IEEE Trans. Syst., Man, Cybern., Part B, vol. 42, no.  4, pp. 1119–1130, Aug. 2012. |
| LexiBoostM1, LexiBoostM2, DualLexiBoost | S. Datta, S. Nag, and S. Das, “Boosting with lexicographic  programming: Addressing class imbalance without cost tuning,” IEEE Trans.  Knowl. Data Eng., vol. 32, no. 5, pp. 883–897, May 2020. |
|               JanEnsemble               | Z. Jan, J. C. Munos, and A. Ali, “A novel method for creating an  optimized ensemble classifier by introducing cluster size reduction and  diversity,” IEEE Trans. Knowl. Data Eng., vol. 34, no. 7, pp. 3072–3081, Jul.  2022. |

