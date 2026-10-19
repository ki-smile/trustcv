---
description: The fifteen Jupyter notebooks in the trustcv repository, what each covers, and how to run them locally or in Google Colab.
---

# Notebooks

The repository has fifteen Jupyter notebooks in [`notebooks/`](https://github.com/ki-smile/trustcv/tree/main/notebooks). They range from a tour of the independent-sample splitters to full studies on public clinical data. Most use scikit-learn's breast cancer dataset or synthetic data; notebooks 09 and 12 download real public datasets.

!!! warning "In Colab, install trustcv first"
    Colab does not come with trustcv. Before running a notebook, add a cell at the top with

    ```bash
    !pip install trustcv
    ```

    and run it. Several notebooks have no install cell of their own (the "Installs trustcv" column below), and they fail at the first `import trustcv` without it. Notebooks that need extra libraries say so near the top.

Some notebooks were written for earlier releases (from 1.0.5 to 1.1.0) and their text mentions those versions. If a cell fails with trustcv 1.2.1, please [open an issue](https://github.com/ki-smile/trustcv/issues).

## All notebooks

| # | Notebook | What it covers | Data | Installs trustcv |
| --- | --- | --- | --- | --- |
| 01 | [IID methods showcase](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/01_IID_Methods_Showcase.ipynb) | Each independent-sample splitter on its own: hold-out, K-fold, stratified, repeated, LOOCV, LPOCV, Monte Carlo, bootstrap and nested CV | Breast cancer | Yes |
| 02 | [Advanced workflow: UniversalCVRunner](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/02_Advanced_Workflow_UniversalRunner.ipynb) | The lower-level pieces: `UniversalCVRunner`, `ClinicalMetrics` and reporting | Breast cancer | From GitHub, if missing |
| 03 | [TrustCVValidator showcase](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/03_TrustCVValidator_Showcase.ipynb) | Hold-out across random seeds, stratified against non-stratified; scores, intervals and checks | Breast cancer | No |
| 04 | [TrustCVValidator: IID comparison](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/04_TrustCVValidator_IID_Comparison.ipynb) | Every independent-sample method through `TrustCVValidator`, with summaries side by side | Breast cancer | No |
| 05 | [Cross-validation comparison](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/05_CrossValidation_Comparison.ipynb) | trustcv next to plain scikit-learn and other libraries (XGBoost, LightGBM, H2O, PyCaret, Keras, PyTorch) | Breast cancer, small synthetic set | No; also needs the other libraries |
| 06 | [Deep learning showcase with glossary](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/06_TrustCV_DeepLearning_Showcase_with_Glossary.ipynb) | Keras models through `UniversalCVRunner` and `KerasSkWrap`, with a glossary of terms. TensorFlow only | MNIST | No (the install line is commented out); needs TensorFlow |
| 07 | [Grouped CV for medical AI](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/07_trustcv_grouped_cv_medical.ipynb) | Every grouped splitter on a multi-site clinical dataset, with leakage checks | Synthetic multi-site | Yes, from GitHub |
| 08 | [Structured scenarios: temporal, spatial, hierarchical](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/08_TrustCV_Structured_Scenarios_S3.ipynb) | How the right split changes when data are time-ordered, spatially correlated or nested | Synthetic | From GitHub, if missing |
| 09 | [UCI HAR: grouped leakage and nested CV](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/09_RealData_UCI_HAR_Grouped_Leakage_NestedCV.ipynb) | Sitting against standing: record-wise K-fold compared with subject-aware and nested validation | UCI HAR (downloaded) | Optional setting in the first cell |
| 10 | [Advanced workflow: clinical metrics](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/10_Advanced_Workflow_oob_clinical_metrics.ipynb) | A custom run with `UniversalCVRunner`, clinical metrics (likelihood ratios, NNT) and an HTML report | Wine, breast cancer | No |
| 11 | [Spatial methods with split_kwargs](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/11_Spatial_Synthetic_EnvironmentalHealth_TrustCV_SplitKwargs.ipynb) | The four spatial splitters through `UniversalCVRunner.run(split_kwargs=...)` | Synthetic environmental health | Optional setting in the first cell |
| 12 | [PhysioNet 2019 sepsis](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/12_PhysioNet2019_Sepsis_RealClinicalData_TrustCV_cloud_TEMPORAL.ipynb) | Grouped, temporal and cross-hospital validation on real ICU data from two hospital systems. Long-running; start with its smoke-test mode | PhysioNet Challenge 2019 (downloaded) | Optional setting in the first cell |
| 13 | [Leakage checker: injection tests](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/13_LeakageChecker_Injection_Boundary_Validation.ipynb) | Controlled injections of patient overlap, duplicates and temporal violations, to test where the checker's rules fire | Synthetic | Optional setting in the first cell |
| 14 | [LOGO metric feasibility warning](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/14_LOGO_metric_feasibility_warning.ipynb) | Why fold-wise AUC can be undefined in leave-one-group-out, and pooled out-of-fold metrics instead | Synthetic, six patients | No |
| 15 | [Trust claims](https://colab.research.google.com/github/ki-smile/trustcv/blob/main/notebooks/15_Trust_Claims_v1.1.ipynb) | Feature selection before CV reported honestly, the Pipeline fix, and `recommend_cv` | Breast cancer | No |

"Optional setting" means the notebook has an install switch in its first cells; set it to install from PyPI or GitHub, or run `!pip install trustcv` yourself.

## A suggested order

1. Start with 01 and 15: the splitters, and why honest checks matter.
2. Then 03 and 04 for the high-level validator, 07 for grouped data and 08 for temporal, spatial and hierarchical data.
3. For real data, 09 (wearable sensors) and 12 (ICU).
4. 02, 10, 11 and 06 cover the lower-level runner and deep learning; 05, 13 and 14 go into specific questions.

## Run them locally

```bash
git clone https://github.com/ki-smile/trustcv.git
cd trustcv
pip install -e ".[all]"
pip install jupyterlab
jupyter lab notebooks/
```

Notebooks run from inside the repository pick up the checked-out source. See [Install](../get-started/install.md) for the optional extras.
