# Depression Dataset: Clustering Analysis and Bayesian Network

An unsupervised learning and probabilistic graphical modeling project analyzing a large depression-related dataset using **multiple clustering algorithms, dimensionality reduction, Bayesian Networks, and explainable AI (SHAP)**.


## Overview

This project investigates patterns and relationships within a large dataset containing socio-economic, lifestyle, and medical attributes of adults.

The analysis combines:

* Data preprocessing and exploratory analysis
* Mixed-type data handling
* Gower distance
* K-Means++
* K-Prototypes
* K-Modes
* Agglomerative Hierarchical Clustering
* DBSCAN
* Gaussian Mixture Models
* PCA and t-SNE
* Bayesian Network structure learning
* Bayesian inference
* Decision Tree surrogate modeling
* SHAP-based explainability

The primary target variable used for evaluation and Bayesian analysis is `History_of_Mental_Illness`, while `Family_History_of_Depression` is treated as a secondary target.

## Dataset

The project uses the **Depression Dataset** from Kaggle:

`anthonytherrien/depression-dataset`

The original dataset contains:

* **413,768 observations**
* **16 columns**

The variables include demographic, socioeconomic, lifestyle, and health-related attributes such as:

* Age
* Marital Status
* Education Level
* Number of Children
* Smoking Status
* Physical Activity Level
* Employment Status
* Income
* Alcohol Consumption
* Dietary Habits
* Sleep Patterns
* History of Mental Illness
* History of Substance Abuse
* Family History of Depression
* Chronic Medical Conditions

The `Name` identifier is removed before analysis.

## Data Preprocessing

Because the dataset contains a mixture of **numerical, ordinal, nominal, and binary variables**, different representations are created for different algorithms.

### Variable types

**Numerical**

* Age
* Income
* Number of Children

**Ordinal**

* Education Level
* Physical Activity Level
* Alcohol Consumption
* Dietary Habits
* Sleep Patterns

**Nominal**

* Marital Status
* Smoking Status
* Employment Status

**Binary**

* History of Substance Abuse
* Family History of Depression
* Chronic Medical Conditions
* History of Mental Illness

This variable-type classification is used as the basis for the subsequent preprocessing and clustering strategies.

### Missing values and duplicates

The dataset contains:

* **0 missing values**
* **0 fully duplicated rows**

Therefore, no missing-value imputation was required.

### Computational sampling

The original dataset is too large for some pairwise-distance and hierarchical-clustering operations.

A stratified sample of **10,000 observations** is therefore created while preserving the distribution of `History_of_Mental_Illness`.

After outlier filtering, **9,900 observations** remain.

## Feature Representations

Three parallel representations are constructed.

### 1. Numerical representation — `X_num`

Used for:

* K-Means++
* GMM
* PCA
* t-SNE
* Euclidean-based analysis

Ordinal variables are converted to ranks, binary variables to `0/1`, and nominal variables are one-hot encoded. Income is log-transformed and the resulting features are standardized.

### 2. Mixed representation — `X_mix`

Used for:

* K-Prototypes
* Gower-based clustering

Numerical variables are MinMax-scaled to `[0,1]`, while categorical variables remain categorical.

### 3. Categorical representation — `X_cat`

Used for:

* K-Modes
* Bayesian Network analysis

Numerical variables are discretized into quartile bins, while ordinal, nominal, and binary variables are represented categorically.

## Clustering Methods

The project compares several clustering approaches.

| Algorithm                  | Data representation | Selected clusters |
| -------------------------- | ------------------- | ----------------: |
| K-Means++                  | Numerical           |             **5** |
| K-Prototypes               | Mixed               |             **5** |
| K-Modes                    | Categorical         |             **6** |
| Agglomerative Hierarchical | Gower distance      |             **3** |
| DBSCAN                     | Numerical           |     Density-based |
| Gaussian Mixture Model     | Numerical           |      BIC-selected |

Cluster numbers are selected using internal validation measures such as the **Silhouette score, Calinski-Harabasz score, inertia/cost, and knee/elbow detection**.

### K-Means++

The elbow method selected **K = 5**.

The best silhouette score occurred at K = 10, but the elbow criterion was used for the final selection.

### K-Prototypes

K-Prototypes was selected with **K = 5**, allowing the analysis to jointly handle numerical and categorical variables.

The resulting cluster sizes were approximately:

* 2,115
* 1,480
* 2,219
* 1,984
* 2,102

### K-Modes

K-Modes was applied to the fully categorical representation and selected **K = 6** using the elbow criterion.

### Hierarchical Clustering

Agglomerative hierarchical clustering uses:

* Gower dissimilarity
* Complete linkage
* A further 2,000-point subset for dendrogram visualization

The selected number of clusters was **3**, with a silhouette score of approximately **0.057**.

## Dimensionality Reduction

The project uses:

* **PCA** for dimensionality reduction and exploratory analysis
* **t-SNE** for nonlinear two-dimensional visualization

t-SNE is applied to the standardized numerical representation with a fixed random seed for reproducibility.

## Clustering Evaluation

The clustering solutions are compared against the `History_of_Mental_Illness` target using:

* Purity
* Macro F1
* Accuracy
* Rand Index
* Fowlkes-Mallows Score
* Adjusted Rand Index (ARI)

For the primary-target comparison, the notebook reports approximately **69.6% accuracy/purity** for K-Means++, K-Prototypes, K-Modes, and Hierarchical clustering in one evaluation table.

A second evaluation is also performed on another subset, where the corresponding accuracy/purity is approximately **73.6%**.

These scores should be interpreted carefully because the clustering algorithms are unsupervised and the mental-health label is not used as a clustering feature.

## Bayesian Network Analysis

The project also learns **Discrete Bayesian Networks** using `pgmpy`.

The network structure is learned using **Hill-Climbing Search** with different scoring criteria.

### BIC

The BIC-based structure contains **25 edges**.

BIC applies a stronger complexity penalty and therefore produces a relatively sparse and interpretable network.

### AIC

The AIC-based structure contains **35 edges**, which is **10 more edges than the BIC model**.

### Expert knowledge

The analysis also introduces domain-inspired temporal constraints by dividing variables into:

1. Demographic/background variables
2. Lifestyle/behavior variables
3. Outcomes/current-health variables

Backward edges between these levels are forbidden, and Hill-Climbing is rerun using BIC. The resulting network contains **24 edges**.

One resulting relationship is:

`Income → History_of_Mental_Illness`

The learned network is used for probabilistic reasoning and interpretation rather than establishing causal relationships.

## Bayesian Network Classification

The learned Bayesian Network is also evaluated as a classifier for `History_of_Mental_Illness`.

On **600 held-out observations**, the notebook reports:

* **Accuracy: 0.688**
* **Macro F1: 0.408**

## Explainable AI

To investigate what drives the learned cluster assignments, a **Decision Tree surrogate model** is trained to predict the selected clustering labels.

**SHAP (SHapley Additive exPlanations)** is then used to analyze feature contributions and rank the variables influencing cluster assignment.

This provides an interpretable bridge between unsupervised cluster structure and the original dataset features.

## Technologies

The main technologies and libraries used include:

* Python
* NumPy
* Pandas
* Scikit-learn
* SciPy
* Matplotlib
* Seaborn
* `kmodes`
* Gower
* `kneed`
* `pgmpy`
* NetworkX
* SHAP
* Yellowbrick

The notebook explicitly uses `pgmpy` for Bayesian Networks, `kmodes` for K-Modes/K-Prototypes, Gower for mixed-type distance, `kneed` for knee detection, and SHAP for explainability.

## Project Structure

```text
.
├── clustering_and_bayesian_network_anaysis.ipynb
├── README.md
└── data/
    └── depression_data.csv
```

> The dataset may need to be downloaded separately from Kaggle because the original notebook uses a local dataset path.

## Reproducibility

A fixed random seed of **42** is used throughout the analysis where applicable.

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

This helps make sampling, clustering, and dimensionality-reduction results reproducible.

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install kmodes gower kneed pgmpy networkx shap yellowbrick
```

### 3. Download the dataset

Download `depression_data.csv` from the Kaggle Depression Dataset and place it in the expected project data directory.

### 4. Open the notebook

```bash
jupyter notebook clustering_and_bayesian_network_anaysis.ipynb
```

Run the notebook cells sequentially.

## Key Takeaways

This project demonstrates how different unsupervised-learning methods can be applied to a **large mixed-type dataset**, where conventional Euclidean distance is not always appropriate.

The main methodological ideas are:

* Match the distance measure to the variable types.
* Use Gower distance for mixed numerical/categorical data.
* Compare different clustering paradigms rather than relying on a single algorithm.
* Use internal and external evaluation measures to assess cluster quality.
* Use Bayesian Networks to model probabilistic dependencies between variables.
* Incorporate expert knowledge as structural constraints.
* Use SHAP to make learned cluster structures more interpretable.

## Disclaimer

This project is an **data-analysis exercise**. The identified relationships and predictions should not be interpreted as clinical diagnoses or evidence of causal relationships between the variables.

---
