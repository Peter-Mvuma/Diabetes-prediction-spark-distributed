# Diabetes Prediction Using Distributed Computing with Apache Spark
This project demonstrates how Apache Spark can be used in a distributed computing environment to train a Decision Tree model for predicting diabetes risk from population health data.  
Developed as part of the SAT 5165 – Introduction to Big Data Analytics course at Michigan Technological University.

## Overview
- **Goal:** Predict diabetes using large-scale health indicators through a distributed Decision Tree classifier.  
- **Framework:** Apache Spark running on a two-node Hadoop cluster (master–worker setup).  
- **Dataset:** [Behavioral Risk Factor Surveillance System (BRFSS 2015)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) — 253,680 survey records with 21 health-related variables.
  
- **Focus:** Demonstrate scalability, interpretability, and computational efficiency through distributed processing.
  
- **Programming Environment:** PySpark (Spark MLlib), Hadoop, and Fedora-based Virtual Machines.

## Implementation Steps

### 1. Cluster Configuration
- Configured HDFS and Spark across two virtual machines (`hadoop1` and `hadoop2`).
- Verified daemons using `jps` and tested inter-node communication via `ssh`.
- Confirmed active Spark Master and Worker nodes through the Spark Web UI (`spark://hadoop1:8080`).

### 2. Data Preparation
- Loaded and cleaned the BRFSS dataset in Spark with automatic schema inference.  
- Selected relevant numeric features and combined them using `VectorAssembler`.  
- Applied inverse-frequency class weights to manage imbalance (86% non-diabetic vs 14% diabetic).  
- Used stratified train-test split (80/20) to preserve class ratios.

### 3. Model Training
- Implemented a Decision Tree Classifier with parameters:  
  `maxDepth=8`, `maxBins=128`, `minInstancesPerNode=20`, `seed=42`.  
- Included class weights for improved sensitivity to minority (diabetic) cases.  
- Distributed data and computations across worker nodes using Spark MLlib.

### 4. Model Evaluation
| **Metric**       | **Value**| **Interpretation** |
| Accuracy           | 0.78 |   78% of instances correctly classified |
| AUC (ROC)          | 0.69 |   Good separation between classes |
| Weighted Precision | 0.85 |   85% of predicted diabetic cases were true positives |
| Weighted Recall    | 0.78 |   78% of actual diabetic cases were detected |
| Specificity        | 0.80 |   80% of non-diabetic cases correctly identified |
| F1 Score           | 0.80 |   Balanced precision–recall performance |

## Performance Comparison
| **Configuration**      | **Runtime** | **Speed Improvement** |
| Single VM (Hadoop1)    | ~18 minutes | 
| Two VMs (Distributed)  | ~2.6 minutes| **6.9× faster (≈86% reduction)** |

> The distributed Spark cluster demonstrated substantial performance gains due to task parallelization and workload distribution across nodes.

## Key Findings
- **Top Predictors/ Feature Importance**  
  - High Blood Pressure (0.4612)  
  - General Health (0.2877)  
  - Body Mass Index (0.1035)

These features are consistent with established clinical risk factors for diabetes, reinforcing the interpretability and practical relevance of the model.

## Lessons Learned
- Distributed computing significantly reduces processing time and improves scalability for large datasets.  
- The Decision Tree model provides interpretable insights but showed limited generalization (AUC ≈ 0.69).  
- System stability can be affected by hardware constraints—resource allocation and memory tuning are crucial.

## Repository Contents
| File | Description |
| `dt_classifier_report.py` | PySpark code for Decision Tree training and evaluation |
| `Small_Project3_Report.pdf` | Final project report with results and discussion |
| `roc_dt.png` | ROC curve visualization |
| `diabetes_binary_health_indicators_BRFSS2015.csv` | Dataset|


## 🧑‍💻 Author

**Peter Mvuma**  
MSc Health Informatics | Michigan Technological University 
📅 November 2025
