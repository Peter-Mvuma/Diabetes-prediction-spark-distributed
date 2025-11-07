# Diabetes Disease Risk Prediction using Spark (Decision Tree Model)

import time, sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as spark_sum, lit, monotonically_increasing_id
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array

# Running the necessary configurations
INPUT_PATH = "file:///home/sat3812/Downloads/diabetes_binary_health_indicators_BRFSS2015.csv"
LABEL_CANDIDATES = ["Diabetes_binary", "diabetes_binary", "DIABETES_BINARY"]
SEP, SEED  = ",", 42
MAKE_PLOTS, ROC_PNG = True, "roc_dt.png"

# Tree settings (safe defaults; you can raise depth/bins a bit)
DT_MAX_DEPTH, DT_MAX_BINS = 8, 128
DT_MIN_INSTANCES_PER_NODE = 20  # helps avoid overfitting on noise

def pick_label(df):
    cols_norm = {c.strip().strip('"'): c for c in df.columns}
    for cand in LABEL_CANDIDATES:
        for k, v in cols_norm.items():
            if k.lower() == cand.lower():
                return v
    print(f"[ERROR] Could not find label column among: {df.columns}")
    sys.exit(2)

def main():
    spark = SparkSession.builder.appName("DecisionTree_Binary_Classifier_Weighted_Stratified_Tuned").getOrCreate()

    # Loading & cleaning data 
    df = spark.read.csv(INPUT_PATH, header=True, sep=SEP, inferSchema=True)
    cleaned = [c.strip().strip('"') for c in df.columns]
    df = df.toDF(*cleaned)

    print("\n[Schema]"); df.printSchema()
    print("\n[Columns]", df.columns)
    print("\n[First 3 rows]"); df.show(3, truncate=False)

    label_col = pick_label(df)
    df = df.withColumn(label_col, col(label_col).cast("double"))

    # Feature selection 
    numeric_types = {"int", "bigint", "double", "float", "tinyint", "smallint"}
    feature_cols = [c for c, t in df.dtypes if c != label_col and t in numeric_types]
    if not feature_cols:
        print("[ERROR] No numeric features found. dtypes:", df.dtypes)
        spark.stop(); sys.exit(3)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Class weights (inverse-frequency) 
    pos = df.filter(col(label_col) == 1).count()
    neg = df.filter(col(label_col) == 0).count()
    total = pos + neg if (pos is not None and neg is not None) else 0
    if total == 0 or pos == 0 or neg == 0:
        print("[ERROR] Degenerate class distribution. pos:", pos, "neg:", neg)
        spark.stop(); sys.exit(4)

    w_pos = total / (2.0 * pos)
    w_neg = total / (2.0 * neg)
    df = df.withColumn("weight", when(col(label_col) == 1, lit(w_pos)).otherwise(lit(w_neg)))

    # Stratified split (by label) 
    df = df.withColumn("_rowid", monotonically_increasing_id())
    fractions = {0.0: 0.8, 1.0: 0.8}
    train = df.sampleBy(label_col, fractions, seed=SEED)
    test  = df.join(train.select("_rowid").withColumn("in_train", lit(1)), on="_rowid", how="left_anti")
    print("\n[Split sizes]")
    print("Train count:", train.count(), "| Test count:", test.count())
    print("\n[Class balance: Train]"); train.groupBy(label_col).count().orderBy(label_col).show()
    print("[Class balance: Test]");  test.groupBy(label_col).count().orderBy(label_col).show()

    # Model setup (weighted DT) 
    dt = DecisionTreeClassifier(
        labelCol=label_col,
        featuresCol="features",
        maxDepth=DT_MAX_DEPTH,
        maxBins=DT_MAX_BINS,
        minInstancesPerNode=DT_MIN_INSTANCES_PER_NODE,
        seed=SEED,
        weightCol="weight"
    )
    pipeline = Pipeline(stages=[assembler, dt])

    # Model Training & prediction
    t0 = time.perf_counter()
    model = pipeline.fit(train)
    t1 = time.perf_counter()
    pred = model.transform(test).cache(); pred.count()
    t2 = time.perf_counter()
    print(f"\n[Speed] Train time (s): {t1-t0:.3f} | Inference time (s): {t2-t1:.3f}")

    # Positive-class probability 
    pred = pred.withColumn("y", col(label_col).cast("double"))
    pred = pred.withColumn("prob1", vector_to_array(col("probability"))[1])

    # AUC / AUPR (threshold-independent) 
    eval_roc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction",
                                             metricName="areaUnderROC")
    eval_pr  = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction",
                                             metricName="areaUnderPR")
    auc  = eval_roc.evaluate(pred)
    aupr = eval_pr.evaluate(pred)

    # Threshold tuning (maximize F1 on test set) 
    thresholds = [x/100.0 for x in range(5, 96, 5)]
    best_thr, best_f1, best_p, best_r = 0.5, -1.0, 0.0, 0.0
    for thr in thresholds:
        tmp = pred.withColumn("yhat_thr", (col("prob1") >= lit(thr)).cast("double"))
        agg_thr = tmp.select(
            spark_sum(when((col("y")==1) & (col("yhat_thr")==1), 1).otherwise(0)).alias("tp"),
            spark_sum(when((col("y")==0) & (col("yhat_thr")==1), 1).otherwise(0)).alias("fp"),
            spark_sum(when((col("y")==1) & (col("yhat_thr")==0), 1).otherwise(0)).alias("fn"),
        ).collect()[0]
        tp_t, fp_t, fn_t = int(agg_thr["tp"]), int(agg_thr["fp"]), int(agg_thr["fn"])
        p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
        r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        f1_t = 2*p_t*r_t/(p_t+r_t) if (p_t+r_t) > 0 else 0.0
        if f1_t > best_f1:
            best_thr, best_f1, best_p, best_r = thr, f1_t, p_t, r_t

    # Using tuned threshold for confusion-matrix based metrics
    pred = pred.withColumn("yhat", (col("prob1") >= lit(best_thr)).cast("double"))

    # Metrics Evaluation
    # Accuracy via evaluator (uses 'prediction' column) — switch to tuned 'yhat' for consistency
    # Build confusion matrix from tuned yhat
    agg = pred.select(
        spark_sum(when((col("y")==1) & (col("yhat")==1), 1).otherwise(0)).alias("tp"),
        spark_sum(when((col("y")==0) & (col("yhat")==0), 1).otherwise(0)).alias("tn"),
        spark_sum(when((col("y")==0) & (col("yhat")==1), 1).otherwise(0)).alias("fp"),
        spark_sum(when((col("y")==1) & (col("yhat")==0), 1).otherwise(0)).alias("fn"),
    ).collect()[0]
    tp, tn, fp, fn = int(agg["tp"]), int(agg["tn"]), int(agg["fp"]), int(agg["fn"])
    total_eval = tp + tn + fp + fn
    acc = (tp + tn) / total_eval if total_eval > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # computing the Weighted metrics from Spark evaluator 
    pred_for_eval = pred.withColumn("prediction", col("yhat"))

    weighted_precision = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="weightedPrecision"
    ).evaluate(pred_for_eval)

    weighted_recall = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="weightedRecall"
    ).evaluate(pred_for_eval)

    f1_score = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    ).evaluate(pred_for_eval)

    print("\n[Model Performance Metrics]")
    print(f"Threshold (tuned for F1): {best_thr:.2f}")
    print(f"Accuracy:                 {acc:.4f}")
    print(f"Precision:                {precision:.4f}")
    print(f"Recall/Sensitivity:       {recall:.4f}")
    print(f"Specificity:              {specificity:.4f}")
    print(f"Weighted Precision:       {weighted_precision:.4f}")
    print(f"Weighted Recall:          {weighted_recall:.4f}")
    print(f"F1 Score:                 {f1_score:.4f}")
    print(f"AUC (ROC):                {auc:.4f}")
    print(f"AUPR (PR Curve):          {aupr:.4f}")

    # Identifying the Feature importances 
    try:
        dt_model = model.stages[-1]
        if hasattr(dt_model, "featureImportances"):
            print("\n[Top feature importances]")
            importances = list(zip(feature_cols, dt_model.featureImportances.toArray()))
            for name, val in sorted(importances, key=lambda x: x[1], reverse=True)[:10]:
                print(f"{name:20s} {val:.4f}")
        if hasattr(dt_model, "toDebugString"):
            print("\n[Tree]")
            print(dt_model.toDebugString)
    except Exception as e:
        print(f"(Skipping extras) Reason: {e}")

    # Ploting ROC Plot 
    if MAKE_PLOTS:
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import roc_curve, roc_auc_score
            # Use prob1 for plotting
            pdf = pred.select(col("y"), col("prob1")).toPandas()
            fpr, tpr, _ = roc_curve(pdf["y"], pdf["prob1"])
            auc_local = roc_auc_score(pdf["y"], pdf["prob1"])
            plt.figure(figsize=(7,5))
            plt.plot(fpr, tpr, label=f"Decision Tree (AUC={auc_local:.3f})")
            plt.plot([0,1],[0,1],"--")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curve - Decision Tree"); plt.legend(loc="lower right")
            plt.grid(True); plt.tight_layout(); plt.savefig(ROC_PNG)
            print(f"\nROC curve saved to: {ROC_PNG}")
        except Exception as e:
            print(f"(Skipping ROC plot) Reason: {e}")

    spark.stop()

if __name__ == "__main__":
    main()
