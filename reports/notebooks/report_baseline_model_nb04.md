# Baseline Modeling – Completion Report

---

## Executive Summary

This notebook documents the successful completion of the **baseline modeling phase** for the Telco Customer Churn dataset.
Using the preprocessed dataset, we trained, evaluated, and persisted a **logistic regression model** as the initial benchmark.
This baseline serves as the reference point against which future improvements (feature engineering, hyperparameter tuning, multimodal integration) will be measured.

The model demonstrated strong discriminatory ability (AUC = 0.84), solid recall of churners, and reasonable precision, confirming the dataset’s predictive power and validating logistic regression as a sound starting point.

---

## Key Achievements

- **Baseline Model Established:** Logistic Regression chosen for interpretability and benchmarking.
- **Performance Evaluation:** Conducted using Accuracy, AUC, Confusion Matrix, Classification Report, Precision-Recall and ROC curves.
- **Model Persistence:** The trained model was saved as:

  ```bash
  logistic_regression_churn_v1.joblib
  ```
- **Separation of Concerns:** Data preprocessing and model training were handled by independent, reproducible modules, ensuring flexibility and modularity.


## Metrics & Business Interpretation
| Metric / Visualization     | Observed Result                                               | Business Interpretation                                                                                                                                                                                          |
| -------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Confusion Matrix**       | 748 TN, 293 TP, 287 FP, 81 FN                                 | The model detects churners reasonably well, but raises false alarms (\~287 FP). This implies additional retention efforts may be allocated to customers who would not churn.                                     |
| **Precision-Recall Curve** | Precision \~0.75 at medium recall, drops as recall increases. | The model balances reasonably well between precision and recall but struggles to maintain both at high levels. Useful for prioritizing high-probability churners, but not sufficient to capture all churn cases. |
| **ROC Curve (AUC = 0.84)** | Strong discriminatory power between churn vs. no churn.       | Indicates that the logistic regression baseline is a robust starting point. An AUC of 0.84 suggests the model already distinguishes customer behavior with high reliability.                                     |


---

## Insights & Rationale

``Strength in Recall:`` The model identifies most churners (recall = 0.78), which is crucial in customer retention contexts.

``Moderate Precision:`` Precision remains at 0.51, meaning some retention efforts could be misallocated.

``Imbalance Awareness:`` Performance is stronger on the majority class (No Churn), underlining the need for strategies to better capture minority churn cases.

``Interpretability:`` Logistic regression coefficients allow for transparent, business-friendly insights into drivers of churn.


---

## Conclusion

The baseline modeling phase has been successfully executed:

* ``Reliable Benchmark:`` Established logistic regression as a reproducible reference model.

* ``Predictive Signal Confirmed:`` AUC of 0.84 validates the dataset’s strength in separating churners from non-churners.

* ``Actionable Insights:`` The model’s recall-heavy profile indicates it is well-suited to detecting at-risk customers but requires refinement to reduce false positives.

* ``Persistence & Reproducibility:`` Model artifacts have been saved in portable format (joblib), ensuring easy comparison with future iterations.


---

## Next Steps

* Feature Engineering to enhance predictive power beyond the baseline.

* Hyperparameter Tuning to optimize model precision-recall trade-offs.

* Multimodal Integration: Incorporate Reddit comments to enrich churn prediction with textual sentiment and behavioral cues.

* Pipeline Consolidation: Consider merging preprocessing and modeling pipelines for production readiness.