import joblib

# dt_model ko apne trained model ke variable name se replace karein
joblib.dump(
    dt_model,
    "starbucks_decision_tree_model.joblib"
)

print("Model successfully saved!")
