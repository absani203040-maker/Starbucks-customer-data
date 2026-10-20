
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Starbucks Decision Tree Predictor",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Starbucks Customer Offer Prediction")
st.write(
    "Decision Tree Machine Learning Model "
    "for Starbucks customer offer predictions."
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
MODEL_PATH = "starbucks_decision_tree_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' not found. "
        "Please place it in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.success("Decision Tree model loaded successfully!")

# ==========================================
# GET EXPECTED FEATURES
# ==========================================
def get_expected_features(model):
    # Works with a scikit-learn Pipeline
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


expected_features = get_expected_features(model)

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose a page:",
    ["Home", "Single Prediction", "Batch Prediction"]
)

# ==========================================
# HOME PAGE
# ==========================================
if page == "Home":

    st.header("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Machine Learning Model")
        st.write("Algorithm: Decision Tree Classifier")
        st.write("Dataset: Starbucks customer/offer data")

    with col2:
        st.subheader("Application Features")
        st.write("- Individual customer predictions")
        st.write("- CSV file batch predictions")
        st.write("- Download prediction results")

    st.info(
        "Use the Single Prediction page to enter customer "
        "features, or Batch Prediction to upload a CSV file."
    )

# ==========================================
# SINGLE PREDICTION
# ==========================================
elif page == "Single Prediction":

    st.header("Predict Customer Offer Outcome")

    if not expected_features:
        st.error(
            "The model does not expose its input feature names. "
            "Please check the saved model and training code."
        )
        st.stop()

    st.write("Enter the feature values expected by the trained model.")

    with st.form("prediction_form"):

        input_values = {}

        for feature in expected_features:
            input_values[feature] = st.text_input(
                label=str(feature),
                value="0",
                key=f"input_{feature}"
            )

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            row = {}

            for feature in expected_features:
                value = input_values[feature].strip()

                try:
                    row[feature] = float(value)
                except ValueError:
                    row[feature] = value

            input_df = pd.DataFrame(
                [row],
                columns=expected_features
            )

            # Restore numeric types where possible
            for column in input_df.columns:
                converted = pd.to_numeric(
                    input_df[column], errors="coerce"
                )
                if converted.notna().all():
                    input_df[column] = converted

            prediction = model.predict(input_df)[0]

            st.subheader("Prediction Result")
            st.write(f"**Predicted class:** {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classes = model.classes_

                st.write("### Prediction Probabilities")

                probability_df = pd.DataFrame({
                    "Class": classes,
                    "Probability": probabilities
                })

                st.dataframe(
                    probability_df,
                    use_container_width=True
                )

        except Exception as e:
            st.error(
                f"Prediction failed: {e}. "
                "Make sure the input values match the training data."
            )

# ==========================================
# BATCH PREDICTION
# ==========================================
elif page == "Batch Prediction":

    st.header("Upload Customer Data CSV")

    st.write(
        "Upload a CSV containing the same input feature columns "
        "used when training the saved model."
    )

    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data Preview")
            st.dataframe(data.head(), use_container_width=True)

            if st.button("Generate Predictions"):

                if expected_features is not None:
                    missing = [
                        col for col in expected_features
                        if col not in data.columns
                    ]

                    if missing:
                        st.error(
                            "Missing required columns: "
                            + ", ".join(missing)
                        )
                        st.stop()

                    model_input = data[expected_features].copy()

                else:
                    model_input = data.copy()

                predictions = model.predict(model_input)

                results = data.copy()
                results["Predicted_Offer_Outcome"] = predictions

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(model_input)
                    results["Prediction_Confidence"] = (
                        np.max(probabilities, axis=1) * 100
                    ).round(2)

                st.subheader("Prediction Results")
                st.dataframe(results, use_container_width=True)

                csv_data = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_data,
                    file_name="starbucks_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("Starbucks Customer Prediction | Decision Tree ML")

