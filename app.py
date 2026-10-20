
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Starbucks Decision Tree",
    page_icon="☕",
    layout="wide"
)

st.title("Starbucks Decision Tree Prediction")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "starbucks_decision_tree_model.joblib"
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file nahi mili: " + MODEL_PATH)
    st.info(
        "Pehle apna Decision Tree train karein "
        "aur model ko joblib file mein save karein."
    )
    st.stop()

try:
    loaded_object = joblib.load(MODEL_PATH)

    # Support a saved model or a dictionary containing a model
    if isinstance(loaded_object, dict):
        model = loaded_object.get("model")
        feature_columns = loaded_object.get("feature_columns", [])
    else:
        model = loaded_object
        feature_columns = list(
            getattr(model, "feature_names_in_", [])
        )

    if model is None or not hasattr(model, "predict"):
        st.error(
            "Saved file mein valid trained model nahi hai. "
            "Model ko training code se dobara save karein."
        )
        st.stop()

    st.success("Model successfully loaded!")

except Exception as e:
    st.error(f"Model load nahi ho saka: {e}")
    st.warning(
        "Agar error mein 'No module named mport...' aaye, "
        "to saved joblib file ko original training code se "
        "dobara banana hoga."
    )
    st.stop()

st.subheader("Enter model input")

if not feature_columns:
    st.warning(
        "Model ke feature names available nahi hain. "
        "Original training code se feature columns save karein."
    )
    st.stop()

with st.form("prediction_form"):
    values = {}

    for col in feature_columns:
        values[col] = st.number_input(
            str(col),
            value=0.0
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_df = pd.DataFrame(
            [values],
            columns=feature_columns
        )

        prediction = model.predict(input_df)

        st.subheader("Prediction Result")
        st.write("Predicted class:", prediction[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            st.write("Prediction probabilities:")

            st.dataframe(
                pd.DataFrame({
                    "Class": model.classes_,
                    "Probability": probabilities
                }),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")
