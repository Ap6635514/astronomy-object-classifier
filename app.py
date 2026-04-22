import gradio as gr
import pandas as pd
import joblib

# ✅ Load model ONCE
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ✅ Prediction
def predict(u, g, r, i, z, redshift):
    data = pd.DataFrame({
        "u":[u], "g":[g], "r":[r],
        "i":[i], "z":[z], "redshift":[redshift]
    })

    data["g_r"] = data["g"] - data["r"]
    data["u_g"] = data["u"] - data["g"]
    data["r_i"] = data["r"] - data["i"]
    data["i_z"] = data["i"] - data["z"]
    data["u_r"] = data["u"] - data["r"]

    prediction = model.predict(data)[0]

    if prediction == "STAR":
        return "⭐ STAR"
    elif prediction == "GALAXY":
        return "🌌 GALAXY"
    else:
        return "💫 QSO"

# UI
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="u"),
        gr.Number(label="g"),
        gr.Number(label="r"),
        gr.Number(label="i"),
        gr.Number(label="z"),
        gr.Number(label="redshift")
    ],
    outputs="text",
    title="🌌 Astronomy Classifier"
)

demo.launch()