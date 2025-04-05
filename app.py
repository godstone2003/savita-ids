from flask import Flask, render_template, request
import torch
import torch.nn as nn
import numpy as np
import joblib

app = Flask(__name__)

# === Device === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Label Mapping === #
label_mapping = {
    0: "Dos",
    1: "Theft",
    2: "DDos",
    3: "Normal",
    4: "Reconnaissance"
}

# === Required Features === #
required_features = [
    'pkts', 'bytes', 'sbytes', 'dbytes', 'rate',
    'mean', 'stddev', 'sum', 'proto', 'state',
    'dur'
]

# === Define Model === #
class SpinalSAENet(nn.Module):
    def __init__(self, input_dim=11, hidden_dims=[128, 64], num_classes=5):
        super(SpinalSAENet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[1], 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


# === Load Model & Scaler === #
model = SpinalSAENet(input_dim=11, hidden_dims=[128, 64], num_classes=5).to(device)
model.load_state_dict(torch.load("spinal_saenet_best.pth", map_location=device))
model.eval()

scaler = joblib.load("scaler.pkl")

# === Flask Routes === #
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get input values
            input_data = [float(request.form[feat]) for feat in required_features]
            input_array = np.array([input_data], dtype=np.float32)
            input_scaled = scaler.transform(input_array)
            X_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

            with torch.no_grad():
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
                predicted_class = int(np.argmax(probs))
                confidence = probs[predicted_class]

            result = {
                "label": label_mapping[predicted_class],
                "confidence": f"{confidence:.4f}",
                "probabilities": {label_mapping[i]: f"{prob:.4f}" for i, prob in enumerate(probs)}
            }

            return render_template("result.html", result=result)

        except Exception as e:
            return f"Error: {e}"

    return render_template("form.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)

#if __name__ == "__main__":
 #   app.run(debug=True)
