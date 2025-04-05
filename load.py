import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Define the model architectur

# === Class label mapping === #
label_mapping = {
    0: "Dos",
    1: "Theft",
    2: "DDos",
    3: "Normal",
    4: "Reconnaissance"
}

# === Required features in order === #
required_features = [
    'pkts', 'bytes', 'sbytes', 'dbytes', 'rate',
    'mean', 'stddev', 'sum', 'proto', 'state',
     'dur'
]

# === Load the trained model === #
try:
    # Use same SpinalSAENet class definition as during training
    model = SpinalSAENet(input_dim=11, hidden_dims=[128, 64], num_classes=5).to(device)
    model.load_state_dict(torch.load("/kaggle/working/spinal_saenet_best.pth", map_location=device))
    model.eval()

    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    raise

# === Load the scaler === #
try:
    scaler = joblib.load('/kaggle/working/scaler.pkl')
    print("Scaler loaded successfully.")
except Exception as e:
    print("Error loading scaler:", e)
    raise

# === Direct custom input (your example) === #
custom_input = {
    'pkts': 100,
    'bytes': 5000,
    'sbytes': 3000,
    'dbytes': 2000,
    'rate': 50.0,
    'mean': 25.0,
    'stddev': 10.0,
    'sum': 10000,
    'proto': 0,  # Encoded value (e.g., 0 for TCP)
    'state': 1,  # Encoded value (e.g., 1 for CON)
    # 'stime': 1620000000,
    'dur': 0.5
}

# === Prediction pipeline === #
try:
    # Convert input to array
    input_array = np.array([[custom_input[feat] for feat in required_features]], dtype=np.float32)

    # Scale the input
    input_scaled = scaler.transform(input_array)

    # Convert to torch tensor
    X_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
        predicted_class = int(np.argmax(probs))
        confidence = probs[predicted_class]

    print("\n=== Prediction Result ===")
    print(f"Predicted Label: {label_mapping[predicted_class]}")
    print(f"Confidence: {confidence:.4f}")
    print("Class Probabilities:")
    for idx, prob in enumerate(probs):
        print(f"  {label_mapping[idx]}: {prob:.4f}")

except Exception as e:
    print("Prediction failed:", e)