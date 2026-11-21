
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ==========================================
# 1. Streamlit page configuration
# ==========================================

st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)


# ==========================================
# 2. CNN Model
# ==========================================

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# ==========================================
# 3. Load trained model
# ==========================================

@st.cache_resource
def load_model():
    model = DigitCNN()

    model.load_state_dict(
        torch.load("digit_cnn.pth", map_location="cpu")
    )

    model.eval()

    return model


model = load_model()


# ==========================================
# 4. App title and description
# ==========================================

st.title("🔢 Handwritten Digit Recognizer")

st.write(
    "Draw a digit from 0 to 9 in the canvas below, "
    "and let the CNN model predict what you wrote!"
)


# ==========================================
# 5. Drawing canvas
# ==========================================

st.subheader("Draw your digit here")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="digit_canvas"
)


# ==========================================
# 6. Prediction
# ==========================================

if st.button("Predict", type="primary"):

    if canvas_result.image_data is None:
        st.warning("Please draw a digit first.")

    else:
        # Convert canvas image to NumPy array
        img_array = canvas_result.image_data.astype(np.uint8)

        # Convert RGBA image to grayscale
        img = Image.fromarray(img_array).convert("L")

        # Resize image to MNIST dimensions (28 x 28)
        img = img.resize((28, 28))

        # Convert image to NumPy array
        img_array = np.array(img).astype(np.float32)

        # Normalize pixel values to [0, 1]
        img_array /= 255.0

        # Check whether the canvas is blank
        if img_array.max() < 0.05:
            st.warning("Please draw a digit before predicting.")

        else:
            # Convert to PyTorch tensor
            # Shape: [1, 1, 28, 28]
            input_tensor = torch.tensor(
                img_array,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)

            # Run model inference
            with torch.no_grad():
                outputs = model(input_tensor)

                probabilities = F.softmax(outputs, dim=1)

                predicted_digit = torch.argmax(
                    probabilities, dim=1
                ).item()

            # ==========================================
            # 7. Display prediction results
            # ==========================================

            confidence = probabilities[0][predicted_digit].item()

            st.markdown(
                f"""
                <div style="font-size: 32px; font-weight: bold; color: #28a745;">
                    Predicted Digit: {predicted_digit}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%"
            )

            # Display all class probabilities
            st.subheader("Class Probabilities")

            prob_array = probabilities[0].numpy()

            for digit, prob in enumerate(prob_array):
                st.write(f"Digit {digit}: {prob * 100:.2f}%")

                st.progress(float(prob))

