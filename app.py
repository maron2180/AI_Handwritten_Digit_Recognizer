

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# --------------------------------
# 1. Streamlit configuration
# --------------------------------

st.set_page_config(
    page_title="AI Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)


# --------------------------------
# 2. Define CNN architecture
# --------------------------------

class DigitCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# --------------------------------
# 3. Load trained model
# --------------------------------

@st.cache_resource
def load_model():

    model = DigitCNN()

    model.load_state_dict(
        torch.load(
            "digit_cnn.pth",
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    return model


model = load_model()


# --------------------------------
# 4. Application interface
# --------------------------------

st.title("🔢 AI Handwritten Digit Recognizer")

st.write(
    "Draw a handwritten digit from 0 to 9 "
    "and let the CNN predict it!"
)

st.write("### Drawing Canvas")

canvas_result = st_canvas(

    fill_color="rgba(255, 255, 255, 1)",

    stroke_width=18,

    stroke_color="#FFFFFF",

    background_color="#000000",

    height=280,

    width=280,

    drawing_mode="freedraw",

    key="digit_canvas"

)


# --------------------------------
# 5. Prediction
# --------------------------------

if st.button("Predict Digit", type="primary"):

    if canvas_result.image_data is None:

        st.warning("Please draw a digit first!")

    else:

        # Convert canvas to grayscale
        img = Image.fromarray(
            canvas_result.image_data.astype(np.uint8)
        ).convert("L")

        # Check whether the canvas is empty
        if np.array(img).max() == 0:

            st.warning("Please draw a digit first!")

        else:

            # Resize to MNIST dimensions
            img = img.resize((28, 28))

            # Convert image to tensor
            img_array = np.array(img).astype(
                np.float32
            ) / 255.0

            tensor = torch.tensor(
                img_array
            ).unsqueeze(0).unsqueeze(0)

            # Model prediction
            with torch.no_grad():

                outputs = model(tensor)

                probabilities = torch.softmax(
                    outputs, dim=1
                )[0]

                predicted_digit = torch.argmax(
                    probabilities
                ).item()

            confidence = probabilities[
                predicted_digit
            ].item()

            # Display prediction
            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            # Display all probabilities
            st.write("### Prediction Probabilities")

            probability_df = pd.DataFrame({
                "Digit": list(range(10)),
                "Probability": probabilities.numpy()
            })

            st.bar_chart(
                probability_df.set_index("Digit")
            )

            # Detailed results
            st.write("### Detailed Results")

            for digit, probability in enumerate(
                probabilities.tolist()
            ):

                st.write(
                    f"Digit {digit}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

