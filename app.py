import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Insect Pest Classification",
    page_icon="🐛",
    layout="centered"
)

# =====================================================
# CUSTOM STYLE
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================

st.title("🐛 Insect Pest Classification")

st.markdown("""
This application classifies insect pest images using a deep learning model based on:

✅ ResNet50  
✅ Transfer Learning  
✅ PyTorch  
✅ Streamlit Deployment  
""")

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# LOAD CLASS NAMES
# =====================================================

with open("classes.txt", "r") as f:

    class_names = [
        line.strip()
        for line in f.readlines()
    ]

NUM_CLASSES = len(class_names)

# =====================================================
# IMAGE TRANSFORM
# =====================================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = models.resnet50(pretrained=False)

    model.fc = torch.nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            "models/resnet50_best.pth",
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model

# =====================================================
# LOAD RESNET MODEL
# =====================================================

model = load_model()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("About")

st.sidebar.info("""
This project performs insect pest classification using deep learning.

Model Used:
- ResNet50

Frameworks:
- PyTorch
- Streamlit

Dataset:
- IP102 Dataset
""")

# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload an insect image",
    type=["jpg", "jpeg", "png"]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    input_tensor = transform(image).unsqueeze(0).to(device)

    # =================================================
    # PREDICT BUTTON
    # =================================================

    if st.button("Predict"):

        with st.spinner("Analyzing Image..."):

            with torch.no_grad():

                output = model(input_tensor)

                probs = F.softmax(output, dim=1)

                confidence, predicted = torch.max(
                    probs,
                    1
                )

                predicted_class = class_names[
                    predicted.item()
                ]

                confidence_score = confidence.item() * 100

        # =============================================
        # RESULTS
        # =============================================

        st.success("Prediction Completed Successfully")

        st.markdown("## Prediction Result")

        st.markdown(
            f"### 🐞 {predicted_class}"
        )

        st.markdown(
            f"### Confidence: {confidence_score:.2f}%"
        )

        # =============================================
        # TOP 5 PREDICTIONS
        # =============================================

        st.markdown("## Top 5 Predictions")

        top5_prob, top5_catid = torch.topk(
            probs,
            5
        )

        for i in range(5):

            cls_name = class_names[
                top5_catid[0][i]
            ]

            prob = top5_prob[0][i].item() * 100

            st.write(
                f"{i+1}. {cls_name} — {prob:.2f}%"
            )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    "Developed using PyTorch and Streamlit"
)