import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision.ops as ops
from PIL import Image
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="D0 Insect Classification",
    page_icon="🐛",
    layout="centered"
)

# =====================================================
# STYLE
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3, h4 {
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

st.title("🐛 D0 Insect Classification")

st.markdown("""
This application classifies insect images using:

✅ ResNet50 + FPN  
✅ PyTorch  
✅ Streamlit
""")

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# D0 CLASS ORDER
# IMPORTANT:
# MUST MATCH EXACT TRAINING FOLDER ORDER
# =====================================================

class_names = [

    "0",
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "2",
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "3",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "39",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

NUM_CLASSES = len(class_names)

# =====================================================
# IMAGE TRANSFORM
# MUST MATCH TRAINING
# =====================================================

transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.CenterCrop((224, 224)),

    transforms.ToTensor()
])

# =====================================================
# RESNET50 + FPN MODEL
# =====================================================

class ResNetFPN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        backbone = models.resnet50(weights=None)

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.fpn = ops.FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(
            256 * 4,
            num_classes
        )

    def forward(self, x):

        c1 = self.layer0(x)

        c2 = self.layer1(c1)

        c3 = self.layer2(c2)

        c4 = self.layer3(c3)

        c5 = self.layer4(c4)

        features = {
            "0": c2,
            "1": c3,
            "2": c4,
            "3": c5
        }

        pyramid = self.fpn(features)

        pooled = []

        for p in pyramid.values():

            pooled.append(
                self.pool(p).flatten(1)
            )

        out = torch.cat(pooled, dim=1)

        out = self.fc(out)

        return out

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model = ResNetFPN(NUM_CLASSES)

    checkpoint = torch.load(
        "models/resnet50_fpn_best.pth",
        map_location=device
    )

    # HANDLE DIFFERENT SAVE FORMATS

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # REMOVE module. PREFIX

    new_state_dict = {}

    for k, v in state_dict.items():

        if k.startswith("module."):

            k = k[7:]

        new_state_dict[k] = v

    model.load_state_dict(
        new_state_dict,
        strict=True
    )

    model.to(device)

    model.eval()

    return model

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model()

# =====================================================
# IMAGE UPLOADER
# =====================================================

uploaded_file = st.file_uploader(
    "Upload Insect Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
        "tiff"
    ]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        input_tensor = transform(image)

        input_tensor = input_tensor.unsqueeze(0).to(device)

        # =================================================
        # PREDICT BUTTON
        # =================================================

        if st.button("Predict"):

            with st.spinner("Analyzing Image..."):

                with torch.no_grad():

                    output = model(input_tensor)

                    # =====================================
                    # TEMPERATURE SCALING
                    # REDUCES FAKE 100% CONFIDENCE
                    # =====================================

                    probs = F.softmax(
                        output / 2.0,
                        dim=1
                    )

                    confidence, predicted = torch.max(
                        probs,
                        1
                    )

                    predicted_idx = predicted.item()

                    predicted_class = class_names[
                        predicted_idx
                    ]

                    confidence_score = (
                        confidence.item() * 100
                    )

            # =================================================
            # RESULTS
            # =================================================

            st.success(
                "Prediction Completed Successfully"
            )

            st.markdown("## Prediction Result")

            st.markdown(
                f"### 🐞 Predicted Class: {predicted_class}"
            )

            st.markdown(
                f"### Confidence: {confidence_score:.2f}%"
            )

            st.markdown(
                "### Model Used: ResNet50 + FPN"
            )

            # =================================================
            # TOP 5 PREDICTIONS
            # =================================================

            st.markdown("## Top 5 Predictions")

            top5_prob, top5_catid = torch.topk(
                probs,
                5
            )

            for i in range(5):

                cls_idx = top5_catid[0][i].item()

                cls_name = class_names[cls_idx]

                prob = (
                    top5_prob[0][i].item() * 100
                )

                st.write(
                    f"{i+1}. Class {cls_name} — {prob:.2f}%"
                )

    except Exception as e:

        st.error(
            f"Error loading image: {e}"
        )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    "Developed using PyTorch and Streamlit"
)