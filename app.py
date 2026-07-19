import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="AI Car Detection Dashboard",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 AI Car Detection Dashboard")
st.caption("Powered by YOLOv8 & Streamlit")
st.write("---")

# Put your trained weights here
MODEL_PATH = "./models/best.pt"

@st.cache_resource
def load_yolo_model():
    return YOLO(MODEL_PATH)

try:
    model = load_yolo_model()
except Exception:
    st.error(
        "⚠️ Could not load the trained model weights. "
        "Make sure you placed 'best.pt' into car_app/models/best.pt"
    )
    st.stop()

st.sidebar.header("🔧 Settings")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
)

uploaded_file = st.file_uploader(
    "Choose an image to scan...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detections")
        with st.spinner("Running YOLOv8 inference..."):
            results = model.predict(source=image, conf=confidence)
            res_plotted = results[0].plot()
            detected_image = Image.fromarray(res_plotted[..., ::-1])

            st.image(detected_image, use_container_width=True)

            car_count = len(results[0].boxes)
            st.metric(label="Vehicles Detected", value=car_count)

