import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download
import time

st.set_page_config(page_title="Malaria Detection", page_icon="🦠", layout="centered")

st.title("Malaria Parasite Detection")

model_path = hf_hub_download(repo_id="adebayokunle/malaria-detection", filename="malaria_ensemble_model.pkl")

model = joblib.load(model_path)
pre_trained_model = ResNet50(weights='imagenet', include_top=False)


def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(180, 180))  # Resize to 224x224
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Apply ResNet50 preprocessing
    features = model.predict(img_array)  # Extract features
    return features.flatten()  # Flatten to 1D vector

st.write("Upload an image of a blood cell to check if it is parasitized or uninfected.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    with st.spinner("🔍 Analyzing image... Please wait."):
        time.sleep(2)  # Simulate processing time
    
    # Preprocess the image
        img = extract_features(uploaded_file, pre_trained_model)

        features = img.reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)

    
    # Display the result
    if prediction[0] == 0:
        st.error(f"🛑 **Malaria Detected!**")
        st.write("💉 **Diagnosis:** The image is classified as **Parasitized**.")
        st.write("🔬 **Recommendation:** Consult a doctor immediately for confirmation and treatment.")

    else:
        st.success(f"✅ **No Malaria Detected!**")
        st.write("🩺 **Diagnosis:** The image is classified as **Uninfected**.")
        st.write("🌟 **Keep Maintaining Good Health!**")
     # Display Additional Information
    st.markdown("---")
    st.subheader("📌 Learn More About Malaria")
    st.write(
        "- **What is Malaria?** Malaria is a life-threatening disease caused by parasites transmitted through mosquito bites."
        "\n- **Symptoms:** Fever, chills, nausea, muscle pain, fatigue."
        "\n- **Prevention:** Use mosquito nets, insect repellents, and anti-malarial drugs."
        "\n- **Treatment:** Consult a medical professional for appropriate medications."
    )

    # Add a "Next Steps" button
    if st.button("📚 Learn More About Malaria Prevention"):
        st.write("🔗 Visit [WHO Malaria Prevention Guide](https://www.who.int/health-topics/malaria)")

    # Add a GIF or Image for More Engagement
    st.image("https://media.giphy.com/media/UJ5RUcT7ImWQ/giphy.gif", caption="Stay Safe & Informed!", use_column_width=True)




    

# import streamlit as st
# import joblib
# import numpy as np
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.resnet50 import preprocess_input

# # Load the saved ensemble model
# try:
#     ensemble_model = joblib.load('malaria_ensemble_model.pkl')
#     st.write("Model loaded successfully!")
# except Exception as e:
#     st.error(f"Error loading model: {e}")

# # Function to preprocess the uploaded image
# def preprocess_image(image):
#     img = load_img(image, target_size=(128, 128))  # Resize the image
#     img = img_to_array(img)  # Convert to numpy array
#     img = preprocess_input(img)  # Preprocess for ResNet50
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# # Streamlit app
# st.title("Malaria Parasite Detection")
# st.write("Upload an image of a blood cell to check if it is parasitized or uninfected.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
#     try:
#         # Preprocess the image
#         img = preprocess_image(uploaded_file)
        
#         # Make a prediction
#         prediction = ensemble_model.predict(img)
        
#         # Display the result
#         if prediction[0] == 1:
#             st.write("**Prediction:** Parasitized")
#         else:
#             st.write("**Prediction:** Uninfected")
#     except Exception as e:
#         st.error(f"Error processing image: {e}")



