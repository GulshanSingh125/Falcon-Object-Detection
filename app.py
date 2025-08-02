import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model(model_path):
    """Loads the YOLOv8 model from the specified path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Please make sure the path to your 'best.pt' file is correct.")
        return None

def show_home_page():
    """Displays the content for the Home page."""
    st.title("Welcome to the Falcon Project")
    st.image("falcon.jpg", use_container_width=True)
    st.header("Continuous Model Improvement with Falcon")

    st.markdown("""
    A machine learning model is never truly "finished." The real world is constantly changing, and to stay accurate, the model must learn and adapt. The **Falcon** workflow is designed for exactly this purpose: to continuously monitor, update, and improve our object detection model over time.
    """)

    st.subheader("The Falcon Maintenance Loop")
    st.markdown("""
    Our process follows a simple, powerful four-step loop:

    **1. Monitor & Collect 🕵️‍♀️**
    * The first step is to identify where the model is struggling. We collect new images from the real world, especially examples where the model makes a mistake.

    **2. Annotate & Augment 🏷️**
    * The new images are labeled with the correct ground truth bounding boxes. This new, high-quality data is the key to teaching the model.

    **3. Re-Train 🧠**
    * The newly labeled images are added to our original dataset. The model is then re-trained on this new, expanded dataset using a powerful cloud GPU backend.

    **4. Evaluate & Deploy 🚀**
    * The newly trained model is rigorously tested. If its performance is better than the old model, it is deployed and replaces the previous version in this application.
    """)

    st.subheader("How Falcon Solves Real-World Problems")
    st.markdown("""
    **Scenario 1: An object's appearance changes.**
    * **Problem:** A new design for a "Fire Extinguisher" is introduced, and the model fails to detect it.
    * **Falcon's Solution:** We collect images of the new design, label them, and **re-train** the model. The updated model now recognizes both designs.

    **Scenario 2: A new, confusing object is introduced.**
    * **Problem:** A new red "First-Aid Box" is added, and the model incorrectly identifies it as a "Fire Extinguisher."
    * **Falcon's Solution:** We collect images of the first-aid box, add them to the dataset, and **re-train**. The model then learns to distinguish between the two objects.
    """)

def show_detection_page():
    """Displays the content for the YOLO Object Detection page."""
    st.title("YOLO Object Detection")
    st.image("yoloimg2.jpg", use_container_width=True)
    st.write("Upload an image and the model will detect objects and display the results.")

    YOUR_MODEL_PATH = r"best.pt"

    model = load_model(YOUR_MODEL_PATH)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file)
        
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("")
        st.write("Detecting objects...")

        results = model.predict(image)
        result = results[0]
        
        annotated_image = result.plot()
        annotated_image_rgb = annotated_image[..., ::-1]

        st.image(annotated_image_rgb, caption='Detected Image', use_container_width=True)

        if len(result.boxes) > 0:
            st.write("### Detection Details:")
            for box in result.boxes:
                class_name = model.names[int(box.cls[0])]
                confidence = box.conf[0]
                st.write(f"- **Object:** {class_name}, **Confidence:** {confidence:.2f}")
        else:
            st.write("No objects detected.")

# --- NEW: Function for the Use Case Page ---
def show_use_case_page():
    """Displays the content for the Real-World Use Case page."""
    st.header("Solving Real-World Problems with Falcon")
    st.subheader("Use Case: Automated Safety & Compliance Audits")

    st.markdown("""
    In industrial environments like workshops, factories, and construction sites, ensuring safety compliance is critical. Regular audits are required to check that essential equipment, such as fire extinguishers and oxygen tanks, are present and correctly placed.
    """)

    st.subheader("The Problem: Manual and Inefficient Audits")
    st.markdown("""
    Traditionally, these safety audits are performed manually. A safety officer must walk through the entire site with a checklist, visually verifying each item. This process is:
    - **Time-Consuming:** Large sites can take hours or days to inspect thoroughly.
    - **Prone to Human Error:** It's easy to overlook a missing item or incorrectly mark a checklist.
    - **Difficult to Document:** Paper-based records are hard to track, audit, and analyze over time.
    """)
    
    # You can uncomment the line below if you have a relevant image to display
    st.image("yoloimg.png", caption="Manual safety checks can be slow and prone to error.", use_container_width=True)

    st.subheader("The Falcon Solution: Real-Time, Intelligent Verification")
    st.markdown("""
    The **Falcon** application transforms this process. By leveraging our custom-trained YOLOv8 model, we can automate and enhance safety audits.

    A safety officer can simply use a device with a camera (like a phone or tablet) to scan an area. Falcon provides instant feedback:

    **1. Real-Time Detection:** The model immediately identifies critical safety assets like **Fire Extinguishers**, **Oxygen Tanks**, and **Toolboxes** in the camera's view.

    **2. Compliance Verification:** The system can be programmed to verify safety rules automatically. For example, it can confirm that a fire extinguisher is present in a designated zone or flag an area where one is missing.

    **3. Automated Reporting:** Falcon can instantly generate a digital report with time-stamped photographic evidence of the audit. Any non-compliance issues are flagged immediately for corrective action, creating a seamless and verifiable audit trail.
    """)

    st.subheader("Benefits of Using Falcon")
    st.markdown("""
    * **Increased Efficiency:** Reduce audit times from hours to minutes.
    * **Improved Accuracy:** Eliminate human error by using a highly accurate deep learning model.
    * **Enhanced Safety:** Ensure faster identification of missing or misplaced safety equipment.
    * **Digital Record-Keeping:** Maintain a secure, searchable, and permanent digital log of all safety checks.
    """)

# --- Main App Logic ---

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Sidebar for navigation
st.sidebar.title('Navigation')
if st.sidebar.button('Home'):
    st.session_state.page = 'Home'
if st.sidebar.button('YOLO Object Detection'):
    st.session_state.page = 'YOLO Object Detection'
# --- NEW: Add button for the Use Case Page ---
if st.sidebar.button('Real-World Use Case'):
    st.session_state.page = 'Use Case'

# Page routing
if st.session_state.page == 'Home':
    show_home_page()
elif st.session_state.page == 'YOLO Object Detection':
    show_detection_page()
# --- NEW: Add routing for the Use Case Page ---
elif st.session_state.page == 'Use Case':
    show_use_case_page()