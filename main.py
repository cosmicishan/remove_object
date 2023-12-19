import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator

model =  YOLO('yolov8n.pt')

def object_recognition(image, results):

    detected = []

    image = np.array(image)

    for r in results:

        annotator = Annotator(image)

        boxes = r.boxes
        for box in boxes:

            b = box.xyxy[0]  
            c = box.cls

            if model.names[int(c)] not in detected:

                detected.append(model.names[int(c)])

            annotator.box_label(b, model.names[int(c)])

    img = annotator.result()

    return img, detected

def remove_object(image, results, remove):

    image = np.array(image)

    for r in results:

        annotator = Annotator(image)

        boxes = r.boxes
        for box in boxes:

            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls

            if remove == model.names[int(c)]:

                cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), thickness = cv2.FILLED)

            else:

                annotator.box_label(b, model.names[int(c)])

    img = annotator.result()

    return img

def process_image(image):

    results = model.predict(source=image, conf=0.75)

    img, detected = object_recognition(image, results)

    st.image(img, caption="Detected Objects", use_column_width=True)

    chosen_field = None

    chosen_field = st.selectbox("Choose an object to remove:", ["Display All"] + detected)

    img = remove_object(image, results, chosen_field)

    st.image(img, caption="Removed Object", use_column_width=True)

def main():
    st.title("Image and Text Display App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        process_image(image)

# Run the Streamlit app
if __name__ == "__main__":
    main()
