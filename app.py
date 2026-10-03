import streamlit as st
import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="OpenCV Image Filters",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ OpenCV Image Filters")
st.write("Upload an image and apply different OpenCV filters.")


# --------------------------------------------------
# Upload Image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Convert PIL image to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Store selected result
    if "filter_result" not in st.session_state:
        st.session_state.filter_result = img

    if "filter_name" not in st.session_state:
        st.session_state.filter_name = "Original Image"


    # --------------------------------------------------
    # Five Filter Buttons
    # --------------------------------------------------

    st.subheader("🎨 Choose a Filter")

    col1, col2, col3, col4, col5 = st.columns(5)


    # Button 1 - Grayscale
    with col1:
        if st.button("⚫ Grayscale", use_container_width=True):

            result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            st.session_state.filter_result = result
            st.session_state.filter_name = "Grayscale"


    # Button 2 - Blur
    with col2:
        if st.button("🌫️ Blur", use_container_width=True):

            result = cv2.GaussianBlur(
                img,
                (15, 15),
                0
            )

            st.session_state.filter_result = result
            st.session_state.filter_name = "Gaussian Blur"


    # Button 3 - Edge Detection
    with col3:
        if st.button("📐 Edge Detection", use_container_width=True):

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            result = cv2.Canny(
                gray,
                100,
                200
            )

            st.session_state.filter_result = result
            st.session_state.filter_name = "Canny Edge Detection"


    # Button 4 - Image Effects
    with col4:

        effect = st.selectbox(
            "🎨 Effects",
            [
                "Select Effect",
                "Sepia",
                "Negative"
            ]
        )

        if st.button("✨ Apply Effect", use_container_width=True):

            if effect == "Sepia":

                sepia = np.array(
                    [
                        [0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]
                    ]
                )

                result = cv2.transform(
                    img,
                    sepia
                )

                result = np.clip(
                    result,
                    0,
                    255
                ).astype(np.uint8)

                st.session_state.filter_result = result
                st.session_state.filter_name = "Sepia"

            elif effect == "Negative":

                result = cv2.bitwise_not(img)

                st.session_state.filter_result = result
                st.session_state.filter_name = "Negative"


    # Button 5 - More Filters
    with col5:

        more_filter = st.selectbox(
            "🔧 More Filters",
            [
                "Select Filter",
                "Sharpen",
                "Threshold"
            ]
        )

        if st.button("🔧 Apply Filter", use_container_width=True):

            if more_filter == "Sharpen":

                kernel = np.array(
                    [
                        [0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]
                    ]
                )

                result = cv2.filter2D(
                    img,
                    -1,
                    kernel
                )

                st.session_state.filter_result = result
                st.session_state.filter_name = "Sharpen"

            elif more_filter == "Threshold":

                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2GRAY
                )

                _, result = cv2.threshold(
                    gray,
                    127,
                    255,
                    cv2.THRESH_BINARY
                )

                st.session_state.filter_result = result
                st.session_state.filter_name = "Binary Threshold"


    # --------------------------------------------------
    # Display Result
    # --------------------------------------------------

    st.divider()

    st.subheader(
        f"🔍 Result: {st.session_state.filter_name}"
    )

    result = st.session_state.filter_result

    # Convert OpenCV BGR to RGB for Streamlit
    if len(result.shape) == 3:
        result_rgb = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )
    else:
        result_rgb = result


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Image")

        original_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            original_rgb,
            use_container_width=True
        )

    with col2:
        st.markdown(
            f"### {st.session_state.filter_name}"
        )

        st.image(
            result_rgb,
            use_container_width=True
        )


else:

    st.info(
        "👆 Please upload a JPG, JPEG, or PNG image to begin."
    )