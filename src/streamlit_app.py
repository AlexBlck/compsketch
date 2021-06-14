import streamlit as st
from streamlit_drawable_canvas import st_canvas
from searcher import Searcher
import cv2
import numpy as np
from PIL import Image, ImageDraw
import requests
import os
import torchvision.transforms.functional as TF


def main():
    st.sidebar.header("Configuration")

    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    topk = st.sidebar.slider("Retrieve top k: ", 1, 100, 10)
    stroke_color = st.sidebar.color_picker("Stroke color: ")
    # bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon")
    )
    # can't use, bugs out
    # canvas_shape = st.sidebar.selectbox("Canvas shape:", ("square", "landscape", "portrait"))

    draw_boxes = st.sidebar.checkbox('Draw bboxes', True)
    resize_images = st.sidebar.checkbox('Resize images', False)

    dataset = st.sidebar.selectbox("Dataset:", ("OpenImages", "Custom"))
    custom = dataset == 'Custom'
    ds_root = None
    index_name = st.sidebar.selectbox("Index:", os.listdir(os.path.join(os.path.dirname(__file__), '../indexes/')))
    if custom:
        ds_root = st.sidebar.text_input("Dataset Root")

    # if canvas_shape == 'square':
    #     canvas_width = 600
    #     canvas_height = 600
    # elif canvas_shape == 'landscape':
    #     canvas_width = 702
    #     canvas_height = 480
    # else:
    #     canvas_width = 480
    #     canvas_height = 720

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        # background_color=bg_color,
        # background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=True,
        height=480,
        width=720,
        drawing_mode=drawing_mode,
        display_toolbar=True,
        key="full_app",
    )

    search_button = st.button('Search!')

    if search_button:
        s = Searcher(custom, index_name, ds_root)
        cols = st.beta_columns(3)
        img = canvas_result.image_data.astype(np.uint8)

        urls, qmasks = s.search(img, topk)
        k = 0
        for k, url in enumerate(urls):
            try:
                if custom:
                    result = url
                else:
                    result = Image.open(requests.get(url, stream=True).raw).convert('RGB')
                if resize_images:
                    result = result.resize((600, 600))
                if draw_boxes:
                    draw = ImageDraw.Draw(result)
                    for qmask in qmasks:
                        box = [x / 31 for x in qmask]
                        draw.rectangle([box[2] * result.width, box[0] * result.height,
                                        box[3] * result.width, box[1] * result.height],
                                       outline=(0, 0, 255), width=int(result.width * 0.005))

                with cols[k % 3]:
                    st.image(result)
                    k += 1
            except Exception as e:
                st.warning(f"Image missing at url: {url}")
                print(e)


if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Compositional Sketch Search")
    main()
