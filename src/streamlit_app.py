import streamlit as st
from streamlit_drawable_canvas import st_canvas
from searcher import Searcher
import numpy as np
from PIL import Image, ImageDraw
import requests
import os
import subprocess


def main():
    st.sidebar.header("Configuration")

    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    topk = st.sidebar.slider("Retrieve top k: ", 1, 100, 30)
    ncols = st.sidebar.slider("Number of columns: ", 1, 10, 3)
    stroke_color = st.sidebar.color_picker("Stroke color: ")
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon")
    )

    draw_boxes = st.sidebar.checkbox('Draw bboxes', True)
    resize_images = st.sidebar.checkbox('Resize images', False)

    dataset = st.sidebar.selectbox("Dataset:", ("OpenImages", "Unsplash", "Custom"))
    custom = dataset == 'Custom'
    ds_root = None
    indexlist = os.listdir(os.path.join(os.path.dirname(__file__), '../indexes/'))
    if len(indexlist) == 0:
        st.warning("Index not found, downloading!")
        subprocess.call('gdown https://drive.google.com/uc?id=1-d43C1sDRAXDK6VrWdZUDdbckgHpq6Ap', shell=True)
    index_name = st.sidebar.selectbox("Index:", os.listdir(os.path.join(os.path.dirname(__file__))))

    if custom:
        ds_root = st.sidebar.text_input("Dataset Root")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        update_streamlit=True,
        height=480,
        width=720,
        drawing_mode=drawing_mode,
        display_toolbar=True
    )

    search_button = st.button('Search!')

    if search_button:
        s = Searcher(dataset, index_name, ds_root)
        cols = st.beta_columns(ncols)
        img = canvas_result.image_data.astype(np.uint8)

        urls, qmasks = s.search(img, topk)
        k = 0
        for url in urls:
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

                with cols[k % ncols]:
                    st.image(result)
                    k += 1
            except Exception as e:
                st.warning(f"Image missing at url: {url}")
                print(e)


if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), f'../indexes/'), exist_ok=True)
    st.set_page_config(layout="wide", page_title="Compositional Sketch Search")
    main()
