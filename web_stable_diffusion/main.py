
import torch
import tabulate
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import PIL
import cv2
import numpy as np

from diffuse import CustomDiffusionPipeline, generate_image

st.set_page_config(initial_sidebar_state="collapsed")

@st.cache(allow_output_mutation=True)
def init_pipeline(device, fp16=False, auth=True):
    kwargs = {}
    model_id = "CompVis/stable-diffusion-v1-4"
    kwargs = {}
    if fp16:
        kwargs["revision"] = "fp16"
        kwargs["torch_dtype"] = torch.float16
    if auth:
        kwargs["use_auth_token"] = True

    pipe = CustomDiffusionPipeline.from_pretrained(model_id, **kwargs)
    pipe.to(device)
    return pipe


def main(device, pipe):
    with st.form(key="request"):
        st.write(
            """
    #  Stable Diffusion
    Type some text in prompt area and see a generated image.
    """
        )

        prompt = st.text_area("Prompt", value="", key="prompt")
        with st.expander("Initial image"):
            stroke_width = st.sidebar.slider("stroke width", 1, 100, 50)
            stroke_color = st.sidebar.color_picker("stroke color", "#00FF00")
            canvas = st_canvas(
                fill_color="rgb(0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color="#000000",
                height=512,
                width=512,
                update_streamlit=True,
                drawing_mode="freedraw",
                key="canvas"
            )

        left_column, right_column = st.columns(2)
        with left_column:
            iterations = int(st.text_input("Iterations", value=50, key="iterations"))
        with right_column:
            seed = int(st.text_input("Random seed", value=42, key="seed"))
        nsfw = st.checkbox("Filter NSFW content", value=True, key="nsfw")
        st.form_submit_button("Submit")

        if prompt:
            if canvas.image_data is not None:
                img_arr = cv2.cvtColor(canvas.image_data, cv2.COLOR_BGRA2BGR)
                # ignore black images
                if np.all(img_arr == 0):
                    initial_image = None
                else:
                    initial_image = PIL.Image.fromarray(img_arr)
            else:
                initial_image = None

            info = [
                ("Device", device),
                ("Prompt", prompt),
                ("Initial image", initial_image is None),
                ("Iterations", iterations),
                ("Seed", seed),
                ("NSFW", nsfw)
            ]
            print(tabulate.tabulate(info))
            image = generate_image(pipe, device, prompt=prompt, initial_image=initial_image, iterations=iterations, initial_seed=seed, nsfw=nsfw)
            st.image(image, use_column_width=True)

if __name__ == "__main__":
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    pipeline = init_pipeline(device, fp16=True, auth=True)
    main(device, pipeline)
