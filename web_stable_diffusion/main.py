import time

import torch
import numpy as np
import streamlit as st
from PIL import Image

from diffuse import CustomDiffusionPipeline, generate_image


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


# inference parameters
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"
PIPE = init_pipeline(DEVICE, fp16=False, auth=True)


def main():
    with st.form(key="request"):
        st.write(
            """
    #  Stable Diffusion
    Type some text in prompt area and see a generated image.
    """
        )

        prompt = st.text_area("Prompt", value="cat with a red hat in a forest", key="prompt")
        left_column, right_column = st.columns(2)
        with left_column:
            iterations = int(st.text_input("Iterations", value=50, key="iterations"))
        with right_column:
            seed = int(st.text_input("Random seed", value=42, key="seed"))
        nsfw = st.checkbox("Filter NSFW content", value=True, key="nsfw")
        st.form_submit_button("Submit")

        if prompt:
            print(prompt, iterations, seed, nsfw)
            image = generate_image(PIPE, DEVICE, prompt=prompt, iterations=iterations, initial_seed=seed, nsfw=nsfw)
            st.image(image, use_column_width=True)


if __name__ == "__main__":
    main()
