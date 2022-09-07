import torch
import tabulate
import streamlit as st

from diffuse import CustomDiffusionPipeline, generate_image


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
        left_column, right_column = st.columns(2)
        with left_column:
            iterations = int(st.text_input("Iterations", value=50, key="iterations"))
        with right_column:
            seed = int(st.text_input("Random seed", value=42, key="seed"))
        nsfw = st.checkbox("Filter NSFW content", value=True, key="nsfw")
        st.form_submit_button("Submit")

        if prompt:
            info = [("Device", device), ("Prompt", prompt), ("Iterations", iterations), ("Seed", seed), ("NSFW", nsfw)]
            print(tabulate.tabulate(info))
            image = generate_image(pipe, device, prompt=prompt, iterations=iterations, initial_seed=seed, nsfw=nsfw)
            st.image(image, use_column_width=True)


if __name__ == "__main__":
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    pipeline = init_pipeline(device, fp16=False, auth=True)
    main(device, pipeline)
