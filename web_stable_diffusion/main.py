import io
import pathlib

from PIL import Image
import torch

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


from diffuse import NSFW_Enabled_DiffusionPipeline


BASE_DIR = pathlib.Path(__file__).resolve().parent

app = FastAPI(docs_url="/docs")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# inference parameters
pipe = None
device = "cpu" if not torch.cuda.is_available() else "cuda"
fp16 = False
auth = True

@app.on_event("startup")
async def startup_event():
    global pipe

    model_id = "CompVis/stable-diffusion-v1-4"
    kwargs = {}
    if fp16:
        kwargs["revision"] = "fp16"
        kwargs["torch_dtype"] = torch.float16
    if auth:
        kwargs["use_auth_token"] = True

    pipe = NSFW_Enabled_DiffusionPipeline.from_pretrained(model_id, **kwargs)
    pipe.to(device)


@app.get("/", response_class=HTMLResponse)
async def intro(request: Request):
    return templates.TemplateResponse("intro.html", {"request": request})


def generate_image(prompt: str, iterations: int = 50, seed: int = 123, nsfw: bool = True) -> Image:
    with torch.autocast(device):
        generator = torch.Generator(device).manual_seed(seed)
        images = pipe(prompt, num_inference_steps=iterations, generator=generator, nsfw=nsfw)["sample"]
    return images[0]


@app.get("/generate", response_class=StreamingResponse)
async def generate(
    prompt: str = "cat in the forest",
    seed: int = 42,
    iterations: int = 50,
    nsfw: bool = True,
):
    img = generate_image(prompt, iterations, seed, nsfw)
    imgio = io.BytesIO()
    img.save(imgio, "JPEG", quality=100)
    imgio.seek(0)
    return StreamingResponse(
        imgio,
        media_type="image/jpeg",
        headers={"Content-Disposition": 'inline; filename="diffusion.jpg"'},
    )

