# app.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def healthcheck():
    return {"status": "ok"}
# demo.py
import gradio as gr
import torch
import numpy as np
from PIL import Image
from model import enhance_net_nopool as ZeroDCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ZeroDCE().to(device)
model.load_state_dict(torch.load("snapshots/Epoch99.pth", map_location=device))
model.eval()

def enhance_fn(img: np.ndarray) -> np.ndarray:
    img_t = torch.from_numpy(img.transpose(2,0,1)/255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t).squeeze(0).cpu().numpy().transpose(1,2,0)
    return (out * 255).clip(0,255).astype(np.uint8)

demo = gr.Interface(
    fn=enhance_fn,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Zero‑DCE Low‑Light Enhancement",
    description="Upload a low‑light image and get an enhanced version."
)

if __name__ == "__main__":
    demo.launch()
