import os

import streamlit as st
import torch
from PIL import Image

from utils import CLASSES, GradCAM, build_model, get_transform, overlay_heatmap

st.set_page_config(page_title="Brain MRI Scan Reader", page_icon="🧠", layout="wide")

# Each class has its own color; the same color is used in the legend, result card and bars.
COLORS = {
    "glioma": "#E5368B",
    "meningioma": "#F59E0B",
    "notumor": "#10B981",
    "pituitary": "#6D5DF6",
}
LABELS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}
ABOUT = {
    "glioma": "A tumor that starts in the glial cells that support neurons.",
    "meningioma": "A tumor in the meninges, the membranes covering the brain.",
    "notumor": "The model found no pattern matching the three tumor types.",
    "pituitary": "A tumor in the pituitary gland at the base of the brain.",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"], .stMarkdown, p, label { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { max-width: 1100px; padding-top: 1.5rem; padding-bottom: 3rem; }

.hero {
  background: linear-gradient(120deg, #4F46E5 0%, #A23BEC 45%, #F0467F 80%, #FF8A3D 100%);
  border-radius: 28px; padding: 2.4rem 2.4rem 2rem; color: #fff;
  box-shadow: 0 18px 40px -18px rgba(109, 93, 246, .6);
}
.hero h1 {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: clamp(2rem, 5vw, 3.2rem); line-height: 1.05; margin: 0 0 .6rem; color: #fff;
}
.hero p { font-size: 1.05rem; max-width: 46ch; margin: 0 0 1.3rem; color: #FFF3FA; }
.legend { display: flex; flex-wrap: wrap; gap: .5rem; }
.chip {
  display: inline-flex; align-items: center; gap: .45rem; background: rgba(255,255,255,.95);
  color: #1B1740; border-radius: 999px; padding: .35rem .85rem .35rem .6rem;
  font-weight: 700; font-size: .85rem;
}
.dot { width: .75rem; height: .75rem; border-radius: 50%; display: inline-block; }

[data-testid="stFileUploaderDropzone"] {
  background: #fff; border: 2.5px dashed #6D5DF6; border-radius: 22px; padding: 1.6rem;
}
[data-testid="stFileUploaderDropzone"]:hover { background: #EFEBFF; }
[data-testid="stFileUploaderDropzone"] button {
  background: #6D5DF6; color: #fff; border: 0; border-radius: 999px; font-weight: 700;
}

.panel {
  background: #fff; border-radius: 22px; padding: 1.2rem 1.2rem 1rem;
  box-shadow: 0 10px 28px -16px rgba(27, 23, 64, .35);
}
.panel h3 { font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.15rem; margin: 0 0 .2rem; }
.panel .sub { color: #5B5780; font-size: .9rem; margin-bottom: .8rem; }
[data-testid="stImage"] img { border-radius: 16px; }

.result {
  border-radius: 24px; padding: 1.4rem 1.6rem; border: 3px solid var(--c);
  background: color-mix(in srgb, var(--c) 12%, white);
}
.result .tag {
  display: inline-block; background: var(--c); color: var(--ink); font-weight: 700;
  border-radius: 999px; padding: .25rem .8rem; font-size: .85rem; margin-bottom: .5rem;
}
.result h2 {
  font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800;
  font-size: clamp(1.8rem, 4vw, 2.6rem); margin: 0 0 .3rem; color: #1B1740;
}
.result p { margin: 0; color: #3A3563; }
.result .conf { font-size: 1.05rem; font-weight: 700; margin-top: .6rem; color: #1B1740; }

.bar-row { display: grid; grid-template-columns: 130px 1fr 56px; align-items: center; gap: .7rem; margin: .55rem 0; }
.bar-name { font-weight: 700; font-size: .92rem; }
.bar-track { background: #ECE9FA; border-radius: 999px; height: 14px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
.bar-val { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }

.heat-scale {
  height: 10px; border-radius: 999px; margin-top: .4rem;
  background: linear-gradient(90deg, #2B3AE8, #25C9E8, #8BE04E, #FFD43B, #FF4D4D);
}
.heat-labels { display: flex; justify-content: space-between; font-size: .8rem; color: #5B5780; }

.notice {
  background: #FFF4DB; border-left: 6px solid #F59E0B; border-radius: 14px;
  padding: .8rem 1.1rem; color: #5A3D00; margin-top: 1.2rem; font-size: .93rem;
}
.lowconf {
  background: #FFE9F2; border-left: 6px solid #E5368B; border-radius: 14px;
  padding: .7rem 1rem; color: #6B0F3A; margin-top: .8rem; font-size: .93rem;
}
@media (max-width: 640px) { .bar-row { grid-template-columns: 100px 1fr 48px; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists("model.pt"):
        return None
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model


chips = "".join(
    f'<span class="chip"><span class="dot" style="background:{COLORS[c]}"></span>{LABELS[c]}</span>'
    for c in CLASSES
)
st.markdown(
    f"""
    <div class="hero">
      <h1>Brain MRI Scan Reader</h1>
      <p>Upload an MRI slice. The model names the most likely class and shows where on the scan it looked.</p>
      <div class="legend">{chips}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()
if model is None:
    st.error("The trained model file model.pt is missing. Run `python train.py` first, then reload this page.")
    st.stop()

st.write("")
file = st.file_uploader("Upload a brain MRI slice (JPG or PNG)", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    x = get_transform()(img).unsqueeze(0)
    cam, probs, idx = GradCAM(model, model.layer4)(x)

    label = CLASSES[idx]
    color = COLORS[label]
    ink = "#FFFFFF" if label in ("pituitary", "glioma") else "#1B1740"
    st.write("")
    st.markdown(
        f"""
        <div class="result" style="--c:{color}; --ink:{ink}">
          <span class="tag">Most likely result</span>
          <h2>{LABELS[label]}</h2>
          <p>{ABOUT[label]}</p>
          <div class="conf">Confidence: {probs[idx]:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if probs[idx] < 0.6:
        st.markdown(
            '<div class="lowconf">The model is unsure about this image. '
            "Try a clearer slice, or treat this result with extra caution.</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="panel"><h3>Your scan</h3><div class="sub">Image as uploaded</div></div>',
                    unsafe_allow_html=True)
        st.image(img, use_container_width=True)
    with col2:
        st.markdown(
            '<div class="panel"><h3>Where the model looked</h3>'
            '<div class="sub">Warm colors mark areas that influenced the result most</div></div>',
            unsafe_allow_html=True,
        )
        st.image(overlay_heatmap(img, cam), use_container_width=True)
        st.markdown(
            '<div class="heat-scale"></div>'
            '<div class="heat-labels"><span>Less influence</span><span>More influence</span></div>',
            unsafe_allow_html=True,
        )

    bars = ""
    for i in sorted(range(len(CLASSES)), key=lambda k: -probs[k]):
        name = CLASSES[i]
        bars += (
            f'<div class="bar-row"><span class="bar-name">{LABELS[name]}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{probs[i]*100:.1f}%;'
            f'background:{COLORS[name]}"></div></div>'
            f'<span class="bar-val">{probs[i]:.1%}</span></div>'
        )
    st.write("")
    st.markdown(
        f'<div class="panel"><h3>How the model scored each class</h3>'
        f'<div class="sub">All four scores add up to 100%</div>{bars}</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("Upload an MRI image above to see the result.")

st.markdown(
    '<div class="notice"><b>For learning only.</b> This is a student project, not a medical device. '
    "It cannot diagnose anything. Please see a qualified doctor for any health concern.</div>",
    unsafe_allow_html=True,
)
