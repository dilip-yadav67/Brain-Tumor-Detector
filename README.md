# Brain Tumor Detector (MRI classification + Grad-CAM)

Classifies brain MRI slices into **glioma, meningioma, pituitary, or no tumor** using a
fine-tuned ResNet18, and shows a Grad-CAM heatmap of what the model looked at.

> Educational project only. Not a medical device. Never use it for diagnosis.

## Setup

```bash
pip install torch torchvision scikit-learn opencv-python pillow streamlit
```

## Dataset

Download "Brain Tumor MRI Dataset" (Masoud Nickparvar) from Kaggle and arrange it as:

```
data/
  Training/{glioma,meningioma,notumor,pituitary}/
  Testing/{glioma,meningioma,notumor,pituitary}/
```

## Run

```bash
python train.py --data data --epochs 8   # saves model.pt and prints a classification report
streamlit run app.py                     # upload an MRI and see prediction + heatmap
```

## Ideas to extend

- Add a tumor segmentation model (U-Net) on the BraTS dataset
- Try EfficientNet or a Vision Transformer and compare accuracy
- Add a confusion matrix and per-class recall to the report
- Export to ONNX and deploy on Hugging Face Spaces
