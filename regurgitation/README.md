# Regurgitation detection model
This repository contains code for **Automated Detection and Quantification of Doppler Echocardiographic Metrics of Valvular Heart Diseases** implemented in PyTorch.

## Requirements
- Python 3.6
- PyTorch 1.1

## Training
An example of dataset is shown in prepare_data.py

Train MR dectection model
```python3 trainer_unet_rcnn.py --cfg ./mr_a4c.yaml --cuda 0```

Train AR dectection model
```python3 trainer_unet_rcnn.py --cfg ./ar_plax.yaml --cuda 0```

## Inference
Use inference.mr_detect and inference.ar_detect to predict regurgitations.