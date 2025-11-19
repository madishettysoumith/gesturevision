
# GestureVision DL — Deep Learning Hand Gesture Recognition (CNN + CNN-LSTM)

This project implements the **research methodology** you provided:
- **Preprocessing**: skin-color segmentation (HSV) + optional background subtraction (MOG2), resize **128×128**, normalization
- **Augmentation**: rotation, horizontal flip, random brightness/contrast
- **Models**:
  - **CNN** for static gesture images
  - **CNN-LSTM** (TimeDistributed CNN + LSTM) for dynamic sequences
- **Training**: Adam (`lr=0.001`), categorical cross-entropy, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Real-time inference**: webcam (static or dynamic mode with sliding window)
- **VS Code ready**: launch configs, tasks, Make targets

> Framework: **TensorFlow 2.14+**, OpenCV, NumPy.


## Folder Structure
```
gesturevision_dl/
├─ .vscode/
│  ├─ launch.json
│  └─ settings.json
├─ config/
│  └─ settings.yaml
├─ dataset/
│  ├─ static/CLASS_NAME/*.jpg
│  └─ dynamic/CLASS_NAME/SEQUENCE_ID/frame_0001.jpg ...
├─ models/
├─ src/
│  ├─ segmentation.py
│  ├─ preprocess.py
│  ├─ cnn_model.py
│  ├─ cnn_lstm_model.py
│  ├─ train_cnn.py
│  ├─ train_cnn_lstm.py
│  ├─ infer_realtime.py
│  ├─ capture_static.py
│  ├─ capture_dynamic.py
│  └─ utils.py
├─ requirements.txt
├─ Makefile
└─ README.md
```

## Quickstart
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 1) Collect data (optional demos)
Static:
```bash
python -m src.capture_static --out dataset/static --labels open_palm fist peace ok
```
Dynamic (records short sequences per class):
```bash
python -m src.capture_dynamic --out dataset/dynamic --labels open_palm fist peace ok --seq-len 16
```

### 2) Train
CNN (static images):
```bash
python -m src.train_cnn --data dataset/static --epochs 50
```
CNN-LSTM (dynamic sequences):
```bash
python -m src.train_cnn_lstm --data dataset/dynamic --seq-len 16 --epochs 50
```

### 3) Run real-time inference
Static (CNN):
```bash
python -m src.infer_realtime --mode static --img-size 128
```
Dynamic (CNN-LSTM):
```bash
python -m src.infer_realtime --mode dynamic --seq-len 16 --img-size 128
```

### One-Cell Commands (copy/paste)
```bash
python -m venv .venv && . .venv/Scripts/activate 2>nul || source .venv/bin/activate
pip install -r requirements.txt
python -m src.capture_static --out dataset/static --labels open_palm fist peace ok
python -m src.train_cnn --data dataset/static --epochs 20
python -m src.infer_realtime --mode static --img-size 128
```

## Notes
- Change labels/classes in the commands or in `config/settings.yaml`.
- Models and `label_map.json` save in `models/`.
- Press **q** to quit any OpenCV window.
"# gesturevision" 
