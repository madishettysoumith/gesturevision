
import argparse, time, os
import numpy as np
import cv2
import tensorflow as tf
from collections import deque
from src.segmentation import segment_hand, to_model_input
from src.utils import load_label_map

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, choices=["static","dynamic"], default="static")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=16)
    args = ap.parse_args()

    labels = load_label_map()
    if args.mode == "static":
        model_path = "models/cnn_final.h5"
    else:
        model_path = "models/cnn_lstm_final.h5"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at {model_path}. Train first.")

    model = tf.keras.models.load_model(model_path, compile=False)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window = deque(maxlen=args.seq_len)

    fps_avg = 0.0
    alpha = 0.1

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        hand, mask = segment_hand(frame)
        inp = to_model_input(hand, img_size=args.img_size)

        if args.mode == "static":
            x = np.expand_dims(inp, axis=0)  # (1,H,W,1)
            pred = model.predict(x, verbose=0)[0]
        else:
            window.append(inp)
            if len(window) < args.seq_len:
                pred = np.zeros((len(labels),), dtype=np.float32)
            else:
                seq = np.stack(list(window), axis=0)  # (T,H,W,1)
                x = np.expand_dims(seq, axis=0)      # (1,T,H,W,1)
                pred = model.predict(x, verbose=0)[0]

        idx = int(np.argmax(pred)) if pred.sum()>0 else -1
        prob = float(np.max(pred)) if pred.sum()>0 else 0.0
        text = "No prediction" if idx==-1 else f"{labels[idx]} ({prob:.2f})"

        # Display
        cv2.putText(frame, f"Mode: {args.mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Pred: {text}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("GestureVision DL", frame)

        # FPS
        dt = time.time() - t0
        fps = 1.0/max(dt,1e-6)
        fps_avg = (1-alpha)*fps_avg + alpha*fps
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
