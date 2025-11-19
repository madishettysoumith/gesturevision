import argparse, os, time, cv2
from src.segmentation import segment_hand, to_model_input

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output root, e.g., dataset/dynamic")
    ap.add_argument("--labels", nargs="+", required=True, help="Class names")
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--sequences-per-class", type=int, default=30)
    ap.add_argument("--img-size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press 'q' to quit at any time.")
    for cls in args.labels:
        cls_dir = os.path.join(args.out, cls)
        os.makedirs(cls_dir, exist_ok=True)
        print(f"Collecting dynamic class '{cls}'...")
        for s in range(args.sequences_per_class):
            seq_dir = os.path.join(cls_dir, f"seq_{int(time.time()*1000)}")
            os.makedirs(seq_dir, exist_ok=True)
            print(f"  Sequence {s+1}/{args.sequences_per_class}")
            for i in range(args.seq_len):
                ok, frame = cap.read()
                if not ok: break
                frame = cv2.flip(frame, 1)
                hand, _ = segment_hand(frame)
                arr = (to_model_input(hand, args.img_size) * 255).astype("uint8")
                cv2.imwrite(os.path.join(seq_dir, f"frame_{i+1:04d}.jpg"), arr)
                cv2.imshow("Capture Dynamic", frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); return
            time.sleep(0.2)
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
if __name__ == "__main__":
    main()
