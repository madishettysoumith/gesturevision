
import argparse, os, time, cv2
from src.segmentation import segment_hand, to_model_input

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output root, e.g., dataset/static")
    ap.add_argument("--labels", nargs="+", required=True, help="Class names")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--samples-per-class", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press 'q' to quit at any time.")
    for cls in args.labels:
        cls_dir = os.path.join(args.out, cls)
        os.makedirs(cls_dir, exist_ok=True)
        count = 0
        print(f"Collecting class '{cls}'...")

        while count < args.samples_per_class:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame, 1)
            hand, mask = segment_hand(frame)
            disp = frame.copy()
            cv2.putText(disp, f"Class: {cls}  ({count}/{args.samples_per_class})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Capture Static", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

            # auto-save every frame
            ts = int(time.time()*1000)
            out_path = os.path.join(cls_dir, f"{cls}_{ts}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor((to_model_input(hand, args.img_size)*255).astype("uint8"), cv2.COLOR_GRAY2BGR))
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
