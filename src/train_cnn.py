
import argparse, os, json
import tensorflow as tf
from src.preprocess import make_static_ds
from src.cnn_model import build_cnn
from src.utils import ensure_dirs, save_label_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to dataset/static root")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--val-split", type=float, default=0.1)
    args = ap.parse_args()

    ensure_dirs()

    train, val, classes = make_static_ds(args.data, img_size=args.img_size, batch_size=args.batch, val_split=args.val_split, augment=True)

    model = build_cnn((args.img_size, args.img_size, 1), num_classes=len(classes))
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    ckpt = tf.keras.callbacks.ModelCheckpoint("models/cnn_best.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
    es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    model.fit(train, validation_data=val, epochs=args.epochs, callbacks=[ckpt, es, rlrop])

    model.save("models/cnn_final.h5")
    save_label_map(classes)

    print("Saved model to models/cnn_final.h5 and label_map.json")

if __name__ == "__main__":
    main()
