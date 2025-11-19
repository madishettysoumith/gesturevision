import argparse, os
import tensorflow as tf
from src.preprocess import make_dynamic_tfds
from src.cnn_lstm_model import build_cnn_lstm
from src.utils import ensure_dirs, save_label_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    ensure_dirs()
    ds, classes = make_dynamic_tfds(args.data, seq_len=args.seq_len, img_size=args.img_size, batch_size=args.batch)
    model = build_cnn_lstm(seq_len=args.seq_len, img_size=args.img_size, channels=1, num_classes=len(classes))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    ckpt = tf.keras.callbacks.ModelCheckpoint("models/cnn_lstm_best.h5", monitor="accuracy", save_best_only=True, verbose=1)
    es = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=8, restore_best_weights=True)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)

    model.fit(ds, epochs=args.epochs, callbacks=[ckpt, es, rlrop], steps_per_epoch=100)

    model.save("models/cnn_lstm_final.h5")
    save_label_map(classes)
    print("Saved model to models/cnn_lstm_final.h5 and label_map.json")

if __name__ == "__main__":
    main()
