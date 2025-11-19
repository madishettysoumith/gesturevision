import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE

def _norm(x,y):
    return (tf.cast(x, tf.float32)/255.0, y)

def _augment_batch(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.clip_by_value(x + tf.random.uniform((), -0.15, 0.15), 0.0, 1.0)
    x = tf.image.random_contrast(x, 0.85, 1.15)
    x.set_shape([None, None, None, 1])
    return x, y

def make_static_ds(root, img_size=128, batch_size=32, val_split=0.1, augment=True):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        root, labels="inferred", label_mode="categorical",
        color_mode="grayscale", image_size=(img_size, img_size),
        validation_split=val_split, subset="training", seed=42, batch_size=batch_size
    )
    val = tf.keras.preprocessing.image_dataset_from_directory(
        root, labels="inferred", label_mode="categorical",
        color_mode="grayscale", image_size=(img_size, img_size),
        validation_split=val_split, subset="validation", seed=42, batch_size=batch_size
    )
    class_names = ds.class_names
    ds = ds.map(_norm, num_parallel_calls=AUTOTUNE)
    val = val.map(_norm, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE), val.prefetch(AUTOTUNE), class_names

import os, glob, random, numpy as np

def _load_frame(path, img_size):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (img_size, img_size))
    im = (im.astype("float32")/255.0)[..., None]
    return im

def list_dynamic_sequences(root):
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])
    items = []
    for ci, c in enumerate(classes):
        cdir = os.path.join(root, c)
        seqs = sorted([s for s in os.listdir(cdir) if os.path.isdir(os.path.join(cdir,s))])
        for s in seqs:
            frames = sorted(glob.glob(os.path.join(cdir, s, "frame_*.jpg")))
            if frames:
                items.append((frames, ci))
    return items, classes

def dynamic_generator(root, seq_len=16, img_size=128):
    items, classes = list_dynamic_sequences(root)
    while True:
        random.shuffle(items)
        for frames, ci in items:
            if len(frames) >= seq_len:
                import random as _r
                start = _r.randint(0, len(frames)-seq_len)
                sel = frames[start:start+seq_len]
            else:
                sel = frames + [frames[-1]]*(seq_len-len(frames))
            arr = np.stack([_load_frame(p, img_size) for p in sel], axis=0)
            y = np.zeros((len(classes),), dtype=np.float32); y[ci] = 1.0
            yield arr, y

def make_dynamic_tfds(root, seq_len=16, img_size=128, batch_size=8):
    items, classes = list_dynamic_sequences(root)
    output_sig = (
        tf.TensorSpec(shape=(seq_len, img_size, img_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(len(classes),), dtype=tf.float32)
    )
    ds = tf.data.Dataset.from_generator(lambda: dynamic_generator(root, seq_len, img_size), output_signature=output_sig)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, classes
