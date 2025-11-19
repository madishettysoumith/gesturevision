
.PHONY: venv install capture_static capture_dynamic train_cnn train_cnnlstm run_static run_dynamic

venv:
	python -m venv .venv

install:
	pip install -r requirements.txt

capture_static:
	python -m src.capture_static --out dataset/static --labels open_palm fist peace ok

capture_dynamic:
	python -m src.capture_dynamic --out dataset/dynamic --labels open_palm fist peace ok --seq-len 16

train_cnn:
	python -m src.train_cnn --data dataset/static --epochs 50

train_cnnlstm:
	python -m src.train_cnn_lstm --data dataset/dynamic --seq-len 16 --epochs 50

run_static:
	python -m src.infer_realtime --mode static --img-size 128

run_dynamic:
	python -m src.infer_realtime --mode dynamic --seq-len 16 --img-size 128
