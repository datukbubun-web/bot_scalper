.PHONY: install install-dev test lint fmt smoke doctor data train-tcn train-lstm train-transformer train-ppo train-sac export calibrate backtest paper live push

install:
	pip install -e .

install-dev:
	pip install -e ".[train,test]"

test:
	pytest -q

lint:
	ruff check src tests scripts

fmt:
	ruff format src tests scripts

smoke:            ## end-to-end offline smoke test (no API keys, no network)
	python -m pxscalper.cli smoke

doctor:           ## verify credentials, clock skew, symbol metadata
	python -m pxscalper.cli doctor

data:             ## download historical bars into data/
	python -m pxscalper.cli data --bars 1M --limit 5000

train-tcn:
	python scripts/training/train_sl.py --arch tcn --symbol EUR/USD --bars 1M

train-lstm:
	python scripts/training/train_sl.py --arch lstm --symbol EUR/USD --bars 1M

train-transformer:
	python scripts/training/train_sl.py --arch transformer --symbol EUR/USD --bars 1M

train-ppo:
	python scripts/training/train_rl.py --algo ppo

train-sac:
	python scripts/training/train_rl.py --algo sac

export:           ## PyTorch -> TorchScript + ONNX (+ latency benchmark)
	python scripts/export_models.py --all

calibrate:        ## fit probability calibrator (temperature/isotonic) on validation slice
	python scripts/calibrate.py

backtest:
	python -m pxscalper.cli backtest --config configs/live.example.yaml

paper:
	PX_LIVE_TRADING=0 python -m pxscalper.cli trade --config configs/live.example.yaml

live:
	PX_LIVE_TRADING=1 python -m pxscalper.cli trade --config configs/live.yaml --i-understand-the-risk

push:             ## push ke github.com/datukbubun-web/bot_scalper (butuh GITHUB_TOKEN atau kredensial git)
	bash scripts/push_to_github.sh
