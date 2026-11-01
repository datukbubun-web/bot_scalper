.PHONY: install install-dev test lint fmt smoke doctor data bias train-xau train-tcn train-lstm train-transformer train-ppo train-sac export calibrate backtest backtest-oos paper live push

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
	python -m pxscalper.cli data --symbols "XAU/USD,XAG/USD" --bars 5M --limit 25000

bias:             ## inspect causal Hybrid Intraday Bias + Scalper state
	python -m pxscalper.cli bias --symbol XAU/USD --bars 5M --htf 15M

train-xau:        ## train, calibrate, export & OOS-verify all XAU/USD Hybrid + single-TF models
	python scripts/training/train_xau.py --bars ALL

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

calibrate:        ## fit probability calibrator (temperature) on validation slice
	python scripts/calibrate.py

backtest:
	python -m pxscalper.cli backtest --config configs/live.hybrid.yaml

backtest-oos:
	python -m pxscalper.cli backtest --config configs/live.hybrid.yaml --oos-frac 0.40

paper:
	PX_LIVE_TRADING=0 python -m pxscalper.cli trade --config configs/live.hybrid.yaml

live:
	PX_LIVE_TRADING=1 python -m pxscalper.cli trade --config configs/live.hybrid.yaml --i-understand-the-risk

push:             ## push ke github.com/datukbubun-web/bot_scalper (butuh GITHUB_TOKEN atau kredensial git)
	bash scripts/push_to_github.sh
