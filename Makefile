.PHONY: setup fetch export-onnx demo validate benchmark docker-build docker-run clean help

UV       ?= uv
PLATFORM ?= linux/arm64
IMAGE    ?= aegisvision:latest

help:
	@echo "Targets:"
	@echo "  setup         Install deps via uv"
	@echo "  fetch         Download the pinned traffic sample video"
	@echo "  export-onnx   Export YOLO and RT-DETR weights to ONNX"
	@echo "  demo          Run the live pipeline on the sample"
	@echo "  validate      Phase 2 validation gate + CoreML EP compute-unit sweep"
	@echo "  benchmark     Run the full model x backend matrix"
	@echo "  docker-build  Build the linux/arm64 ONNX image"
	@echo "  docker-run    Run the container with data/ and logs/ mounted"
	@echo "  clean         Remove venv and runtime artifacts"

setup:
	$(UV) sync

fetch:
	$(UV) run python scripts/fetch_sample.py

export-onnx:
	$(UV) run python scripts/export_onnx.py

demo:
	$(UV) run python -m aegisvision.pipeline

validate:
	$(UV) run python scripts/phase2_validate.py

benchmark:
	$(UV) run python scripts/benchmark.py

docker-build:
	docker buildx build --platform $(PLATFORM) -t $(IMAGE) --load .

docker-run:
	docker run --rm --platform $(PLATFORM) \
	  -v $(PWD)/data:/app/data \
	  -v $(PWD)/logs:/app/logs \
	  $(IMAGE)

clean:
	rm -rf .venv data/samples/*.mp4 logs/*.jsonl results/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
