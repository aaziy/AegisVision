.PHONY: setup fetch demo benchmark docker-build docker-run clean help

UV       ?= uv
PLATFORM ?= linux/arm64
IMAGE    ?= aegisvision:latest

help:
	@echo "Targets:"
	@echo "  setup         Install deps via uv"
	@echo "  fetch         Download the pinned traffic sample video"
	@echo "  demo          Run the live pipeline on the sample"
	@echo "  benchmark     Run the full model x backend matrix"
	@echo "  docker-build  Build the linux/arm64 ONNX image"
	@echo "  docker-run    Run the container with data/ and logs/ mounted"
	@echo "  clean         Remove venv and runtime artifacts"

setup:
	$(UV) sync

fetch:
	$(UV) run python scripts/fetch_sample.py

demo:
	$(UV) run python -m aegisvision.pipeline

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
