.PHONY: install demo server test docker-build docker-up clean

install:
	pip install -r requirements.txt
	pip install -e .

demo:
	python icm_demo.py

server:
	python applications/icm_server.py

test:
	pytest -v --tb=short

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

cli:
	python applications/cli_chat.py

colab:
	@echo "Open https://colab.research.google.com/github/varshinicb1/hyper-ssm-ultimate/blob/main/icm_demo.ipynb"
