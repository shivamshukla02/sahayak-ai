.PHONY: help install embed run pull-model docker-up docker-down clean

# Colors for terminal styling
YELLOW = \033[0;33m
GREEN = \033[0;32m
NC = \033[0m

help:
	@echo "--------------------------------------------------------"
	@echo "      SAHAYAK AI SOS LOCAL SERVICE MAKE WORKFLOW"
	@echo "--------------------------------------------------------"
	@echo "  make install     : Set up virtual env & install Python dependencies"
	@echo "  make pull-model   : Pull local Gemma 4 model in Ollama"
	@echo "  make embed       : Index NDMA and first-aid manuals into ChromaDB"
	@echo "  make run         : Start the local FastAPI backend (Uvicorn)"
	@echo "  make docker-up   : Spin up FastAPI and Ollama via Docker Compose"
	@echo "  make docker-down : Tear down Docker Compose containers"
	@echo "  make clean       : Clear Python cache, temp stores, and database"
	@echo "--------------------------------------------------------"

install:
	@echo -e "$(YELLOW)[1/2] Creating Python virtual environment...$(NC)"
	python -m venv venv
	@echo -e "$(GREEN)Virtual environment created successfully.$(NC)"
	@echo -e "$(YELLOW)[2/2] Installing requirements...$(NC)"
	./venv/Scripts/pip install -r backend/requirements.txt
	@echo -e "$(GREEN)Dependencies successfully installed inside ./venv/$(NC)"

pull-model:
	@echo -e "$(YELLOW)Pulling Gemma 4 (8B-instruct-q4) local model via Ollama...$(NC)"
	ollama pull gemma4:8b-instruct-q4_K_M
	@echo -e "$(GREEN)Model pulled and ready. Ensure Ollama service is running!$(NC)"

embed:
	@echo -e "$(YELLOW)Starting offline document indexing into ChromaDB...$(NC)"
	./venv/Scripts/python backend/embed_docs.py
	@echo -e "$(GREEN)ChromaDB vector embedding generation complete.$(NC)"

run:
	@echo -e "$(YELLOW)Starting FastAPI server at http://localhost:8000 (Hot reloading active)...$(NC)"
	./venv/Scripts/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	@echo -e "$(YELLOW)Building and spinning up Docker containers...$(NC)"
	docker-compose up --build -d
	@echo -e "$(GREEN)Docker containers active. Backend on port 8000, Ollama on port 11434.$(NC)"

docker-down:
	@echo -e "$(YELLOW)Tearing down Docker containers...$(NC)"
	docker-compose down
	@echo -e "$(GREEN)Containers stopped and removed.$(NC)"

clean:
	@echo -e "$(YELLOW)Cleaning caches and temporary databases...$(NC)"
	rm -rf backend/__pycache__
	rm -rf backend/chroma_db
	rm -rf .pytest_cache
	@echo -e "$(GREEN)Clean complete.$(NC)"
