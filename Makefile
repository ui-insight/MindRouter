.PHONY: help dev test test-unit test-int test-e2e test-smoke test-stress test-matrix test-thinking test-a11y test-sidecar test-all lint format migrate seed docker-up docker-down clean

# Default target
help:
	@echo "MindRouter2 Development Commands"
	@echo "================================"
	@echo ""
	@echo "Development:"
	@echo "  make dev           - Start development server with hot reload"
	@echo "  make install       - Install dependencies"
	@echo "  make install-dev   - Install dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run pytest suite (unit + integration)"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-int      - Run integration tests (live backends)"
	@echo "  make test-e2e      - Run end-to-end chat tests (live stack)"
	@echo "  make test-smoke    - Run API smoke tests (live stack)"
	@echo "  make test-stress   - Run stress/load test (120s, live stack)"
	@echo "  make test-matrix   - Run structured output matrix tests (live stack)"
	@echo "  make test-thinking - Run structured output + thinking compliance tests (live stack)"
	@echo "  make test-a11y     - Run accessibility tests (WCAG 2.1)"
	@echo "  make test-sidecar  - Run GPU sidecar tests"
	@echo "  make test-all      - Run unit + integration + sidecar tests"
	@echo "  make coverage      - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code with black"
	@echo "  make typecheck     - Run mypy type checking"
	@echo ""
	@echo "Database:"
	@echo "  make migrate       - Run database migrations"
	@echo "  make migrate-new   - Create new migration (NAME=migration_name)"
	@echo "  make seed          - Seed development data"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     - Start docker compose stack"
	@echo "  make docker-down   - Stop docker compose stack"
	@echo "  make docker-build  - Build docker images"
	@echo "  make docker-logs   - View docker logs"
	@echo "  make docker-dev    - Start with dev profile (includes mocks)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make clean-all     - Remove all generated files including DB"

# Development
dev:
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest backend/app/tests -v

test-unit:
	pytest backend/app/tests/unit -v

test-int:
	pytest backend/app/tests/integration -v

test-e2e:
	pytest backend/app/tests/e2e -v

test-smoke:
	@echo "Running API smoke tests against live deployment..."
	python test.py --api-key $(API_KEY) --base-url $(or $(BASE_URL),http://localhost:8000) --admin-key $(or $(ADMIN_KEY),$(API_KEY)) --timeout 180

test-stress:
	@echo "Running stress test (120s)..."
	python stress.py --api-key $(API_KEY) --base-url $(or $(BASE_URL),http://localhost:8000) --duration $(or $(DURATION),120) --concurrency $(or $(CONCURRENCY),10)

test-matrix:
	@echo "Running structured output matrix tests..."
	MINDROUTER_API_KEY=$(API_KEY) MINDROUTER_BASE_URL=$(or $(BASE_URL),http://localhost:8000) pytest backend/app/tests/integration/test_structured_output_matrix.py -v --tb=short

test-thinking:
	@echo "Running structured output + thinking compliance tests..."
	MINDROUTER_API_KEY=$(API_KEY) python tests/test_structured_thinking.py

test-a11y:
	pytest backend/app/tests/unit/test_accessibility.py -v

test-sidecar:
	pytest sidecar/tests/ -v

test-all:
	@echo "Running all automated tests..."
	pytest backend/app/tests -v
	pytest sidecar/tests/ -v

coverage:
	pytest backend/app/tests --cov=backend/app --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# Code Quality
lint:
	ruff check backend/
	ruff format --check backend/

format:
	ruff format backend/
	ruff check --fix backend/

typecheck:
	mypy backend/app

# Database
migrate:
	alembic upgrade head

migrate-new:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make migrate-new NAME=migration_name"; \
		exit 1; \
	fi
	alembic revision --autogenerate -m "$(NAME)"

migrate-down:
	alembic downgrade -1

seed:
	python scripts/seed_dev_data.py

# Docker
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-build:
	docker compose build

docker-logs:
	docker compose logs -f

docker-dev:
	docker compose --profile dev up -d

docker-shell:
	docker compose exec app bash

docker-migrate:
	docker compose exec app alembic upgrade head

docker-seed:
	docker compose exec app python scripts/seed_dev_data.py

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true

clean-all: clean
	docker compose down -v 2>/dev/null || true
	rm -rf .venv 2>/dev/null || true

# Demo
demo:
	python scripts/demo_fairness.py
