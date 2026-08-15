.PHONY: ci backend-ci frontend-ci install install-dev test lint typecheck clean

# ── Install ──
install-dev:
	pip install -e ".[dev]"
	cd frontend && npm ci

# ── Lint ──
lint:
	ruff format --check backend desktop tests
	ruff check backend desktop tests
	cd frontend && npm run lint && npm run format:check

# ── Typecheck ──
typecheck:
	mypy backend desktop
	cd frontend && npm run check

# ── Test ──
test:
	pytest tests/backend -m "not ml_regression and not addon and not e2e"
	cd frontend && npm test

# ── Full CI parity ──
ci: lint typecheck test

# ── Clean ──
clean:
	rm -rf build/ dist/ .mypy_cache/ .ruff_cache/ .pytest_cache/ htmlcov/ coverage.xml
	rm -rf frontend/dist/ frontend/node_modules/ frontend/.svelte-kit/
	find . -type d -name __pycache__ -exec rm -rf {} +
