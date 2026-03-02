# Contributing to MindRouter

Welcome to the MindRouter project! We appreciate your interest in contributing. MindRouter is a FastAPI-based LLM inference load balancer that routes requests across heterogeneous model backends (vLLM, Ollama, OpenAI-compatible APIs) with fair-share scheduling, health monitoring, and protocol translation.

We welcome contributions of all kinds:

- **Bug reports** -- help us identify and fix issues
- **Feature requests** -- suggest improvements or new capabilities
- **Code contributions** -- bug fixes, new features, performance improvements
- **Documentation** -- improvements to docs, examples, and guides
- **Tests** -- expanding test coverage and improving test quality

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Reporting Bugs](#reporting-bugs)
7. [Requesting Features](#requesting-features)
8. [Database Migrations](#database-migrations)
9. [Documentation](#documentation)
10. [License](#license)

---

## Getting Started

### Prerequisites

- Python 3.11 or later
- Docker and Docker Compose
- Git

### Setting Up Your Development Environment

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/MindRouter.git
   cd MindRouter
   ```

2. **Copy the environment configuration:**

   ```bash
   cp .env.example .env
   ```

   Review `.env` and adjust any settings for your local environment if needed.

3. **Start infrastructure services:**

   ```bash
   docker compose up -d
   ```

   This starts MariaDB and Redis, which are required for local development.

4. **Install Python dependencies (with dev extras):**

   ```bash
   pip install -e ".[dev]"
   ```

5. **Run database migrations:**

   ```bash
   alembic upgrade head
   ```

6. **Seed development data:**

   ```bash
   python scripts/seed_dev_data.py
   ```

You should now have a fully functional local development environment.

---

## Development Workflow

1. **Create a feature branch from `main`:**

   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

   Use a descriptive branch name (e.g., `fix/ollama-timeout`, `feature/add-quota-endpoint`).

2. **Make your changes.** Write code, add tests, update documentation as needed.

3. **Run the test suite:**

   ```bash
   python -m pytest backend/app/tests/unit/ -v
   ```

4. **Run the linter:**

   ```bash
   ruff check backend/
   ```

5. **Run the formatter:**

   ```bash
   black backend/
   ```

6. **Commit your changes using conventional commit messages:**

   ```
   feat: add weighted round-robin scheduling option
   fix: prevent race condition in backend health check
   docs: update API reference for /v1/models endpoint
   test: add unit tests for OpenAI input translator
   refactor: extract common validation logic into shared module
   chore: update dependency versions
   ```

   Keep commit messages concise but descriptive. Use the body of the commit message for additional context when needed.

7. **Push your branch and open a pull request.**

---

## Code Style

MindRouter uses **ruff** for linting and **black** for code formatting. Both are configured in `pyproject.toml`.

- **Line length:** 100 characters
- **Type hints:** Encouraged for all function signatures. Use `typing` module constructs as needed.
- **Docstrings:** Required for public functions and classes. Use a clear, concise style that describes parameters, return values, and any notable behavior.
- **Imports:** Let ruff handle import sorting. Avoid wildcard imports.

Before submitting a PR, ensure your code passes both tools without errors:

```bash
ruff check backend/
black --check backend/
```

To auto-fix formatting issues:

```bash
black backend/
ruff check --fix backend/
```

---

## Testing

MindRouter has a comprehensive test suite with over 525 unit tests. All new features and bug fixes must include corresponding tests.

### Test Location

- Unit tests: `backend/app/tests/unit/`

### Running Tests

```bash
# Run all unit tests
python -m pytest backend/app/tests/unit/ -v

# Run all unit tests via Makefile
make test-unit

# Run a specific test file
python -m pytest backend/app/tests/unit/test_example.py -v

# Run a specific test
python -m pytest backend/app/tests/unit/test_example.py::test_function_name -v
```

### Test Guidelines

- Use **pytest-asyncio** for async tests (add `@pytest.mark.asyncio` decorator).
- Keep tests focused and independent -- each test should verify one behavior.
- Use fixtures and mocks to isolate the code under test from external dependencies (database, network, etc.).
- Refer to **TESTING.md** at the project root for the full test manifest, test categories, and conventions.

### Test Categories

MindRouter supports several test categories beyond unit tests:

| Category | Command |
|---|---|
| Unit | `make test-unit` |
| Integration | `make test-int` |
| End-to-end | `make test-e2e` |
| Smoke | `make test-smoke` |
| Stress | `make test-stress` |
| Accessibility | `make test-a11y` |

See TESTING.md for details on each category and how to run them.

---

## Pull Request Process

1. **Ensure all tests pass** before opening a PR. The CI pipeline will run the full test suite, but catching issues locally saves time.

2. **Keep PRs focused.** Each pull request should address a single feature, bug fix, or concern. Avoid bundling unrelated changes.

3. **Write a clear PR description** that includes:
   - A summary of what changed and why
   - Links to related GitHub Issues (use `Fixes #123` or `Relates to #456`)
   - Any breaking changes or migration steps
   - Screenshots or logs if the change affects the UI or observable behavior

4. **Respond to review feedback.** Maintainers will review your PR and may request changes. Please address feedback in new commits rather than force-pushing, so the review history is preserved.

5. **Maintainers will merge** approved PRs. We aim to review contributions within a reasonable timeframe. If your PR has been open for a while without feedback, feel free to leave a comment to bump it.

---

## Reporting Bugs

Use [GitHub Issues](https://github.com/ui-insight/MindRouter/issues) to report bugs.

A good bug report includes:

- **Summary:** A clear, concise description of the bug.
- **Steps to reproduce:** Detailed steps to trigger the issue.
- **Expected behavior:** What you expected to happen.
- **Actual behavior:** What actually happened.
- **Environment details:** Python version, OS, Docker version, relevant configuration.
- **Logs:** Include relevant log output if available. **Sanitize any sensitive information** (API keys, credentials, internal hostnames) before sharing.
- **Screenshots:** If the bug is visual, include screenshots.

---

## Requesting Features

Use [GitHub Issues](https://github.com/ui-insight/MindRouter/issues) with the **"feature request"** label to propose new features.

A good feature request includes:

- **Use case:** Describe the problem you are trying to solve or the workflow you want to improve.
- **Proposed solution:** Describe how you envision the feature working.
- **Alternatives considered:** List any alternative approaches you thought about and why you prefer the proposed solution.
- **Additional context:** Any other information, mockups, or examples that help clarify the request.

---

## Database Migrations

MindRouter uses **Alembic** for database schema migrations against MariaDB.

### Creating a Migration

```bash
alembic revision --autogenerate -m "description of the change"
```

Review the generated migration file in `alembic/versions/` to ensure it accurately reflects your intended changes. Auto-generated migrations are not always perfect.

### Testing Migrations

Always test both the upgrade and downgrade paths:

```bash
alembic upgrade head
alembic downgrade -1
alembic upgrade head
```

### Important Notes

- **MariaDB DDL is non-transactional.** If a migration fails partway through, the database will be left in a partially migrated state that must be cleaned up manually. Test your migrations thoroughly before merging.
- **Foreign key constraints and indexes:** MariaDB will not drop an index that backs a foreign key constraint (error 1553). Always drop foreign key constraints before dropping their backing indexes in migration scripts.
- **Include both upgrade and downgrade.** Every migration must have a working `downgrade()` function.

---

## Documentation

MindRouter maintains documentation in several locations:

| Location | Purpose |
|---|---|
| `docs/index.md` | Main project documentation |
| `docs/architecture.md` | System architecture and design |
| `backend/app/dashboard/templates/public/documentation.html` | In-app documentation (served by the dashboard) |
| `TESTING.md` | Test manifest and testing conventions |

When adding or changing features:

- Update the relevant documentation files to reflect the new behavior.
- Keep `docs/` and the in-app documentation in sync so users get consistent information regardless of where they look.
- Use clear, concise language. Avoid jargon where possible.

---

## License

MindRouter is licensed under the [Apache License 2.0](LICENSE).

By submitting a contribution to this project, you agree that your contribution will be licensed under the same Apache 2.0 license. You represent that you have the right to license your contribution under these terms.

---

Thank you for contributing to MindRouter!
