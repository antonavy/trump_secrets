FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed (e.g. for building some python packages)
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy configuration files
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create a virtual environment (install directly in system python)
RUN poetry config virtualenvs.create false

# Install dependencies (no dev dependencies, no interaction)
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY . .

# Run the application
CMD ["python", "trump_feed_parser.py"]
