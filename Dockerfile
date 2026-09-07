FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir .

ENV AETHERGRAPH_CONFIG_DIR=/app/configs
EXPOSE 8080
CMD ["aethergraph", "serve", "--host", "0.0.0.0", "--port", "8080"]
