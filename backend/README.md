# LLM Evaluation Framework

A comprehensive framework for evaluating LLM-based micro-agents performance across various metrics.

## Features

- **Prompt Evaluation**: Create or select prompts and upload datasets for evaluation
- **Micro-Agent-Level Evaluation**: Generate detailed performance reports for specific micro-agents
- **Prompt Comparison**: Compare two prompts side-by-side
- **Advanced Evaluation Techniques**: Support for RAG, chatbots, and AI-generated text
- **Integration with Evaluation Tools**: RAGAS, DeepEval, and custom scoring
- **Experiment Tracking & Dataset Versioning**: Using MLflow or W&B
- **User & Access Control**: OIDC-based authentication with RBAC
- **Scalability & Performance Optimization**: Caching and background task processing
- **User-Friendly Dashboard**: Interactive visualization with HTMX
- **FastAPI-Based APIs**: High-performance API development

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-evaluation-framework.git
cd llm-evaluation-framework

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Run with Docker
docker-compose up -d
```

## Usage

See the [documentation](./docs) for detailed usage instructions.

## Development

See [Development Guide](./docs/guides/development.md) for information on contributing to this project.

## License

MIT
