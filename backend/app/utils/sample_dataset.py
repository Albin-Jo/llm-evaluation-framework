# File: backend/app/utils/sample_dataset.py
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import uuid

from backend.app.core.config import settings
from backend.app.db.models.orm.models import DatasetType, User

# Configure logging
logger = logging.getLogger(__name__)


class SampleDatasetGenerator:
    """Utility class for generating sample datasets for testing evaluation."""

    @staticmethod
    def generate_question_answer_dataset(
            name: str = "Sample QA Dataset",
            num_samples: int = 10,
            domain: str = "general",
            include_contexts: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a sample question-answer dataset.

        Args:
            name: Dataset name
            num_samples: Number of samples to generate
            domain: Domain for questions
            include_contexts: Whether to include contexts

        Returns:
            Tuple[str, Dict[str, Any]]: File path and metadata
        """
        samples = []

        # Define sample data based on domain
        if domain == "general":
            qa_pairs = [
                {
                    "query": "What is the capital of France?",
                    "context": "France is a country in Western Europe with a population of 67 million. Its capital city is Paris, which is known as the City of Light.",
                    "ground_truth": "The capital of France is Paris."
                },
                {
                    "query": "Who wrote the novel 'Pride and Prejudice'?",
                    "context": "Pride and Prejudice is a romantic novel by Jane Austen, published in 1813. It follows the character development of Elizabeth Bennet.",
                    "ground_truth": "Jane Austen wrote the novel 'Pride and Prejudice'."
                },
                {
                    "query": "What is the tallest mountain in the world?",
                    "context": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. It stands at an elevation of 29,032 feet (8,849 meters).",
                    "ground_truth": "The tallest mountain in the world is Mount Everest."
                },
                {
                    "query": "What is the chemical symbol for gold?",
                    "context": "Gold is a chemical element with the symbol Au (from Latin: aurum) and atomic number 79. It is a bright, slightly orange-yellow, dense, soft, malleable, and ductile metal in its pure form.",
                    "ground_truth": "The chemical symbol for gold is Au."
                },
                {
                    "query": "Who painted the Mona Lisa?",
                    "context": "The Mona Lisa is a half-length portrait painting by Italian artist Leonardo da Vinci. Considered an archetypal masterpiece of the Italian Renaissance, it is often described as 'the best known, the most visited, the most written about, the most sung about, the most parodied work of art in the world'.",
                    "ground_truth": "Leonardo da Vinci painted the Mona Lisa."
                },
                {
                    "query": "What is the largest ocean on Earth?",
                    "context": "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south and is bounded by the continents of Asia and Australia in the west and the Americas in the east.",
                    "ground_truth": "The largest ocean on Earth is the Pacific Ocean."
                },
                {
                    "query": "What is the capital of Japan?",
                    "context": "Japan is an island country in East Asia. It is situated in the northwest Pacific Ocean and is bordered by the Sea of Japan to the west. Tokyo is Japan's capital and largest city.",
                    "ground_truth": "The capital of Japan is Tokyo."
                },
                {
                    "query": "Who is the author of 'To Kill a Mockingbird'?",
                    "context": "To Kill a Mockingbird is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
                    "ground_truth": "Harper Lee is the author of 'To Kill a Mockingbird'."
                },
                {
                    "query": "What is the boiling point of water?",
                    "context": "Water is a transparent, tasteless, odorless, and nearly colorless chemical substance. Its chemical formula is H2O, meaning that each of its molecules contains one oxygen and two hydrogen atoms. The boiling point of water at standard atmospheric pressure (1 atmosphere = 1013.25 mbar) is 100 degrees Celsius or 212 degrees Fahrenheit.",
                    "ground_truth": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure."
                },
                {
                    "query": "Which planet is known as the Red Planet?",
                    "context": "Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System, being larger than only Mercury. In English, Mars carries the name of the Roman god of war and is often referred to as the 'Red Planet' because the iron oxide prevalent on its surface gives it a reddish appearance.",
                    "ground_truth": "Mars is known as the Red Planet."
                },
            ]
        elif domain == "programming":
            qa_pairs = [
                {
                    "query": "What is Python?",
                    "context": "Python is an interpreted high-level general-purpose programming language. Its design philosophy emphasizes code readability with its use of significant indentation. Its language constructs as well as its object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.",
                    "ground_truth": "Python is an interpreted high-level general-purpose programming language known for its readability and simple syntax."
                },
                {
                    "query": "What is a REST API?",
                    "context": "REST (Representational State Transfer) is a software architectural style that defines a set of constraints to be used for creating Web services. Web services that conform to the REST architectural style, called RESTful Web services, provide interoperability between computer systems on the internet. RESTful Web services allow the requesting systems to access and manipulate textual representations of Web resources by using a uniform and predefined set of stateless operations.",
                    "ground_truth": "A REST API is an architectural style for designing networked applications that uses HTTP requests to access and manipulate data."
                },
                {
                    "query": "What is the difference between SQL and NoSQL databases?",
                    "context": "SQL databases are primarily called Relational Databases (RDBMS); whereas NoSQL databases are primarily called non-relational or distributed databases. SQL databases are table-based, while NoSQL databases are document, key-value, graph, or wide-column stores. SQL databases are vertically scalable, while NoSQL databases are horizontally scalable. SQL databases have a predefined schema, whereas NoSQL databases have a dynamic schema for unstructured data.",
                    "ground_truth": "SQL databases are relational, table-based, vertically scalable, and have predefined schemas, while NoSQL databases are non-relational, document/key-value/graph/wide-column based, horizontally scalable, and have dynamic schemas."
                },
                {
                    "query": "What is Docker?",
                    "context": "Docker is a set of platform as a service (PaaS) products that use OS-level virtualization to deliver software in packages called containers. Containers are isolated from one another and bundle their own software, libraries and configuration files; they can communicate with each other through well-defined channels. All containers are run by a single operating-system kernel and are thus more lightweight than virtual machines.",
                    "ground_truth": "Docker is a platform that uses containers to package applications and their dependencies for easy deployment and scaling."
                },
                {
                    "query": "What is Git?",
                    "context": "Git is a distributed version-control system for tracking changes in source code during software development. It is designed for coordinating work among programmers, but it can be used to track changes in any set of files. Its goals include speed, data integrity, and support for distributed, non-linear workflows.",
                    "ground_truth": "Git is a distributed version control system for tracking changes in source code during software development."
                },
            ]
        else:
            # Default to general QA pairs if domain is not recognized
            qa_pairs = [
                {
                    "query": "What is machine learning?",
                    "context": "Machine learning is a field of inquiry devoted to understanding and building methods that 'learn' – that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.",
                    "ground_truth": "Machine learning is a field of AI that enables systems to learn from data without being explicitly programmed."
                },
            ]

        # Generate samples by repeating from the qa_pairs list if needed
        for i in range(num_samples):
            pair = qa_pairs[i % len(qa_pairs)]

            sample = {
                "id": str(i + 1),
                "query": pair["query"],
                "ground_truth": pair["ground_truth"]
            }

            if include_contexts:
                sample["context"] = pair["context"]

            samples.append(sample)

        # Create a unique filename
        filename = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}.json"
        storage_path = Path(os.path.abspath(settings.STORAGE_LOCAL_PATH)) / "datasets"
        os.makedirs(storage_path, exist_ok=True)

        # Write dataset to file
        file_path = storage_path / filename
        with open(file_path, 'w') as f:
            json.dump(samples, f, indent=2)

        logger.info(f"Generated sample dataset with {len(samples)} samples: {file_path}")

        # Create metadata
        metadata = {
            "name": name,
            "description": f"Sample {domain} QA dataset with {num_samples} questions",
            "domain": domain,
            "type": DatasetType.QUESTION_ANSWER.value,
            "row_count": len(samples),
            "version": "1.0.0",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "query": {"type": "string"},
                        "ground_truth": {"type": "string"}
                    }
                }
            }
        }

        if include_contexts:
            metadata["schema"]["items"]["properties"]["context"] = {"type": "string"}

        return str(file_path), metadata

    @staticmethod
    async def create_sample_dataset_in_db(
            db_session,
            owner: User,
            name: str = "Sample QA Dataset",
            num_samples: int = 10,
            domain: str = "general",
            include_contexts: bool = True
    ) -> 'Dataset':
        """
        Create a sample dataset and store it in the database.

        Args:
            db_session: Database session
            owner: Dataset owner
            name: Dataset name
            num_samples: Number of samples to generate
            domain: Domain for questions
            include_contexts: Whether to include contexts

        Returns:
            Dataset: Created dataset
        """
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm.models import Dataset

        # Generate sample dataset
        file_path, metadata = SampleDatasetGenerator.generate_question_answer_dataset(
            name=name,
            num_samples=num_samples,
            domain=domain,
            include_contexts=include_contexts
        )

        # Create dataset in database
        dataset_repo = BaseRepository(Dataset, db_session)

        dataset_data = {
            "name": metadata["name"],
            "description": metadata["description"],
            "type": DatasetType.QUESTION_ANSWER,
            "file_path": file_path,
            "schema": metadata["schema"],
            "meta_info": {"domain": domain, "generated": True},
            "version": metadata["version"],
            "row_count": metadata["row_count"],
            "is_public": True,
            "owner_id": owner.id
        }

        dataset = await dataset_repo.create(dataset_data)
        logger.info(f"Created sample dataset in database: {dataset.id}")

        return dataset


class SamplePromptGenerator:
    """Utility class for generating sample prompts for testing evaluation."""

    @staticmethod
    def get_rag_system_prompt(domain: str = "general") -> str:
        """
        Get a sample RAG system prompt.

        Args:
            domain: Domain for the prompt

        Returns:
            str: System prompt
        """
        prompts = {
            "general": """You are a helpful AI assistant that answers questions based on the provided context. 
Follow these guidelines:
1. Use ONLY the information from the provided context to answer questions.
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Answer in a clear, concise, and direct manner.
4. Do not make up information or use external knowledge beyond the provided context.
5. If the question is unclear or ambiguous, ask for clarification.

Context: {context}
Question: {query}""",

            "programming": """You are a technical assistant specialized in programming and software development.
Answer questions based ONLY on the provided technical documentation or context.
Follow these guidelines:
1. Use ONLY the information from the provided context to answer questions.
2. If the context doesn't contain the relevant technical information, say "The documentation doesn't cover this topic."
3. Answer in a precise and technical manner, using proper terminology.
4. Include code examples when appropriate, but only if they are derived from the provided context.
5. For technical concepts, provide clear explanations suitable for the implied expertise level of the question.

Technical Documentation: {context}
Technical Question: {query}"""
        }

        # Default to general if domain not found
        return prompts.get(domain, prompts["general"])

    @staticmethod
    async def create_sample_prompt_in_db(
            db_session,
            owner: User,
            name: str = "Sample RAG Prompt",
            domain: str = "general"
    ) -> 'Prompt':
        """
        Create a sample prompt and store it in the database.

        Args:
            db_session: Database session
            owner: Prompt owner
            name: Prompt name
            domain: Domain for the prompt

        Returns:
            Prompt: Created prompt
        """
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm.models import Prompt

        # Get prompt content
        content = SamplePromptGenerator.get_rag_system_prompt(domain)

        # Create prompt in database
        prompt_repo = BaseRepository(Prompt, db_session)

        prompt_data = {
            "name": name,
            "description": f"Sample RAG prompt for {domain} domain",
            "content": content,
            "parameters": {"placeholders": ["context", "query"]},
            "version": "1.0.0",
            "is_public": True,
            "owner_id": owner.id
        }

        prompt = await prompt_repo.create(prompt_data)
        logger.info(f"Created sample prompt in database: {prompt.id}")

        return prompt


class SampleMicroAgentGenerator:
    """Utility class for generating sample micro-agents for testing evaluation."""

    @staticmethod
    async def create_sample_microagent_in_db(
            db_session,
            name: str = "Sample OpenAI Agent",
            domain: str = "general",
            use_azure: bool = False
    ) -> 'MicroAgent':
        """
        Create a sample micro-agent and store it in the database.

        Args:
            db_session: Database session
            name: Micro-agent name
            domain: Domain for the micro-agent
            use_azure: Whether to use Azure OpenAI

        Returns:
            MicroAgent: Created micro-agent
        """
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm.models import MicroAgent

        # Determine API endpoint based on use_azure
        if use_azure:
            api_endpoint = f"{settings.AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
            provider = "azure_openai"
        else:
            api_endpoint = "http://localhost:8000/api/test/microagent/ask"  # Local test endpoint
            provider = "openai"

        # Create micro-agent in database
        agent_repo = BaseRepository(MicroAgent, db_session)

        agent_data = {
            "name": name,
            "description": f"Sample {domain} micro-agent for evaluation testing",
            "api_endpoint": api_endpoint,
            "domain": domain,
            "config": {
                "provider": provider,
                "model": "gpt-4",
                "temperature": 0.0,
                "max_tokens": 1000
            },
            "is_active": True
        }

        agent = await agent_repo.create(agent_data)
        logger.info(f"Created sample micro-agent in database: {agent.id}")

        return agent


class SampleEvaluationBuilder:
    """Utility for building sample evaluations for testing."""

    @staticmethod
    async def create_sample_evaluation(
            db_session,
            user: User,
            method: str = "ragas",
            num_samples: int = 5,
            domain: str = "general",
            include_contexts: bool = True
    ) -> Tuple['Evaluation', 'Dataset', 'Prompt', 'MicroAgent']:
        """
        Create a complete sample evaluation setup.

        Args:
            db_session: Database session
            user: Evaluation owner
            method: Evaluation method
            num_samples: Number of samples in the dataset
            domain: Domain for the evaluation
            include_contexts: Whether to include contexts in the dataset

        Returns:
            Tuple: (Evaluation, Dataset, Prompt, MicroAgent)
        """
        from backend.app.db.repositories.base import BaseRepository
        from backend.app.db.models.orm.models import Evaluation, EvaluationMethod, EvaluationStatus

        # Create sample dataset
        dataset = await SampleDatasetGenerator.create_sample_dataset_in_db(
            db_session=db_session,
            owner=user,
            name=f"Sample {domain.capitalize()} Dataset",
            num_samples=num_samples,
            domain=domain,
            include_contexts=include_contexts
        )

        # Create sample prompt
        prompt = await SamplePromptGenerator.create_sample_prompt_in_db(
            db_session=db_session,
            owner=user,
            name=f"Sample {domain.capitalize()} Prompt",
            domain=domain
        )

        # Create sample micro-agent
        microagent = await SampleMicroAgentGenerator.create_sample_microagent_in_db(
            db_session=db_session,
            name=f"Sample {domain.capitalize()} Agent",
            domain=domain
        )

        # Create evaluation
        eval_repo = BaseRepository(Evaluation, db_session)

        eval_method = getattr(EvaluationMethod, method.upper()) if hasattr(EvaluationMethod,
                                                                           method.upper()) else EvaluationMethod.RAGAS

        eval_data = {
            "name": f"Sample {method.upper()} Evaluation - {domain.capitalize()}",
            "description": f"Sample evaluation of {domain} domain using {method} method",
            "method": eval_method,
            "status": EvaluationStatus.PENDING,
            "config": {
                "metrics": ["faithfulness", "answer_relevancy", "context_relevancy"],
                "batch_size": 2
            },
            "metrics": ["faithfulness", "answer_relevancy", "context_relevancy"],
            "created_by_id": user.id,
            "micro_agent_id": microagent.id,
            "dataset_id": dataset.id,
            "prompt_id": prompt.id
        }

        evaluation = await eval_repo.create(eval_data)
        logger.info(f"Created sample evaluation in database: {evaluation.id}")

        return evaluation, dataset, prompt, microagent

    @staticmethod
    async def run_sample_evaluation(evaluation_service, evaluation_id):
        """
        Run a sample evaluation.

        Args:
            evaluation_service: Evaluation service
            evaluation_id: Evaluation ID

        Returns:
            dict: Results summary
        """
        # Queue the evaluation
        await evaluation_service.queue_evaluation_job(evaluation_id)

        # In a real application, we would wait for the job to complete
        # Here we'll use a direct approach for simplicity
        evaluation = await evaluation_service.get_evaluation(evaluation_id)

        # Get results
        results = await evaluation_service.get_evaluation_results(evaluation_id)

        # Return summary
        return {
            "evaluation_id": evaluation_id,
            "status": evaluation.status,
            "results_count": len(results),
            "statistics": await evaluation_service.get_evaluation_statistics(evaluation_id)
        }