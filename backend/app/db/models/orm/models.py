# File: app/db/models/orm/models.py
from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SQLEnum, Float, ForeignKey,
    Integer, JSON, String, Table, Text, func
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models.orm.base import Base, ModelMixin, TimestampMixin


class UserRole(str, Enum):
    ADMIN = "admin"
    EVALUATOR = "evaluator"
    VIEWER = "viewer"


class User(Base, TimestampMixin, ModelMixin):
    """
    User model - linked with external OIDC authentication.
    We only store the external ID and necessary profile info.
    """

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Relationships
    datasets: Mapped[List["Dataset"]] = relationship(back_populates="owner")
    prompts: Mapped[List["Prompt"]] = relationship(back_populates="owner")
    evaluations: Mapped[List["Evaluation"]] = relationship(back_populates="created_by")


class Agent(Base, TimestampMixin, ModelMixin):
    """
    Agent model representing individual specialized agents in the system.
    """

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    api_endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    config: Mapped[dict] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    # New fields
    model_type: Mapped[str] = mapped_column(String(100), nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    created_by_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("user.id"), nullable=True)
    tags: Mapped[List[str]] = mapped_column(JSON, nullable=True)

    # Relationships
    evaluations: Mapped[List["Evaluation"]] = relationship(back_populates="agent")
    created_by: Mapped[Optional["User"]] = relationship(foreign_keys=[created_by_id])


class DatasetType(str, Enum):
    USER_QUERY = "user_query"
    CONTEXT = "context"
    QUESTION_ANSWER = "question_answer"
    CONVERSATION = "conversation"
    CUSTOM = "custom"


class Dataset(Base, TimestampMixin, ModelMixin):
    """
    Dataset model for evaluation data.
    """

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    type: Mapped[DatasetType] = mapped_column(SQLEnum(DatasetType), nullable=False)
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    schema: Mapped[dict] = mapped_column(JSON, nullable=True)
    meta_info: Mapped[dict] = mapped_column(JSON, nullable=True)  # Renamed from metadata
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    row_count: Mapped[int] = mapped_column(Integer, nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    owner: Mapped["User"] = relationship(back_populates="datasets")
    evaluations: Mapped[List["Evaluation"]] = relationship(back_populates="dataset")


class PromptTemplate(Base, TimestampMixin, ModelMixin):
    """
    Prompt template model.
    """

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    variables: Mapped[dict] = mapped_column(JSON, nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")

    # Relationships
    prompts: Mapped[List["Prompt"]] = relationship(back_populates="template")


class Prompt(Base, TimestampMixin, ModelMixin):
    """
    Concrete Prompt instance, possibly based on a template.
    """

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    owner_id: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    owner: Mapped["User"] = relationship(back_populates="prompts")
    template_id: Mapped[Optional[UUID]] = mapped_column(
        ForeignKey("prompttemplate.id"), nullable=True
    )
    template: Mapped[Optional["PromptTemplate"]] = relationship(back_populates="prompts")
    evaluations: Mapped[List["Evaluation"]] = relationship(back_populates="prompt")


class EvaluationMethod(str, Enum):
    RAGAS = "ragas"
    DEEPEVAL = "deepeval"
    CUSTOM = "custom"
    MANUAL = "manual"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Evaluation(Base, TimestampMixin, ModelMixin):
    """
    Evaluation model representing a specific evaluation run.
    """

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    method: Mapped[EvaluationMethod] = mapped_column(
        SQLEnum(EvaluationMethod), nullable=False
    )
    status: Mapped[EvaluationStatus] = mapped_column(
        SQLEnum(EvaluationStatus), nullable=False, default=EvaluationStatus.PENDING
    )
    config: Mapped[dict] = mapped_column(JSON, nullable=True)
    metrics: Mapped[List[str]] = mapped_column(JSON, nullable=True)
    experiment_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    created_by_id: Mapped[UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    created_by: Mapped["User"] = relationship(back_populates="evaluations")
    agent_id: Mapped[UUID] = mapped_column(ForeignKey("agent.id"), nullable=False)
    agent: Mapped["Agent"] = relationship(back_populates="evaluations")
    dataset_id: Mapped[UUID] = mapped_column(ForeignKey("dataset.id"), nullable=False)
    dataset: Mapped["Dataset"] = relationship(back_populates="evaluations")
    prompt_id: Mapped[UUID] = mapped_column(ForeignKey("prompt.id"), nullable=False)
    prompt: Mapped["Prompt"] = relationship(back_populates="evaluations")
    results: Mapped[List["EvaluationResult"]] = relationship(back_populates="evaluation")


class MetricScore(Base, TimestampMixin, ModelMixin):
    """
    Individual metric score within an evaluation result.
    """

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    meta_info: Mapped[dict] = mapped_column(JSON, nullable=True)  # Renamed from metadata

    # Relationships
    result_id: Mapped[UUID] = mapped_column(ForeignKey("evaluationresult.id"), nullable=False)
    result: Mapped["EvaluationResult"] = relationship(back_populates="metric_scores")


class EvaluationResult(Base, TimestampMixin, ModelMixin):
    """
    Results of an evaluation run.
    """

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    overall_score: Mapped[float] = mapped_column(Float, nullable=True)
    raw_results: Mapped[dict] = mapped_column(JSON, nullable=True)
    dataset_sample_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    input_data: Mapped[dict] = mapped_column(JSON, nullable=True)
    output_data: Mapped[dict] = mapped_column(JSON, nullable=True)
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    evaluation_id: Mapped[UUID] = mapped_column(ForeignKey("evaluation.id"), nullable=False)
    evaluation: Mapped["Evaluation"] = relationship(back_populates="results")
    metric_scores: Mapped[List["MetricScore"]] = relationship(back_populates="result")


# Many-to-many association table for comparing evaluations
evaluation_comparison = Table(
    "evaluation_comparison",
    Base.metadata,  # This is SQLAlchemy's metadata, not our column
    Column("id", PostgresUUID(as_uuid=True), primary_key=True, default=uuid4),
    Column("name", String(255), nullable=False),
    Column("description", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), server_default=func.now()),
    Column("created_by_id", ForeignKey("user.id"), nullable=False),
    Column("evaluation_a_id", ForeignKey("evaluation.id"), nullable=False),
    Column("evaluation_b_id", ForeignKey("evaluation.id"), nullable=False),
    Column("comparison_results", JSON, nullable=True)
)