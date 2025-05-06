from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean, Column, DateTime, Enum as SQLEnum, Float, ForeignKey,
    Integer, JSON, String, Table, Text, func, Index
)
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.models.base import Base, ModelMixin, TimestampMixin


class UserRole(str, Enum):
    ADMIN = "admin"
    EVALUATOR = "evaluator"
    VIEWER = "viewer"


class User(Base, TimestampMixin, ModelMixin):
    """
    User model - linked with external OIDC authentication.
    We only store the external ID and necessary profile info.
    """
    __tablename__ = "user"
    __table_args__ = (
        Index('idx_user_external_id', 'external_id'),
        Index('idx_user_email', 'email'),
    )

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    external_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole), nullable=False, default=UserRole.VIEWER)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class Agent(Base, TimestampMixin, ModelMixin):
    """
    Agent model representing individual specialized agents in the system.
    """
    __tablename__ = "agent"
    __table_args__ = (
        Index('idx_agent_name', 'name'),
        Index('idx_agent_domain', 'domain'),
        Index('idx_agent_is_active', 'is_active'),
        Index('idx_agent_domain_is_active', 'domain', 'is_active'),
    )

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    api_endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str] = mapped_column(String(100), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # New fields
    model_type: Mapped[str] = mapped_column(String(100), nullable=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    tags: Mapped[List[str]] = mapped_column(JSON, nullable=True)

    # Relationships
    evaluations: Mapped[List["Evaluation"]] = relationship(back_populates="agent")


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
    __tablename__ = "dataset"
    __table_args__ = (
        Index('idx_dataset_name', 'name'),
        Index('idx_dataset_type', 'type'),
        Index('idx_dataset_is_public', 'is_public'),
        Index('idx_dataset_type_is_public', 'type', 'is_public'),
    )

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    type: Mapped[DatasetType] = mapped_column(SQLEnum(DatasetType), nullable=False)
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    schema_definition: Mapped[dict] = mapped_column(JSON, nullable=True)
    meta_info: Mapped[dict] = mapped_column(JSON, nullable=True)  # Renamed from metadata
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    row_count: Mapped[int] = mapped_column(Integer, nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Relationships
    evaluations: Mapped[List["Evaluation"]] = relationship(back_populates="dataset")


class PromptTemplate(Base, TimestampMixin, ModelMixin):
    """
    Prompt template model.
    """
    __tablename__ = "prompttemplate"
    __table_args__ = (
        Index('idx_prompttemplate_name', 'name'),
        Index('idx_prompttemplate_is_public', 'is_public'),
    )

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
    __tablename__ = "prompt"
    __table_args__ = (
        Index('idx_prompt_name', 'name'),
        Index('idx_prompt_is_public', 'is_public'),
        Index('idx_prompt_template_id', 'template_id'),
    )

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
    __tablename__ = "evaluation"
    __table_args__ = (
        Index('idx_evaluation_name', 'name'),
        Index('idx_evaluation_status', 'status'),
        Index('idx_evaluation_method', 'method'),
        Index('idx_evaluation_agent_id', 'agent_id'),
        Index('idx_evaluation_dataset_id', 'dataset_id'),
        Index('idx_evaluation_prompt_id', 'prompt_id'),
        Index('idx_evaluation_status_created_at', 'status', 'created_at'),
    )

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
    agent_id: Mapped[UUID] = mapped_column(ForeignKey("agent.id"), nullable=False)
    agent: Mapped["Agent"] = relationship(back_populates="evaluations")
    dataset_id: Mapped[UUID] = mapped_column(ForeignKey("dataset.id"), nullable=False)
    dataset: Mapped["Dataset"] = relationship(back_populates="evaluations")
    prompt_id: Mapped[UUID] = mapped_column(ForeignKey("prompt.id"), nullable=False)
    prompt: Mapped["Prompt"] = relationship(back_populates="evaluations")
    results: Mapped[List["EvaluationResult"]] = relationship(back_populates="evaluation")
    reports: Mapped[List["Report"]] = relationship(back_populates="evaluation")


class MetricScore(Base, TimestampMixin, ModelMixin):
    """
    Individual metric score within an evaluation result.
    """
    __tablename__ = "metricscore"
    __table_args__ = (
        Index('idx_metricscore_result_id', 'result_id'),
        Index('idx_metricscore_name', 'name'),
    )

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
    __tablename__ = "evaluationresult"
    __table_args__ = (
        Index('idx_evaluationresult_evaluation_id', 'evaluation_id'),
        Index('idx_evaluationresult_overall_score', 'overall_score'),
    )

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
    Column("evaluation_a_id", ForeignKey("evaluation.id"), nullable=False),
    Column("evaluation_b_id", ForeignKey("evaluation.id"), nullable=False),
    Column("comparison_results", JSON, nullable=True)
)


class ReportStatus(str, Enum):
    DRAFT = "draft"
    GENERATED = "generated"
    SENT = "sent"
    FAILED = "failed"


class ReportFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    JSON = "json"


# Add this class for Report model
class Report(Base, TimestampMixin, ModelMixin):
    """
    Report model for generated evaluation reports.
    """
    __tablename__ = "report"
    __table_args__ = (
        Index('idx_report_name', 'name'),
        Index('idx_report_status', 'status'),
        Index('idx_report_evaluation_id', 'evaluation_id'),
        Index('idx_report_status_format', 'status', 'format'),
    )

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True), primary_key=True, default=uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[ReportStatus] = mapped_column(
        SQLEnum(ReportStatus), nullable=False, default=ReportStatus.DRAFT
    )
    format: Mapped[ReportFormat] = mapped_column(
        SQLEnum(ReportFormat), nullable=False, default=ReportFormat.PDF
    )
    content: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    last_sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    evaluation_id: Mapped[UUID] = mapped_column(ForeignKey("evaluation.id"), nullable=False)
    evaluation: Mapped["Evaluation"] = relationship(back_populates="reports")
