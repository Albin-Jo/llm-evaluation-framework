"""Updated Agents model.Added Integration type.
Revision ID: f4809cf6e553
Revises: c4f1e6b457a2
Create Date: 2025-05-11 08:38:01.815099
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f4809cf6e553"
down_revision: Union[str, None] = "c4f1e6b457a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    # Create the enum types FIRST
    integration_type_enum = postgresql.ENUM(
        "AZURE_OPENAI", "MCP", "DIRECT_API", "CUSTOM",
        name="integrationtype",
        create_type=True
    )
    integration_type_enum.create(op.get_bind())

    auth_type_enum = postgresql.ENUM(
        "API_KEY", "BEARER_TOKEN", "NONE",
        name="authtype",
        create_type=True
    )
    auth_type_enum.create(op.get_bind())

    # Now add the columns using these enum types
    op.add_column(
        "agent",
        sa.Column(
            "integration_type",
            sa.Enum(
                "AZURE_OPENAI", "MCP", "DIRECT_API", "CUSTOM", name="integrationtype"
            ),
            server_default="AZURE_OPENAI",  # Add default value for existing rows
            nullable=False,
        ),
    )
    op.add_column(
        "agent",
        sa.Column(
            "auth_type",
            sa.Enum("API_KEY", "BEARER_TOKEN", "NONE", name="authtype"),
            server_default="API_KEY",  # Add default value for existing rows
            nullable=False,
        ),
    )
    op.add_column("agent", sa.Column("auth_credentials", sa.JSON(), nullable=True))
    op.add_column("agent", sa.Column("request_template", sa.JSON(), nullable=True))
    op.add_column(
        "agent", sa.Column("response_format", sa.String(length=50), nullable=True)
    )
    op.add_column("agent", sa.Column("retry_config", sa.JSON(), nullable=True))
    op.add_column("agent", sa.Column("content_filter_config", sa.JSON(), nullable=True))
    op.create_index("idx_agent_auth_type", "agent", ["auth_type"], unique=False)
    op.create_index(
        "idx_agent_integration_type", "agent", ["integration_type"], unique=False
    )

    # Set default retry configuration for existing agents
    op.execute("""
    UPDATE agent 
    SET retry_config = '{"max_retries": 3, "backoff_factor": 1.5, "status_codes": [429, 500, 502, 503, 504]}'
    WHERE retry_config IS NULL
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # First drop indexes and columns
    op.drop_index("idx_agent_integration_type", table_name="agent")
    op.drop_index("idx_agent_auth_type", table_name="agent")
    op.drop_column("agent", "content_filter_config")
    op.drop_column("agent", "retry_config")
    op.drop_column("agent", "response_format")
    op.drop_column("agent", "request_template")
    op.drop_column("agent", "auth_credentials")
    op.drop_column("agent", "auth_type")
    op.drop_column("agent", "integration_type")

    # Then drop the enum types LAST
    op.execute('DROP TYPE IF EXISTS integrationtype')
    op.execute('DROP TYPE IF EXISTS authtype')