"""add data_hash to model_metadata

Revision ID: a1b2c3d4e5f6
Revises: 5c5dace0df8b
Create Date: 2026-04-27 16:35:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "5c5dace0df8b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add data_hash column for idempotent training detection."""
    op.add_column(
        "model_metadata",
        sa.Column("data_hash", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    """Remove data_hash column."""
    op.drop_column("model_metadata", "data_hash")
