"""Initial migration

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
  # Create cameras table
  op.create_table(
    'cameras',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('location', sa.String(), nullable=True),
    sa.Column('rtsp_url', sa.String(), nullable=True),
    sa.Column('width', sa.Integer(), nullable=True, default=640),
    sa.Column('height', sa.Integer(), nullable=True, default=480),
    sa.Column('fps', sa.Integer(), nullable=True, default=30),
    sa.Column('is_calibrated', sa.Boolean(), nullable=True, default=False),
    sa.Column('pixels_per_meter', sa.Float(), nullable=True),
    sa.Column('calibration_mode', sa.String(), nullable=True),
    sa.Column('calibration_points', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('features', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('active_models', postgresql.JSON(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
    sa.PrimaryKeyConstraint('id')
  )

  # Create index on name for faster lookups
  op.create_index('idx_cameras_name', 'cameras', ['name'])
  op.create_index('idx_cameras_active', 'cameras', ['is_active'])


def downgrade():
  op.drop_index('idx_cameras_active')
  op.drop_index('idx_cameras_name')
  op.drop_table('cameras')