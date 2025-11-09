"""Initial migration for MySQL

Revision ID: 001_mysql
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers
revision = '001_mysql'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
  print("Upgrading MySQL")
  # Create cameras table with MySQL optimizations
  op.create_table(
    'cameras',
    sa.Column('id', sa.String(36), nullable=False),
    sa.Column('name', sa.String(255), nullable=False),
    sa.Column('location', sa.String(255), nullable=True),
    sa.Column('rtsp_url', sa.String(500), nullable=True),
    sa.Column('width', sa.Integer(), nullable=True, default=640),
    sa.Column('height', sa.Integer(), nullable=True, default=480),
    sa.Column('fps', sa.Integer(), nullable=True, default=30),
    sa.Column('is_calibrated', sa.Boolean(), nullable=True, default=False),
    sa.Column('pixels_per_meter', sa.Float(), nullable=True),
    sa.Column('calibration_mode', sa.String(50), nullable=True),
    sa.Column('calibration_points', mysql.JSON(), nullable=True),
    sa.Column('features', mysql.JSON(), nullable=True),
    sa.Column('active_models', mysql.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, onupdate=sa.text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')),
    sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
    sa.PrimaryKeyConstraint('id'),
    mysql_engine='InnoDB',  # Use InnoDB for better concurrency
    mysql_charset='utf8mb4',  # Support emoji and special characters
    mysql_collate='utf8mb4_unicode_ci'
  )

  # Create indexes for faster lookups
  op.create_index('idx_cameras_name', 'cameras', ['name'])
  op.create_index('idx_cameras_active', 'cameras', ['is_active'])
  op.create_index('idx_cameras_created_at', 'cameras', ['created_at'])


def downgrade():
  op.drop_index('idx_cameras_created_at', 'cameras')
  op.drop_index('idx_cameras_active', 'cameras')
  op.drop_index('idx_cameras_name', 'cameras')
  op.drop_table('cameras')