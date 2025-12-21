"""Add alert configuration fields to cameras

Revision ID: 002_alerts
Revises: 001_mysql
Create Date: 2024-01-02 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers
revision = '002_alerts'
down_revision = '001_mysql'
branch_labels = None
depends_on = None


def upgrade():
    """Add alert configuration columns to cameras table"""
    print("Adding alert configuration fields to cameras table...")
    
    # Add alert_email column
    op.add_column(
        'cameras',
        sa.Column('alert_email', sa.String(255), nullable=True)
    )
    
    # Add alert_config column (JSON)
    op.add_column(
        'cameras',
        sa.Column('alert_config', mysql.JSON(), nullable=True)
    )
    
    # Create index for alert_email for faster lookups
    op.create_index('idx_cameras_alert_email', 'cameras', ['alert_email'])
    
    print("✅ Alert configuration fields added successfully")


def downgrade():
    """Remove alert configuration columns"""
    op.drop_index('idx_cameras_alert_email', 'cameras')
    op.drop_column('cameras', 'alert_config')
    op.drop_column('cameras', 'alert_email')
