"""Celery application configuration."""

from celery import Celery

# For Redis, the broker URL format is: redis://:password@hostname:port/db_number
# We assume Redis is running locally without a password on the default port.
REDIS_URL = "redis://localhost:6379/0"

celery_app = Celery(
    "findataanalyzer",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["findataanalyzer.image_analysis.tasks"]  # Path to the tasks module
)

celery_app.conf.update(
    task_track_started=True,
) 