#!/bin/sh

# Start the Celery worker
celery -A src.celery_app worker --loglevel=info -Q analysis_queue
