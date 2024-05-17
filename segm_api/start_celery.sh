#!/bin/sh

# Start the Celery worker
celery -A src.tasks worker --loglevel=info -Q analysis_queue
