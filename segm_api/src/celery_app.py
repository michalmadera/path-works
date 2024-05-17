from celery import Celery
import os

redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')

celery_app = Celery('tasks', broker=redis_url, backend=redis_url, include=['src.tasks'])

celery_app.conf.task_routes = {
    'tasks.perform_analysis': {'queue': 'analysis_queue'}
}
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
celery_app.autodiscover_tasks(['src'])