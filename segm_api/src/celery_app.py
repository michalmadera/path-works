from celery import Celery

celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

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

celery_app.autodiscover_tasks(['tasks'])
