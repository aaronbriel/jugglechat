from celery import Celery

from .config import (
    BACKEND_URL,
    BROKER_URL
)

"""
Celery instantiation
"""
app = Celery('celery_app',
             broker=BROKER_URL,
             backend=BACKEND_URL,
             include=['celery_app.tasks'])

if __name__ == '__main__':
    app.start()
