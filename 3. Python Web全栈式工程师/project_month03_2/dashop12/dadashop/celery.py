from celery import Celery
from django.conf import settings
import os

#为celery设置 Django settings的位置
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dadashop.settings')

app = Celery('dadashop')
app.conf.update(

    BROKER_URL='redis://@127.0.0.1:6379/8'
)
app.autodiscover_tasks(settings.INSTALLED_APPS)
