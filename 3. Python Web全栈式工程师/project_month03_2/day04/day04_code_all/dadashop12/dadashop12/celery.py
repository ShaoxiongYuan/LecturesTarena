from celery import Celery
from django.conf import settings
import os

#添加django环境变量
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dadashop12.settings')

app = Celery('dashop12')
app.conf.update(

    BROKER_URL = 'redis://@127.0.0.1:6379/1'

)

app.autodiscover_tasks(settings.INSTALLED_APPS)

