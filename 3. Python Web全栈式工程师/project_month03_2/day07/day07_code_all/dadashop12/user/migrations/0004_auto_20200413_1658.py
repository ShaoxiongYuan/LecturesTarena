# -*- coding: utf-8 -*-
# Generated by Django 1.11.8 on 2020-04-13 16:58
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ('user', '0003_auto_20200413_1653'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userprofile',
            name='created_time',
            field=models.DateTimeField(auto_now_add=True),
        ),
        migrations.AlterField(
            model_name='userprofile',
            name='updated_time',
            field=models.DateTimeField(auto_now=True),
        ),
    ]
