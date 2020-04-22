from django.contrib import admin

# Register your models here.
from .models import UserProfile,Address,WeiboProfile
admin.site.register(UserProfile)
admin.site.register(Address)
admin.site.register(WeiboProfile)