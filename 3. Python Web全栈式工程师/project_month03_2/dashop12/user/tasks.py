from django.conf import settings
from django.core.mail import send_mail
from dadashop.celery import app

@app.task
def send_active_email(email_address, verify_url):
    # 发送邮箱激活链接
    subject = '达达商城激活邮件'
    html_message = '<p>尊敬的用户您好！</p>' \
                   '<p>欢迎注册使用达达商城。</p>' \
                   '<p>您的邮箱为：%s，请点击此链接激活您的邮箱(30分钟有效)：</p>' \
                   '<p><a href="%s" target="_blank"><b>请点击激活</b></a></p>' % (email_address, verify_url)
    send_mail(subject, '', settings.EMAIL_FROM, [email_address], html_message=html_message)


def send_password_email(email_address, code):

    subject = '达达商城密码找回邮件'
    html_message = '<p>尊敬的用户您好！</p>' \
                   '<p>欢迎注册使用达达商城。</p>' \
                   '<p>您的邮箱为：%s，您的邮箱验证码为：< /p>' \
                   '<b>%s</b><p>10分钟之内有效。</p>' % (email_address, code)
    send_mail(subject, '', settings.EMAIL_FROM, [email_address], html_message=html_message)

