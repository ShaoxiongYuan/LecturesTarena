from django.core.management.base import BaseCommand,CommandError
import os


class Command(BaseCommand):


    def handle(self, *args, **options):
        applist = []
        os.system("ls -l | grep ^d | awk '{print $9}' > c_commands/management/commands/applist")
        fileobjs = open('c_commands/management/commands/applist')

        for fileobj in fileobjs:
            applist.append(fileobj.replace('\n',''))

        for i in applist:
            # 强行删除无需判断
            os.system('rm -rf %s/migrations/0*'%i)
            os.system('rm -rf %s/migrations/__pycache__'%i)
            print(i,'清理完成')

        os.system('rm -f c_commands/management/commands/applist')

        try:
            os.system('python3 manage.py makemigrations')
        except CommandError as e:
            print(e)
            print("makemigrations 失败")
        print("makemigrations 成功")

        try:
            os.system('python3 manage.py migrate')
        except CommandError as e:
            print(e)
            print("migrate 失败")
        print("migrate 成功")


        # 强行删除无需判断
        # os.system('rm -rf user/migrations/0*')
        # os.system('rm -rf user/migrations/__pycache__')
        # print('user清理完成')
        #
        # os.system('rm -rf order/migrations/0*')
        # os.system('rm -rf order/migrations/__pycache__')
        # print('order清理完成')
        #
        # os.system('rm -rf goods/migrations/0*')
        # os.system('rm -rf goods/migrations/__pycache__')
        # print('goods清理完成')
        #
        # try:
        #     os.system('python3 manage.py makemigrations')
        # except CommandError as e:
        #     print(e)
        #     print("makemigrations 失败")
        # print("makemigrations 成功")
        #
        # try:
        #     os.system('python3 manage.py migrate')
        # except CommandError as e:
        #     print(e)
        #     print("migrate 失败")
        # print("migrate 成功")
