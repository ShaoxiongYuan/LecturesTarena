"""
模仿死锁现象
"""
from threading import Lock,Thread
from time import sleep

# 账户
class Account:
    def __init__(self,id,balance,lock):
        self.id = id
        self.balance = balance # 存款
        self.lock = lock

# 初始化2个账户
Tom = Account('1',5000,Lock())
Abby = Account('2',4000,Lock())

# 转账
def transfer(f,t,amount):
    """
    :param f:  转出
    :param t:  转入
    :param amount: 金额
    """
    f.lock.acquire() # 自己的锁住
    f.balance -= amount
    f.lock.release()  # 账户解锁   不会产生死锁

    sleep(0.1)
    t.lock.acquire() # 对方的账户锁住
    t.balance += amount

    # f.lock.release() # 账户解锁 会产生死锁
    t.lock.release()

t1 = Thread(target=transfer,args=(Tom,Abby,1500))
t2 = Thread(target=transfer,args=(Abby,Tom,500))
t1.start()
t2.start()
t1.join()
t2.join()

print("Tom:",Tom.balance)
print("Abby:",Abby.balance)