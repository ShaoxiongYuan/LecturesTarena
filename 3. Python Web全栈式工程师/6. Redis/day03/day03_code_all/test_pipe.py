import redis
# 创建连接池并连接到redis
pool = redis.ConnectionPool(host = '127.0.0.1',db=0,port=6379)
r = redis.Redis(connection_pool=pool)

def withpipeline(r):
    p = r.pipeline()
    for i in range(1000):
        key = 'test1' + str(i)
        value = i+1
        p.set(key, value)
    p.execute()

def withoutpipeline(r):
    for i in range(1000):
        key = 'test2' + str(i)
        value = i+1
        r.set(key, value)

if __name__ == '__main__':

    import time
    t1 = time.time()
    #time is 0.04256486892700195
    #withpipeline(r)

    #time is 0.24103689193725586
    withoutpipeline(r)
    t2 = time.time()
    print('time is %s'%(t2-t1))


