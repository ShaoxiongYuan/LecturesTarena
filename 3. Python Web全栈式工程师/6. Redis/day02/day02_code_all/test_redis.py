import redis

r = redis.Redis(host='127.0.0.1', port=6379, db=0)

#通用命令演示
# key_list = r.keys('*')
# for key in key_list:
#     print(key)

#返回值 1 代表存在  0 代表不存在
#print(r.exists('l1'))

#############list##############
#r.lpush('pyl1','a','b','c','d')
#返回值 [b'd', b'c', b'b', b'a']
#print(r.lrange('pyl1',0, -1))

#返回值 字节串
#print(r.rpop('pyl1'))

#返回值 True 代表成功
#print(r.ltrim('pyl1', 0, 1))
#print(r.lrange('pyl1', 0, -1))


#########string##########
# r.set('pyusername', 'guoxiaonao')
# print(r.get('pyusername'))

# mset key1 value1 key2 value2
#r.mset({'pyusername1':'wanglaoshi','pyusername2':'lvlaoshi'})
#[b'wanglaoshi', b'lvlaoshi']
#print(r.mget('pyusername1', 'pyusername2'))

#r.incr('pyage')
#返回值是 int
#print(r.incrby('pyage',10))

######hash######
#r.hset('pyh1', 'name', 'guoxiaonao')
#返回值 b'guoxiaonao'
#print(r.hget('pyh1','name'))

#r.hmset('pyh1',{'age':18, 'desc':'hahaha'})
#返回值 {b'name': b'guoxiaonao', b'age': b'18', b'desc': b'hahaha'}
#print(r.hgetall('pyh1'))

#######set#######
#r.sadd('pys1', 'a', 'b', 'c', 'd')
#返回值 python的集合  {b'c', b'a', b'b', b'd'}
#print(r.smembers('pys1'))

#返回值 字节串
# m1 = r.spop('pys1')
# print(m1)

#r.sadd('pys2', 'a', 'b', 'c', 'd')
#返回值 {b'a', b'c', b'd'}
#print(r.sinter('pys1', 'pys2'))


######sorted_set######

#r.zadd('pyz1',{'guoxiaonao':8000,'wangweichao':12000})
#[(b'guoxiaonao', 8000.0), (b'wangweichao', 12000.0)]
#print(r.zrange('pyz1', 0, -1, withscores=True))

r.zadd('pyz2',{'guoxiaonao':6000})
r.zinterstore('pyz3',('pyz1','pyz2'),aggregate='max')
print(r.zrange('pyz3', 0, -1 , withscores=True))


