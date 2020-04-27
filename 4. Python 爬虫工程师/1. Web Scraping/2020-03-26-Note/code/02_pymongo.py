import pymongo

conn = pymongo.MongoClient('172.40.91.111', 27017)
db = conn['AID1911db']
myset = db['stu_set']

myset.insert_one({'name': '凯哥'})
