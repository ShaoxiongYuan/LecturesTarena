# Django dumpdata and loaddata

## dumpdata 命令

​	这个是Django自带的管理命令，可以用来备份你的模型实例和数据库

### dumpdata 基本数据库的转存

​	下面的命令将把整个数据转存到db.json文件中 

```shell
python3 manage.py dumpdata > db.json
```

​	

### dumpdata 备份特定的 app

​	下面的命令将在django goods app 转存到 goods.json文件中 

```shell
python3 manage.py dumpdata goods > goods.json
```

goods.json内容

```json
[{"model": "goods.catalog", "pk": 1, "fields": {"create_time": "2019-12-04T21:21:00.815", "update_time": "2019-12-04T21:21:00.815", "name": "\u624b\u63d0\u5305"}}, {"model": "goods.brand", "pk": 1, "fields": {"create_time": "2019-12-04T21:20:24.283", "update_time": "2019-12-04T21:20:24.283", "name": "\u5b89\u8e0f", "logo": "c04303ce2b639e4e4a67944ab353b22.jpg", "first_letter": "A"}}, {"model": "goods.brand", "pk": 2, "fields": {"create_time": "2019-12-04T21:20:50.972", "update_time": "2019-12-04T21:20:50.972", "name": "\u963f\u8fea\u8fbe\u65af", "logo": "addias_FeCs4M8.jpg", "first_letter": "A"}}, {"model": "goods.spu", "pk": 1, "fields": {"create_time": "2019-12-04T21:22:27.196", "update_time": "2019-12-04T21:22:27.196", "name": "\u5b89\u8e0fA", "sales": 1000, "comments": 1000, "brand": 1, "catalog": 1}}, {"model": "goods.spu", "pk": 2, "fields": {"create_time": "2019-12-04T21:22:44.194", "update_time": "2019-12-04T21:22:44.194", "name": "\u5b89\u8e0fB", "sales": 1000, "comments": 1000, "brand": 1, "catalog": 1}}, {"model": "goods.spu", "pk": 3, "fields": {"create_time": "2019-12-04T21:23:04.072", "update_time": "2019-12-04T21:23:04.072", "name": "CLAS1", "sales": 1000, "comments": 1000, "brand": 2, "catalog": 1}}, {"model": "goods.spusaleattr", "pk": 1, "fields": {"create_time": "2019-12-04T21:23:48.585", "update_time": "2019-12-04T21:23:48.585", "SPU_id": 1, "sale_attr_name": "\u5b89\u8e0fA/\u5c3a\u5bf8"}}, {"model": "goods.spusaleattr", "pk": 2, "fields": {"create_time": "2019-12-04T21:23:56.275", "update_time": "2019-12-04T21:23:56.275", "SPU_id": 1, "sale_attr_name": "\u5b89\u8e0fA/\u989c\u8272"}}, {"model": "goods.spusaleattr", "pk": 3, "fields": {"create_time": "2019-12-04T21:24:02.901", "update_time": "2019-12-04T21:24:02.901", "SPU_id": 2, "sale_attr_name": "\u5b89\u8e0fB/\u5c3a\u5bf8"}}, {"model": "goods.spusaleattr", "pk": 4, "fields": {"create_time": "2019-12-04T21:24:07.892", "update_time": "2019-12-04T21:24:07.892", "SPU_id": 2, "sale_attr_name": "\u5b89\u8e0fB/\u989c\u8272"}}, {"model": "goods.spusaleattr", "pk": 5, "fields": {"create_time": "2019-12-04T21:24:12.751", "update_time": "2019-12-04T21:24:12.751", "SPU_id": 3, "sale_attr_name": "CLAS1/\u5c3a\u5bf8"}}, {"model": "goods.spusaleattr", "pk": 6, "fields": {"create_time": "2019-12-04T21:24:18.093", "update_time": "2019-12-04T21:24:18.093", "SPU_id": 3, "sale_attr_name": "CLAS1/\u989c\u8272"}}, {"model": "goods.sku", "pk": 1, "fields": {"create_time": "2019-12-04T21:25:10.613", "update_time": "2019-12-23T20:04:15.419", "name": "\u5b89\u8e0fA\u84dd\u8272\u5c0f\u5c3a\u5bf8", "caption": "\u84dd\u8272\u5c0f\u5c3a\u5bf8", "SPU_ID": 1, "price": "100.00", "cost_price": "1000.00", "market_price": "1000.00", "stock": 893, "sales": 1107, "comments": 100, "is_launched": true, "default_image_url": "2_5pdezhv.jpg", "version": 17}}, {"model": "goods.sku", "pk": 2, "fields": {"create_time": "2019-12-04T21:25:58.561", "update_time": "2019-12-13T17:02:11.021", "name": "\u5b89\u8e0fA\u7ea2\u8272\u5927\u5c3a\u5bf8", "caption": "\u7ea2\u8272\u5927\u5c3a\u5bf8", "SPU_ID": 1, "price": "10000.00", "cost_price": "1000.00", "market_price": "1000.00", "stock": 919, "sales": 1081, "comments": 1000, "is_launched": true, "default_image_url": "3_JGA6F97.jpg", "version": 10}}, {"model": "goods.sku", "pk": 3, "fields": {"create_time": "2019-12-04T21:26:23.278", "update_time": "2019-12-13T17:02:22.045", "name": "\u5b89\u8e0fB\u7ea2\u8272\u5927\u5c3a\u5bf8", "caption": "\u84dd\u8272\u5c0f\u5c3a\u5bf8", "SPU_ID": 2, "price": "1000.00", "cost_price": "1000.00", "market_price": "1000.00", "stock": 904, "sales": 1096, "comments": 1000, "is_launched": true, "default_image_url": "4_z3FdRMq.jpg", "version": 10}}, {"model": "goods.saleattrvalue", "pk": 1, "fields": {"create_time": "2019-12-04T21:26:51.674", "update_time": "2019-12-04T21:26:51.674", "sale_attr_id": 1, "sku": 1, "sale_attr_value_name": "15\u5bf8"}}, {"model": "goods.saleattrvalue", "pk": 2, "fields": {"create_time": "2019-12-04T21:27:05.600", "update_time": "2019-12-04T21:27:05.600", "sale_attr_id": 2, "sku": 1, "sale_attr_value_name": "\u84dd\u8272"}}, {"model": "goods.saleattrvalue", "pk": 12, "fields": {"create_time": "2019-12-04T22:32:35.827", "update_time": "2019-12-04T22:32:35.827", "sale_attr_id": 1, "sku": 2, "sale_attr_value_name": "18\u5bf8"}}, {"model": "goods.saleattrvalue", "pk": 13, "fields": {"create_time": "2019-12-04T22:32:50.020", "update_time": "2019-12-04T22:32:50.020", "sale_attr_id": 2, "sku": 2, "sale_attr_value_name": "\u7eff\u8272"}}, {"model": "goods.saleattrvalue", "pk": 14, "fields": {"create_time": "2019-12-04T22:34:25.974", "update_time": "2019-12-04T22:34:25.974", "sale_attr_id": 3, "sku": 3, "sale_attr_value_name": "18\u5bf8"}}, {"model": "goods.saleattrvalue", "pk": 15, "fields": {"create_time": "2019-12-04T22:34:33.273", "update_time": "2019-12-04T22:34:33.273", "sale_attr_id": 4, "sku": 3, "sale_attr_value_name": "\u84dd\u8272"}}]
```

### dumpdata 备份特定的表

​	下面的命令将只转存django admin.sku 表中的内容 

```shell
python3 manage.py dumpdata goods.sku > sku.json
```

​	下面的命令将只转存django goods.spu 表中的内容 

```shell
python3 manage.py dumpdata goods.spu > spu.json
```



### dumpdata (--exclude)

​	您可以使用 --exclude 选择不需要备份的app或者表 

​	您可以使用 --indent 加上一个代表空格数的数字 格式化输出 

​	下面的命令将除了goods app的sku表 其余的goods表都备份，按照2个空格的格式生成goods.json

```
python3 manage.py dumpdata goods --exclude goods.sku --indent 2 >goods.json
```

查看goods.json内容如下：

```json
[
{
  "model": "goods.catalog",
  "pk": 1,
  "fields": {
    "create_time": "2019-12-04T21:21:00.815",
    "update_time": "2019-12-04T21:21:00.815",
    "name": "\u624b\u63d0\u5305"
  }
},
{
  "model": "goods.brand",
  "pk": 1,
  "fields": {
    "create_time": "2019-12-04T21:20:24.283",
    "update_time": "2019-12-04T21:20:24.283",
    "name": "\u5b89\u8e0f",
    "logo": "c04303ce2b639e4e4a67944ab353b22.jpg",
    "first_letter": "A"
  }
},
{
  "model": "goods.brand",
  "pk": 2,
  "fields": {
    "create_time": "2019-12-04T21:20:50.972",
    "update_time": "2019-12-04T21:20:50.972",
    "name": "\u963f\u8fea\u8fbe\u65af",
    "logo": "addias_FeCs4M8.jpg",
    "first_letter": "A"
  }
},
{
  "model": "goods.spu",
  "pk": 1,
  "fields": {
    "create_time": "2019-12-04T21:22:27.196",
    "update_time": "2019-12-04T21:22:27.196",
    "name": "\u5b89\u8e0fA",
    "sales": 1000,
    "comments": 1000,
    "brand": 1,
    "catalog": 1
  }
},
{
  "model": "goods.spu",
  "pk": 2,
  "fields": {
    "create_time": "2019-12-04T21:22:44.194",
    "update_time": "2019-12-04T21:22:44.194",
    "name": "\u5b89\u8e0fB",
    "sales": 1000,
    "comments": 1000,
    "brand": 1,
    "catalog": 1
  }
},
{
  "model": "goods.spu",
  "pk": 3,
  "fields": {
    "create_time": "2019-12-04T21:23:04.072",
    "update_time": "2019-12-04T21:23:04.072",
    "name": "CLAS1",
    "sales": 1000,
    "comments": 1000,
    "brand": 2,
    "catalog": 1
  }
},
{
  "model": "goods.spusaleattr",
  "pk": 1,
  "fields": {
    "create_time": "2019-12-04T21:23:48.585",
    "update_time": "2019-12-04T21:23:48.585",
    "SPU_id": 1,
    "sale_attr_name": "\u5b89\u8e0fA/\u5c3a\u5bf8"
  }
},
{
  "model": "goods.spusaleattr",
  "pk": 2,
  "fields": {
    "create_time": "2019-12-04T21:23:56.275",
    "update_time": "2019-12-04T21:23:56.275",
    "SPU_id": 1,
    "sale_attr_name": "\u5b89\u8e0fA/\u989c\u8272"
  }
},
{
  "model": "goods.spusaleattr",
  "pk": 3,
  "fields": {
    "create_time": "2019-12-04T21:24:02.901",
    "update_time": "2019-12-04T21:24:02.901",
    "SPU_id": 2,
    "sale_attr_name": "\u5b89\u8e0fB/\u5c3a\u5bf8"
  }
},
{
  "model": "goods.spusaleattr",
  "pk": 4,
  "fields": {
    "create_time": "2019-12-04T21:24:07.892",
    "update_time": "2019-12-04T21:24:07.892",
    "SPU_id": 2,
    "sale_attr_name": "\u5b89\u8e0fB/\u989c\u8272"
  }
},
{
  "model": "goods.spusaleattr",
  "pk": 5,
  "fields": {
    "create_time": "2019-12-04T21:24:12.751",
    "update_time": "2019-12-04T21:24:12.751",
    "SPU_id": 3,
    "sale_attr_name": "CLAS1/\u5c3a\u5bf8"
  }
},
{
  "model": "goods.spusaleattr",
  "pk": 6,
  "fields": {
    "create_time": "2019-12-04T21:24:18.093",
    "update_time": "2019-12-04T21:24:18.093",
    "SPU_id": 3,
    "sale_attr_name": "CLAS1/\u989c\u8272"
  }
},
{
  "model": "goods.saleattrvalue",
  "pk": 1,
  "fields": {
    "create_time": "2019-12-04T21:26:51.674",
    "update_time": "2019-12-04T21:26:51.674",
    "sale_attr_id": 1,
    "sku": 1,
    "sale_attr_value_name": "15\u5bf8"
  }
},
{
  "model": "goods.saleattrvalue",
  "pk": 2,
  "fields": {
    "create_time": "2019-12-04T21:27:05.600",
    "update_time": "2019-12-04T21:27:05.600",
    "sale_attr_id": 2,
    "sku": 1,
    "sale_attr_value_name": "\u84dd\u8272"
  }
},
{
  "model": "goods.saleattrvalue",
  "pk": 12,
  "fields": {
    "create_time": "2019-12-04T22:32:35.827",
    "update_time": "2019-12-04T22:32:35.827",
    "sale_attr_id": 1,
    "sku": 2,
    "sale_attr_value_name": "18\u5bf8"
  }
},
{
  "model": "goods.saleattrvalue",
  "pk": 13,
  "fields": {
    "create_time": "2019-12-04T22:32:50.020",
    "update_time": "2019-12-04T22:32:50.020",
    "sale_attr_id": 2,
    "sku": 2,
    "sale_attr_value_name": "\u7eff\u8272"
  }
},
{
  "model": "goods.saleattrvalue",
  "pk": 14,
  "fields": {
    "create_time": "2019-12-04T22:34:25.974",
    "update_time": "2019-12-04T22:34:25.974",
    "sale_attr_id": 3,
    "sku": 3,
    "sale_attr_value_name": "18\u5bf8"
  }
},
{
  "model": "goods.saleattrvalue",
  "pk": 15,
  "fields": {
    "create_time": "2019-12-04T22:34:33.273",
    "update_time": "2019-12-04T22:34:33.273",
    "sale_attr_id": 4,
    "sku": 3,
    "sale_attr_value_name": "\u84dd\u8272"
  }
}
]
```



### dumpdata (--format)

​	通常，dumpdata将会格式化数据输出为json格式，但是你也可以使用 --format 来选择自己想要的格式

​	命令支持选择的格式 

**json**

**xml**

**yaml**

​	下面备份goods app sku表中内容 格式为xml

```shell
python3 manage.py dumpdata goods.sku --indent 2 --format xml >sku.xml
```

​	sku.xml内容如下：

```xml
<?xml version="1.0" encoding="utf-8"?>
<django-objects version="1.0">
  <object model="goods.sku" pk="1">
    <field name="create_time" type="DateTimeField">2019-12-04T21:25:10.613395</field>
    <field name="update_time" type="DateTimeField">2019-12-23T20:04:15.419622</field>
    <field name="name" type="CharField">安踏A蓝色小尺寸</field>
    <field name="caption" type="CharField">蓝色小尺寸</field>
    <field name="SPU_ID" rel="ManyToOneRel" to="goods.spu">1</field>
    <field name="price" type="DecimalField">100.00</field>
    <field name="cost_price" type="DecimalField">1000.00</field>
    <field name="market_price" type="DecimalField">1000.00</field>
    <field name="stock" type="IntegerField">893</field>
    <field name="sales" type="IntegerField">1107</field>
    <field name="comments" type="IntegerField">100</field>
    <field name="is_launched" type="BooleanField">True</field>
    <field name="default_image_url" type="FileField">2_5pdezhv.jpg</field>
    <field name="version" type="IntegerField">17</field>
  </object>
  <object model="goods.sku" pk="2">
    <field name="create_time" type="DateTimeField">2019-12-04T21:25:58.561235</field>
    <field name="update_time" type="DateTimeField">2019-12-13T17:02:11.021269</field>
    <field name="name" type="CharField">安踏A红色大尺寸</field>
    <field name="caption" type="CharField">红色大尺寸</field>
    <field name="SPU_ID" rel="ManyToOneRel" to="goods.spu">1</field>
    <field name="price" type="DecimalField">10000.00</field>
    <field name="cost_price" type="DecimalField">1000.00</field>
    <field name="market_price" type="DecimalField">1000.00</field>
    <field name="stock" type="IntegerField">919</field>
    <field name="sales" type="IntegerField">1081</field>
    <field name="comments" type="IntegerField">1000</field>
    <field name="is_launched" type="BooleanField">True</field>
    <field name="default_image_url" type="FileField">3_JGA6F97.jpg</field>
    <field name="version" type="IntegerField">10</field>
  </object>
  <object model="goods.sku" pk="3">
    <field name="create_time" type="DateTimeField">2019-12-04T21:26:23.278899</field>
    <field name="update_time" type="DateTimeField">2019-12-13T17:02:22.045703</field>
    <field name="name" type="CharField">安踏B红色大尺寸</field>
    <field name="caption" type="CharField">蓝色小尺寸</field>
    <field name="SPU_ID" rel="ManyToOneRel" to="goods.spu">2</field>
    <field name="price" type="DecimalField">1000.00</field>
    <field name="cost_price" type="DecimalField">1000.00</field>
    <field name="market_price" type="DecimalField">1000.00</field>
    <field name="stock" type="IntegerField">904</field>
    <field name="sales" type="IntegerField">1096</field>
    <field name="comments" type="IntegerField">1000</field>
    <field name="is_launched" type="BooleanField">True</field>
    <field name="default_image_url" type="FileField">4_z3FdRMq.jpg</field>
    <field name="version" type="IntegerField">10</field>
  </object>
```





## loaddata 命令

这个命令可以用来导入固定格式的数据(数据库 dumps)到数据库 

```
python3 manage.py loaddata goods.json
```

















