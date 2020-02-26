--  插入练习
-- book数据表中插入一些数据
--
-- 作者： 老舍   鲁迅  钱钟书  沈从文
-- 书的价格  30 -- 120
-- 出版社 ：  中国文学  机械工业  中国教育

insert into book (title,author,publication,price,comment) values
('边城','沈从文','机械工业出版社',36,'小城故事多'),
('骆驼祥子','老舍','中国教育出版社',42,'你是祥子么'),
('茶馆','老舍','中国文学出版社',55,'老北京'),
('呐喊','鲁迅','中国教育出版社',70,'最后的声音'),
('围城','钱钟书','中国文学出版社',52,'你的围城是什么');

insert into book (title,author,publication,price) values
('林家铺子','矛盾','机械工业出版社',61.5),
('朝花夕拾','鲁迅','中国文学出版社',46);

-- where字句示例
select * from class_1  where score between 80 and 90;

select * from class_1  where age in (11,13,15);

select * from class_1  where sex is null;


-- 练习2 查找练习
--
-- 1. 查找30多图书
      select * from book where price >= 30 and price < 40;
-- 2. 查找中国教育出版社出版的图书
      select * from book where publication = "中国教育出版社";
-- 3. 查找老舍写的，中国文学出版社出版的
      select * from book where author = '老舍' and publication="中国文学出版社";
-- 4. 查找备注不为空的
      select * from book comment is not null;
-- 5. 查找价格查过60的，只看书名和价格
      select title,price from book where price > 60;
-- 7. 查找鲁迅写的 或者 矛盾写的
      select * from book where author ='鲁迅' or author ='矛盾';
      select * from book where author in ('鲁迅','矛盾');


-- alter 操作
alter table interest add tel char(11) after price;
alter table interest drop level;
alter table interest modify tel  char(16);
alter table interest change tel phone char(16);
alter table class_1 rename cls;

-- 时间类型
select * from marathon where birthday > '1990-01-01';
select * from marathon where registration_time < now();
select * from marathon where birthday > '1990-01-01';



练习3： 使用book表完成

1. 将呐喊的价格改为45元
update book set price=45 where title ='呐喊';

2. 增加一个字段 出版日期 类型为 date 放在价格后面
alter table book add publication_time date after price;

3. 修改所有老舍的作品出版日期为2016-10-1
update book set publication_time='2016-10-1' where author = '老舍';

4. 修改所有中国文学出版社的图书作品出版日期为2018-1-1，但是老舍的不要改
update book set publication_time='2018-1-1'
where publication='中国文学出版社' and author !='老舍';

5. 删除所有在60元以上的图书
delete from book where price > 60;

6. 修改价格字段的数据类型为 decimal(5,2)
alter table book modify price decimal(5,2);

7. 查找鲁迅写的 2017年以后出版的图书
select * from book where author='鲁迅' and publication_time >= "2017-01-01";


-- 高级查询部分示例
模糊查询
select * from cls where name like '%m%';
select * from cls where name like '___';

正则表达式查询
select * from cls where name regexp 'T.+';

as 操作

select name  姓名,age  年龄 from cls as c where c.score > 85;

排序
select * from cls order by age desc,score desc;

限制
select * from cls where sex='m' order by score desc limit 1;
update cls set score = 96 where sex='w' limit 1;

联合查询
select name,age,sex from cls where sex='w' union all select name,age,score from cls;

子查询
select * from cls where score > (select score from cls where name = 'Tony');
select 姓名,分数 from (select name as 姓名,age as 年龄,score as 分数 from cls where sex='w') as s where 分数 > 90 ;





-- 练习 4：

# 创建sanguo表
create table sanguo
(
    id      int primary key auto_increment,
    name    varchar(30),
    gender  enum ('男','女'),
    country enum ('吴','蜀','魏'),
    attack  smallint,
    defense tinyint
);

insert into sanguo
values (1, '曹操', '男', '魏', 256, 63),
       (2, '张辽', '男', '魏', 328, 69),
       (3, '甄姬', '女', '魏', 168, 34),
       (4, '夏侯渊', '男', '魏', 366, 83),
       (5, '刘备', '男', '蜀', 220, 59),
       (6, '诸葛亮', '男', '蜀', 170, 54),
       (7, '赵云', '男', '蜀', 377, 66),
       (8, '张飞', '男', '蜀', 370, 80),
       (9, '孙尚香', '女', '蜀', 249, 62),
       (10, '大乔', '女', '吴', 190, 44),
       (11, '小乔', '女', '吴', 188, 39),
       (12, '周瑜', '男', '吴', 303, 60),
       (13, '吕蒙', '男', '吴', 330, 71);

1. 查找所有蜀国人的信息，按照攻击力排名
select *
from sanguo
where country = '蜀'
order by attack desc;

2. 将赵云的攻击力设置为360 防御力设置为70
update sanguo
set attack=360,defense=70
where name = '赵云';

3. 吴国英雄攻击力超过300的改为300 （最多改2个）
update sanguo
set attack = 300
where country ='吴' and attack > 300
limit 2;

4. 查找攻击力超过200的魏国英雄名字和攻击力 并显示为 （姓名，攻击力）
select name as 姓名,attack as 攻击力
from sanguo
where attack > 200 and country='魏'；

5. 所有英雄的攻击力按照降序排序，如果攻击力相同则按照防御力降序排序
select * from sanguo  order by attack desc,defense desc;

6. 查找所有名字为3个字的英雄
select * from sanguo  where name like '___';

7. 找到比魏国最高攻击力的英雄还要高的蜀国英雄
select * from sanguo where country='蜀' and
attack > (select attack from sanguo  where country='魏' order by attack desc limit 1);

8. 找到魏国防御力前2名的英雄
select * from sanguo where country='魏' order by defense desc limit 2;

9. 查找所有女性角色，同时查找所有男性角色英雄中攻击力少于250的
select * from sanguo where gender='女'
union
select * from sanguo where gender='男' and attack < 250;

聚合操作

select avg(attack) as 平均攻击力,min(attack) as 最小攻击力 from sanguo;
select country,gender,count(*) from sanguo group by country,gender;
select country,count(id) as number from sanguo  where gender='男' group by country order by number desc limit 2;















