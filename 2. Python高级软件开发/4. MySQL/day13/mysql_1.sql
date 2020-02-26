-- 练习1：
--
--     * 创建一个数据库books 使用utf8编码
--     * 在数据库下创建一个数据表 book
--       字段如下：
--       id  书名  作者  出版社  价格  备注
--
--     * 自己设定表的字段数据类型和约束条件，完成创建

create database books character set utf8;

use  books;

create table book (id int primary key auto_increment,
title varchar(30) not null,
author varchar(30) not null,
publication varchar(50),
price float,
comment text);

-- 插入数据操作
insert into class_1 (name,age,sex) values ('Emma',12,'w');
insert into interest values (1,'Emma','sing,dance','B+',16800.00,"骨骼惊奇，练舞奇才");
insert into interest values (2,'Lily','dance','C',8800.00,"骨骼惊奇");

