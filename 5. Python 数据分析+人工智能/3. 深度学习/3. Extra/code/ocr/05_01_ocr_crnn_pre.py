# 预处理数据，将其转化为标准格式。同时将数据拆分成两份，以便训练和计算预估准确率
import codecs
import os
import random
import shutil
from PIL import Image

train_ratio = 9 / 10  # 训练集大小
all_file_dir = "data/data6927/word-recognition"  # 数据文件路径
image_path_pre = os.path.join(all_file_dir, "imageSet")  # 路径

# 创建训练集路径
train_image_dir = os.path.join(all_file_dir, "trainImageSet")
if not os.path.exists(train_image_dir):
    os.makedirs(train_image_dir)

eval_image_dir = os.path.join(all_file_dir, "evalImageSet")
if not os.path.exists(eval_image_dir):
    os.makedirs(eval_image_dir)

train_file = codecs.open(os.path.join(all_file_dir, "train.txt"), 'w')
eval_file = codecs.open(os.path.join(all_file_dir, "eval.txt"), 'w')
label_list = os.path.join(all_file_dir, "image_label.txt")  # 标签文件

train_count = 0
eval_count = 0
class_set = set()
with open(label_list) as f:
    for line in f:
        parts = line.strip().split()
        file, label = parts[0], parts[1]
        if '/' in label or '\'' in label or '.' in label or '!' in label or '-' in label or '$' in label or '&' in label or '@' in label or '?' in label or '%' in label or '(' in label or ')' in label or '~' in label:
            continue
        for e in label:
            class_set.add(e)
        if random.uniform(0, 1) <= train_ratio:
            shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(train_image_dir, file))
            train_file.write("{0}\t{1}\n".format(os.path.join(train_image_dir, file), label))
            train_count += 1
        else:
            shutil.copyfile(os.path.join(image_path_pre, file), os.path.join(eval_image_dir, file))
            eval_file.write("{0}\t{1}\n".format(os.path.join(eval_image_dir, file), label))
            eval_count += 1

print("train image count: {0} eval image count: {1}".format(train_count, eval_count))
class_list = list(class_set)
class_list.sort()
print("class num: {0}".format(len(class_list)))
print(class_list)
with codecs.open(os.path.join(all_file_dir, "label_list.txt"), "w") as label_list:
    label_id = 0
    for c in class_list:
        label_list.write("{0}\t{1}\n".format(c, label_id))
        label_id += 1