import os.path
import os
import random
from shutil import copy

from sympy.logic.algorithms.dpll import pl_true_int_repr


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

# 获取data文件夹下所有文件夹名（即需要分类的类名）
file_path = 'data_cat_dog'
flower_class = [cla for cla in os.listdir(file_path)]

# 创建 训练集train 文件夹，并由类名在其目录下创建5个子目录
mkfile('data/train')
for cla in flower_class:
    mkfile('data/train/' + cla)

# 创建 验证集val 文件夹，并由类名在其目录下创建子目录
mkfile('data/test')
for cla in flower_class:
    mkfile('data/test/' + cla)

# 划分比例，训练集:测试集 = 9:1
split_rate = 0.1

# 遍历所有类别的全部图像并按比例分成训练集和验证集
for cla in flower_class:
    cla_path = file_path + '/' + cla + '/'  # 某一类别的子目录
    images = os.listdir(cla_path)
    num = len(images)
    eval_index = random.sample(images, k=int(num * split_rate))
    for index, image in enumerate(images):
        # eval_index保存验证集val的图像名称
        if image in eval_index:
            image_path = cla_path + image
            new_path = 'data/test/' + cla
            copy(image_path, new_path)

        # 其余的图像保存在训练集train中
        else:
            image_path = cla_path + image
            new_path = 'data/train/' + cla
            copy(image_path, new_path)
        print("\r[{}] processing [{}/{}]".format(cla, index + 1, num), end="")
    print()

print("processing done!")




