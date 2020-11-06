import os
import random

label_path = "/data/csy/cs_zz/mislead_v1/annotations/xmls/"
img_path = "/data/csy/cs_zz/mislead_v1/images/"

label_index_list = []
for sub_dir in os.listdir(label_path):
    for label_name in os.listdir(os.path.join(label_path, sub_dir)):
        if label_name.startswith('.'):
            pass
        label_index_list.append(sub_dir + "/" + label_name.split(".")[0])

print("len label index list:", len(label_index_list))

img_index_list = []
for sub_dir in os.listdir(img_path):
    for img_name in os.listdir(os.path.join(img_path, sub_dir)):
        if img_name.startswith('.'):
            pass
        img_index_list.append(sub_dir + "/" + img_name.split(".")[0])

print("len img index list:", len(img_index_list))

print("after filter:")
for label_name in label_index_list:
    if label_name not in img_index_list:
        label_index_list.remove(label_name)

print("len label index list:", len(label_index_list))

random.shuffle(label_index_list)

with open("train.txt", 'w') as ft:
    with open("val.txt", 'w') as fv:
        for index, label in enumerate(label_index_list):
            if index > int(len(label_index_list) * 0.8):
                fv.writelines(label + "\n")
            else:
                ft.writelines(label + "\n")
