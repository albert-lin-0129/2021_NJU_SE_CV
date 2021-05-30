# author: ShuoTu
# date: 2021/05/19

import json
import os

str_train_json_path = "./data/train.json"
str_val_json_path = "./data/val.json"

str_train_txt_path = "./data/train/labels/"
str_val_txt_path = "./data/val/labels/"

list_whole_category = []


def create_txt(str_train_path, str_txt_path, output=False):
    with open(str_train_path) as file_json:
        dict_result = json.load(file_json)
        for pic_name, pic_info in dict_result.items():
            int_height = pic_info['height']
            int_width = pic_info['width']
            int_depth = pic_info['depth']
            dict_objects = pic_info['objects']
            list_obj_info = []
            for dict_obj in dict_objects.values():
                str_category = dict_obj['category']
                if str_category not in list_whole_category:
                    list_whole_category.append(str_category)
                int_index = list_whole_category.index(str_category)
                float_width = (float(dict_obj['bbox'][2]) - float(dict_obj['bbox'][0])) / int_width
                float_height = (float(dict_obj['bbox'][3]) - float(dict_obj['bbox'][1])) / int_height
                float_x = (float(dict_obj['bbox'][2]) + float(dict_obj['bbox'][0])) / int_width / 2
                float_y = (float(dict_obj['bbox'][3]) + float(dict_obj['bbox'][1])) / int_height / 2
                list_obj_info.append([int_index, float_x, float_y, float_width, float_height])
            str_output = ""
            for list_single_obj_attr in list_obj_info:
                str_output += f"{list_single_obj_attr[0]:<4d} {list_single_obj_attr[1]:<.6f} {list_single_obj_attr[2]:<.6f} {list_single_obj_attr[3]:<.6f} {list_single_obj_attr[4]:<.6f}\n"
            # print(str_output)
            if output:
                with open(str_txt_path + f"{pic_name.split('.')[0]}.txt", 'w') as file_txt:
                    file_txt.write(str_output)
                print(f"[INFO]已处理{pic_name}")


if __name__ == '__main__':
    create_txt(str_train_json_path, str_train_txt_path, True)
    create_txt(str_val_json_path, str_val_txt_path, True)
    print(list_whole_category)
