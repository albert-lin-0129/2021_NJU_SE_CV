import os

from PIL import Image

from facenet import Facenet

gallery_path_list = [f"./data/gallery/{i}.jpg" for i in range(0, 50)]

def facePredict(pic_dir, output_path):
    print(f"[INFO]开始扫描{pic_dir}的文件")
    output_message = ""
    model = Facenet()
    for filename in os.listdir(pic_dir):
        lowest_distance = 100
        lowest_index = -1
        pic_path = pic_dir + "/" + filename
        for i in range(50):
            image_1 = Image.open(pic_path)
            image_2 = Image.open(gallery_path_list[i])
            probability = model.detect_image(image_1, image_2)
            if probability < lowest_distance:
                lowest_distance = probability
                lowest_index = i
        output_message += f"{filename} {lowest_index}\n"
        print(f"[INFO]完成{filename} {lowest_index}的匹配")
    with open(output_path, "w") as f:
        f.write(output_message)
    print(f"[INFO]全部完成，结果输出到了{output_path}中")


if __name__ == '__main__':

    test_dir = "./data/test"
    output_test_dir = "./test1.txt"
    val_dir = "./data/val"
    output_val_dir = "./val1.txt"
    facePredict(val_dir, output_val_dir)
    facePredict(test_dir, output_test_dir)
