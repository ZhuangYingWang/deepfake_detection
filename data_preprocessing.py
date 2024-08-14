#!/usr/bin/env python3
import os
import shutil

# 定义基础目录
base_dir = "/home/dell/桌面/kaggle/deepfake_detection/data_phase1/phase1"

# 定义生成目录的函数
def create_directories(base_dir):
    directories = {
        'train_real': os.path.join(base_dir, 'trainset', 'real'),
        'train_fake': os.path.join(base_dir, 'trainset', 'fake'),
        'val_real': os.path.join(base_dir, 'valset', 'real'),
        'val_fake': os.path.join(base_dir, 'valset', 'fake'),
    }
    for dir_name, path in directories.items():
        os.makedirs(path, exist_ok=True)

# 定义处理数据集的函数
def process_dataset(label_path, image_dir, class_dirs):
    with open(os.path.join(base_dir, label_path), 'r') as file:
        for line in file:
            if line.strip():  # 确保不处理空行
                image_label = line.strip().split(',')
                if len(image_label) == 2:
                    image_name, target = image_label[0], int(image_label[1])
                    target_dir = class_dirs['real'] if target == 0 else class_dirs['fake']
                    src_image_path = os.path.join(base_dir, image_dir, image_name)
                    dst_image_path = os.path.join(target_dir, image_name)
                    
                    if os.path.isfile(src_image_path):
                        shutil.move(src_image_path, dst_image_path)
                    else:
                        print(f"Warning: Source file '{src_image_path}' not found. Skipping.")
                else:
                    print(f"Warning: Incorrect line format in label file: '{line.strip()}'. Skipping.")

# 创建必要的目录
create_directories(base_dir)

# 定义类别目录
class_dirs_train = {'real': os.path.join(base_dir, 'trainset', 'real'), 'fake': os.path.join(base_dir, 'trainset', 'fake')}
class_dirs_val = {'real': os.path.join(base_dir, 'valset', 'real'), 'fake': os.path.join(base_dir, 'valset', 'fake')}

# 处理训练集和验证集
def main():
    process_dataset('trainset_label.txt', 'trainset', class_dirs_train)
    process_dataset('valset_label.txt', 'valset', class_dirs_val)

if __name__ == '__main__':
    main()
