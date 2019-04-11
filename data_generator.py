import os
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import cfg


# batch样本生成器，默认生成训练数据
def gen(batch_size=cfg.batch_size, is_val=False):
    img_h, img_w = cfg.max_train_img_size, cfg.max_train_img_size
    # x为图像矩阵，[batch_size, img_h, img_w, cfg.num_channels]
    x = np.zeros((batch_size, img_h, img_w, cfg.num_channels), dtype=np.float32)
    pixel_num_h = img_h // cfg.pixel_size
    pixel_num_w = img_w // cfg.pixel_size
    # y是啥呢？[batch_size, pixel_num_h, pixel_num_w, 7]
    y = np.zeros((batch_size, pixel_num_h, pixel_num_w, 7), dtype=np.float32)
    if is_val:
        with open(os.path.join(cfg.data_dir, cfg.val_fname), 'r') as f_val:
            f_list = f_val.readlines()
    else:
        with open(os.path.join(cfg.data_dir, cfg.train_fname), 'r') as f_train:
            f_list = f_train.readlines()
    while True:
        for i in range(batch_size):
            # 随机选择一张图，构造其访问路径
            # random gen an image name
            random_img = np.random.choice(f_list)
            img_filename = str(random_img).strip().split(',')[0]
            # load img and img anno
            img_path = os.path.join(cfg.data_dir,
                                    cfg.train_image_dir_name,
                                    img_filename)
            # 使用keras的接口读入图像，转换为矩阵
            img = image.load_img(img_path)
            img = image.img_to_array(img)
            # 图片矩阵读入x中
            x[i] = preprocess_input(img, mode='tf')
            gt_file = os.path.join(cfg.data_dir,
                                   cfg.train_label_dir_name,
                                   img_filename[:-4] + '_gt.npy')
            y[i] = np.load(gt_file)
        yield x, y
