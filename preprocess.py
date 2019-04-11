import numpy as np
from PIL import Image, ImageDraw
import os
import random
from tqdm import tqdm

import cfg
from label import shrink


def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


# 顶点重排序，确保所有的box顶点顺序一致
def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)    # 生成一个一样维数的元素均为0的矩阵
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered = np.argsort(xy_list, axis=0)       # 按列排序，(x, y)
    xmin1_index = ordered[0, 0]                 # 拿到最小的x的index
    xmin2_index = ordered[1, 0]                 # 拿到次小的x的index

    # 找到x最小的点为起始点
    # 如果有一样的x，则y更小的为起始点，左（x）上（y）
    if xy_list[xmin1_index, 0] == xy_list[xmin2_index, 0]:
        if xy_list[xmin1_index, 1] <= xy_list[xmin2_index, 1]:
            reorder_xy_list[0] = xy_list[xmin1_index]
            first_v = xmin1_index
        else:
            reorder_xy_list[0] = xy_list[xmin2_index]
            first_v = xmin2_index
    else:
        reorder_xy_list[0] = xy_list[xmin1_index]
        first_v = xmin1_index
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    others = list(range(4))
    others.remove(first_v)    # 从四个坐标中删去起始点坐标
    
    # k存储的是（其他点的y - 第一个点的y）/ （其他点的x - 第一个点的x） +  epsilon
    # 也就是各个点与起始点的直线的斜率
    k = np.zeros((len(others),))
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                    / (xy_list[index, 0] - xy_list[first_v, 0] + cfg.epsilon)
    
    # 找到起始点的对角点？
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0

    # 四个点为逆时针排列
    # box直立时，四个点顺序为左上、左下、右下、右上
    # 直立box逆时针旋转一定角度时，四个点顺序为最左、最下、最右、最上
    # 直立box顺时针旋转一定角度时，四个点顺序为最左、最下、最右、最上
    for index, i in zip(others, range(len(others))):
        # delta = y - (k * x + b)
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]

    # 对比k13 和k24，决定最终的顺序
    # compare slope of 13 and 24, determine the final order
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + cfg.epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


# 缩放图片
def resize_image(im, max_img_size=cfg.max_train_img_size):
    im_width = np.minimum(im.size[0], max_img_size)
    if im_width == max_img_size < im.size[0]:
        im_height = int((float(im_width) / im.size[0]) * im.size[1])
    else:
        im_height = im.size[1]
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((float(o_height) / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def preprocess():
    data_dir = cfg.data_dir
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
    # 如果训练图片文件夹不存在，创建他
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    draw_gt_quad = cfg.draw_gt_quad
    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)
    # 创建显示gt的图片文件夹
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    # 创建显示act图片文件夹
    show_act_image_dir = os.path.join(cfg.data_dir, cfg.show_act_image_dir_name)
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    # 列出origin文件夹下的所有图片
    o_img_list = os.listdir(origin_image_dir)
    print('found %d origin images.' % len(o_img_list))
    train_val_set = []
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):
        with Image.open(os.path.join(origin_image_dir, o_img_fname)) as im:
            # d_wight, d_height = resize_image(im)
            # 根据cfg中设置的图片尺寸，求图片缩放比率
            d_wight, d_height = cfg.max_train_img_size, cfg.max_train_img_size
            scale_ratio_w = d_wight / im.size[0]
            scale_ratio_h = d_height / im.size[1]
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
            show_gt_im = im.copy()
            # 在图片上绘制直线，这里先创建一个Draw对象
            draw = ImageDraw.Draw(show_gt_im)
            with open(os.path.join(origin_txt_dir,
                                   o_img_fname[:-4] + '.txt'), 'r') as f:
                anno_list = f.readlines()

            # anno_list中的元素逐个输出时的结果
            # 48.45,231.83,17.87,178.79,179.84,11.1,228.79,47.95,时尚袋袋
            # 228.97,527.45,228.97,576.4,479.64,575.4,473.64,524.45,百事可乐

            xy_list_array = np.zeros((len(anno_list), 4, 2))     # xy_list_array存储的是一张图片所有box的四个vertex坐标
            # 读取一张图片的gt文本文件，在图片绘制出
            for anno, i in zip(anno_list, range(len(anno_list))):
                anno_colums = anno.strip().split(',')
                anno_array = np.array(anno_colums)
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))

                # xy_list = [
                # [48.45,231.83],
                # [17.87,178.79],
                # [179.84,11.1],
                # [228.79,47.95]
                # ]
                 
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w    # 缩放了，所以w坐标位置发生了变化
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h    # 缩放了，所以h坐标位置发生了变化
                xy_list = reorder_vertexes(xy_list)              # 为四边形顶点重新排序，调整后的顺序为左上、左下、右下、右上
                xy_list_array[i] = xy_list
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)     # 如果是一个正立矩形，长宽内缩尺寸均为原矩形短边长度的0.2
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)    # shrink_1缩了x方向，y方向仍是原来的
                if draw_gt_quad:
                    draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                               tuple(xy_list[2]), tuple(xy_list[3]),
                               tuple(xy_list[0])
                               ],
                              width=2, fill='green')
                    draw.line([tuple(shrink_xy_list[0]),
                               tuple(shrink_xy_list[1]),
                               tuple(shrink_xy_list[2]),
                               tuple(shrink_xy_list[3]),
                               tuple(shrink_xy_list[0])
                               ],
                              width=2, fill='blue')
                    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                    for q_th in range(2):
                        draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                                   tuple(shrink_1[vs[long_edge][q_th][1]]),
                                   tuple(shrink_1[vs[long_edge][q_th][2]]),
                                   tuple(xy_list[vs[long_edge][q_th][3]]),
                                   tuple(xy_list[vs[long_edge][q_th][4]])],
                                  width=3, fill='yellow')
            if cfg.gen_origin_img:
                im.save(os.path.join(train_image_dir, o_img_fname))    # 保存仅仅进行了缩放的图片到指定的训练数据保存文件夹
            np.save(os.path.join(
                train_label_dir,
                o_img_fname[:-4] + '.npy'),
                xy_list_array)    # 保存重排序（因为进行了缩放啊）后的四个顶点坐标们
            if draw_gt_quad:
                show_gt_im.save(os.path.join(show_gt_image_dir, o_img_fname))  # 把绘制了线条的图保存到指定的文件夹
            train_val_set.append('{},{},{}\n'.format(o_img_fname,  # 训练数据又添一员，先保存到内存中，待最后统一写入到一个文件中啊
                                                     d_wight,
                                                     d_height))

    train_img_list = os.listdir(train_image_dir)
    print('found %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    random.shuffle(train_val_set)    # 打乱所有的训练样本
    val_count = int(cfg.validation_split_ratio * len(train_val_set))    # 按照训练、验证比例求得序号
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:     # 序号前的为验证样本
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train: # 序号后的训练样本
        f_train.writelines(train_val_set[val_count:])


if __name__ == '__main__':
    preprocess()
