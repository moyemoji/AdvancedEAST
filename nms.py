# coding=utf-8
import numpy as np

import cfg


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)    # isdisjoint() 方法用于判断两个集合是否包含相同的元素，如果不包含返回 True，否则返回 False


# 得到一个像素行的所有领域像素
def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor


# 合并像素行，生成像素区域
def region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D


# region_list：所有的像素行
# m：下标？
# S：全部下标
def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:    # 剩下的像素行中，如果该像素行与m相邻的话，将n添加至tmp
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))    # 递归调用，让一个块连接完全
    return rows


# predict：所有像素的预测结果
# activation_pixels：被激活的像素，即矩形内部像素
# side_vertex_pixel_threshold：边界像素阈值

def nms(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []    # 每一个元素为一行相互连接的激活像素
    for i, j in zip(activation_pixels[0], activation_pixels[1]):    # 猜测这里的0和1代表像素的行列号
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:    # 找到每一个文本行的第一个像素
            region_list.append({(i, j)})
    
    # 合并各个region_list，将属于同一个区域的像素行合并成一个文本区域
    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))    # 四维的目的是为了说明这个多边形有四个顶点能构成完整的四边形
    score_list = np.zeros((len(D), 4))

    # 遍历所有的文本区域
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        # 遍历文本区域的每一行
        for row in group:
            # ij为每一行像素的行列号
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 1]                       # 是边界像素的得分
                if score >= threshold:                                 # 如果得分大于边界阈值的话
                    ith_score = predict[ij[0], ij[1], 2:3]             # 第三个参数即[头/尾]得分，如果ith_score位于[0.1,0.9]之外的话，这里能够保证四边形有四个顶点
                    if not (cfg.trunc_threshold <= ith_score < 1 -
                            cfg.trunc_threshold):
                        ith = int(np.around(ith_score))                # 头尾得分参数四舍五入后取整，要么是0，要么是1
                        total_score[ith * 2:(ith + 1) * 2] += score    # 如果为0的话，头部两个顶点得分加分，如果为1，尾部的两个顶点得分加分
                        px = (ij[1] + 0.5) * cfg.pixel_size
                        py = (ij[0] + 0.5) * cfg.pixel_size
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7],
                                              (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v    # 这是什么操作？
        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
    return score_list, quad_list
