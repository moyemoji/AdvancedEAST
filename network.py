# coding=utf-8
from keras import Input, Model
from keras.applications.vgg16 import VGG16
from keras.layers import Concatenate, Conv2D, UpSampling2D, BatchNormalization

import cfg

"""
input_shape=(img.height, img.width, 3), height and width must scaled by 32.
So images's height and width need to be pre-processed to the nearest num that
scaled by 32.And the annotations xy need to be scaled by the same ratio 
as height and width respectively.
"""


class East:

    def __init__(self):
        # 初始化网络时，定义输入图片，vgg16结构，定义各层的feature map为self.f
        self.input_img = Input(name='input_img',
                               shape=(None, None, cfg.num_channels),
                               dtype='float32')
        vgg16 = VGG16(input_tensor=self.input_img,
                      weights='imagenet',
                      include_top=False)
        if cfg.locked_layers:
            # locked first two conv layers
            locked_layers = [vgg16.get_layer('block1_conv1'),
                             vgg16.get_layer('block1_conv2')]
            for layer in locked_layers:
                layer.trainable = False
        
        # vgg16.get_layer(block5_pool).output
        # vgg16.get_layer(block4_pool).output
        # vgg16.get_layer(block3_pool).output
        # vgg16.get_layer(block2_pool).output
        # self.f = [block5_pool, block4_pool, block3_pool, block2_pool]

        self.f = [vgg16.get_layer('block%d_pool' % i).output
                  for i in cfg.feature_layers_range]

        # self.f = [None, block5_pool,block4_pool, block3_pool, block2_pool]
        self.f.insert(0, None)
        self.diff = cfg.feature_layers_range[0] - cfg.feature_layers_num    # 5-4=1

    def g(self, i):
        # i+diff in cfg.feature_layers_range
        # 如果i + diff不在特征层范围，报错退出
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == cfg.feature_layers_num:
            bn = BatchNormalization()(self.h(i))    # 这是最终的结果，最终结果先BatchNormalization，再卷积一次
            return Conv2D(32, 3, activation='relu', padding='same')(bn)    
        else:
            return UpSampling2D((2, 2))(self.h(i))    # 中间过程的上采样

    def h(self, i):
        # i+diff in cfg.feature_layers_range
        # 如果i + diff不在特征层范围，报错退出
        assert i + self.diff in cfg.feature_layers_range, \
            ('i=%d+diff=%d not in ' % (i, self.diff)) + \
            str(cfg.feature_layers_range)
        if i == 1:
            return self.f[i]
        else:
            concat = Concatenate(axis=-1)([self.g(i - 1), self.f[i]])    # 将第5层的特征上采样，与第4层特征合并，一次类推
            bn1 = BatchNormalization()(concat)                           # 合并的图层进行BN_1
            conv_1 = Conv2D(128 // 2 ** (i - 2), 1,                      # 对BN_1进行卷积，卷积核个数为128 // 2 ** (i - 2)，尺寸为1，激活函数relu，边界补0（padding=SAME）
                            activation='relu', padding='same',)(bn1)
            bn2 = BatchNormalization()(conv_1)                           # 对卷积后的图层进行BN_2
            conv_3 = Conv2D(128 // 2 ** (i - 2), 3,                      # 对BN_2进行卷积
                            activation='relu', padding='same',)(bn2)
            return conv_3

    def east_network(self):
        # feature_layers_num = 4  总共四个特征层
        # h和g函数采用堆栈结构实现了特征的merge，最终得到了综合特征
        before_output = self.g(cfg.feature_layers_num)
        inside_score = Conv2D(1, 1, padding='same', name='inside_score'        # 是否为内部像素 
                              )(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code'     # 是否为头尾像素，是头还是尾
                             )(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord'   # 两个顶点的坐标(x1,y1,x2,y2)
                              )(before_output)
        east_detect = Concatenate(axis=-1,
                                  name='east_detect')([inside_score,
                                                       side_v_code,
                                                       side_v_coord])
        return Model(inputs=self.input_img, outputs=east_detect)


if __name__ == '__main__':
    east = East()
    east_network = east.east_network()
    east_network.summary()
