import os

train_task_id = '3T736'    # train任务id
initial_epoch = 0          # 起始epoch
epoch_num = 24             # 总共训练epoch数
lr = 1e-3                  # 学习率
decay = 5e-4               # 学习率衰减率
# clipvalue = 0.5  # default 0.5, 0 means no clip

patience = 5
load_weights = False                   # 是否加载权值
lambda_inside_score_loss = 4.0         # 几个loss的lambda，内部得分loss
lambda_side_vertex_code_loss = 1.0     # 几个loss的lambda，边界得分loss
lambda_side_vertex_coord_loss = 1.0    # 几个loss的lambda，角点得分loss

total_img = 10000                                            # 图片总数                      
validation_split_ratio = 0.1                                 # 验证图片比例
max_train_img_size = int(train_task_id[-3:])                 # 训练图像的尺寸为train id的后三位数
max_predict_img_size = int(train_task_id[-3:])  # 2400       # 最大的预测图像尺寸为train id的后三位数

# 与其让它在运行最崩溃，不如在出现错误条件时就崩溃，这时候就需要assert断言的帮助
# assert 表达式, 参数
# if not 表达式: 参数
assert max_train_img_size in [256, 384, 512, 640, 736], \
    'max_train_img_size must in [256, 384, 512, 640, 736]'

# 为不同尺寸的训练图像指定不同的batch_size
if max_train_img_size == 256:
    batch_size = 8
elif max_train_img_size == 384:
    batch_size = 4
elif max_train_img_size == 512:
    batch_size = 2
else:
    batch_size = 1

# 每一个epoch的迭代总次数等于训练图片除以batch的大小，双线除号表示取整
steps_per_epoch = total_img * (1 - validation_split_ratio) // batch_size
validation_steps = total_img * validation_split_ratio // batch_size

data_dir = 'icpr/'                                                 # 数据文件夹
origin_image_dir_name = 'image_10000/'                             # 原始image文件夹
origin_txt_dir_name = 'txt_10000/'                                 # 原始gt文件夹
train_image_dir_name = 'images_%s/' % train_task_id                # 训练任务的image文件夹
train_label_dir_name = 'labels_%s/' % train_task_id                # 训练任务的label文件夹
show_gt_image_dir_name = 'show_gt_images_%s/' % train_task_id      # 展示gt图片的文件夹
show_act_image_dir_name = 'show_act_images_%s/' % train_task_id    # 展示act图片的文件夹
gen_origin_img = True                                              # 是否产生原始图像
draw_gt_quad = True                                                # 是否绘制gt
draw_act_quad = True                                               # 是否绘制act
val_fname = 'val_%s.txt' % train_task_id                           # 保存验证数据gt信息的文本
train_fname = 'train_%s.txt' % train_task_id                       # 保存训练数据gt信息的文本
# in paper it's 0.3, maybe to large to this problem
shrink_ratio = 0.2                                                 # 文本框的收缩比率
# pixels between 0.2 and 0.6 are side pixels
shrink_side_ratio = 0.6                                            # 左右边框的收缩比率
epsilon = 1e-4

num_channels = 3                                                   # 图像通道数
feature_layers_range = range(5, 1, -1)                             # 特征层的范围 [5、4、3、2]
# feature_layers_range = range(3, 0, -1)
feature_layers_num = len(feature_layers_range)                     # 特征层的数量
# pixel_size = 4
pixel_size = 2 ** feature_layers_range[-1]                         # 像素尺寸等于2的feature_layers_range的最后一个元素次方，即2**2 = 4
print(pixel_size)
locked_layers = False

# 创建保存模型的文件夹
if not os.path.exists('model'):
    os.mkdir('model')
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')

# 训练过程的模型保存路径
model_weights_path = 'model/weights_%s.{epoch:03d}-{val_loss:.3f}.h5' \
                     % train_task_id
# 最终模型文件及权值的保存位置
saved_model_file_path = 'saved_model/east_model_%s.h5' % train_task_id
saved_model_weights_file_path = 'saved_model/east_model_weights_%s.h5'\
                                % train_task_id

pixel_threshold = 0.9                       # 像素的阈值？
side_vertex_pixel_threshold = 0.9           # 边界顶点像素阈值？
trunc_threshold = 0.1                       # 截断阈值？
predict_cut_text_line = False               # 是否预测被切割的文本行？
predict_write2txt = True                    # 是否将结果保存为txt
