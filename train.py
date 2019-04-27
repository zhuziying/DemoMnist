import os
from PIL import Image # 导入图像处理模块
import matplotlib.pyplot as plt
import numpy
import paddle # 导入paddle模块
import paddle.fluid as fluid

# 一个minibatch中有64个数据
BATCH_SIZE = 64

# 每次读取训练集中的500个数据并随机打乱，传入batched reader中，batched reader 每次 yield 64个数据
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)
# 读取测试集的数据，每次 yield 64个数据
test_reader = paddle.batch(
    paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

label = fluid.layers.data(name='label', shape=[1], dtype='int64')

img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')

# 以softmax为激活函数的全连接层，输出层的大小必须为数字的个数10
predict = fluid.layers.fc(
    input=img, size=10, act='softmax')

# 使用类交叉熵函数计算predict和label之间的损失函数
cost = fluid.layers.cross_entropy(input=predict, label=label)
# 计算平均损失
avg_cost = fluid.layers.mean(cost)
# 计算分类准确率
acc = fluid.layers.accuracy(input=predict, label=label)

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_cost)

use_cuda = False  # 如想使用GPU，请设置为 True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
main_program = fluid.default_main_program()
test_program = fluid.default_main_program().clone(for_test=True)

def train_program():


    PASS_NUM = 5  # 训练5轮
    epochs = [epoch_id for epoch_id in range(PASS_NUM)]

    # 将模型参数存储在名为 save_dirname 的文件中
    save_dirname = "recognize_digits.inference.model"



    lists = []
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(main_program,
                              feed=feeder.feed(data),
                              fetch_list=[avg_cost, acc])
            if step % 100 == 0:  # 每训练100次 打印一次log
                print("Pass %d, Batch %d, Cost %f" % (step, epoch_id, metrics[0]))

            step += 1

            # 测试每个epoch的分类效果
        avg_loss_val, acc_val = train_test(train_test_program=test_program,
                                               train_test_reader=test_reader,
                                               train_test_feed=feeder)
        lists.append((epoch_id, avg_loss_val, acc_val))

        # 保存训练好的模型参数用于预测
        if save_dirname is not None:
            fluid.io.save_inference_model(save_dirname,
                                          ["img"], [predict], exe,
                                          model_filename=None,
                                          params_filename=None)

    # 选择效果最好的pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
    print('Th-----------------------------------------e classification accuracy is %.2f%%' % (float(best[2]) * 100))

def train_test(train_test_program,
                   train_test_feed, train_test_reader):

    # 将分类准确率存储在acc_set中
    acc_set = []
    # 将平均损失存储在avg_loss_set中
    avg_loss_set = []
    # 将测试 reader yield 出的每一个数据传入网络中进行训练
    for test_data in train_test_reader():
        acc_np, avg_loss_np = exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=[acc, avg_cost])
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
    # 获得测试数据上的准确率和损失值
    acc_val_mean = numpy.array(acc_set).mean()
    avg_loss_val_mean = numpy.array(avg_loss_set).mean()
    # 返回平均损失值，平均准确率
    return avg_loss_val_mean, acc_val_mean
train_program()