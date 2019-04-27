
from PIL import Image  # 导入图像处理模块
import numpy as np
import paddle  # 导入paddle模块


train_pic = paddle.dataset.mnist.train()

train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=1), batch_size=5)



def saveImg(id,data):

    data2 = np.reshape(data[0], (28, 28)).astype(np.uint8)
    new_im = Image.fromarray(data2)
    new_im.save("./pics/"+str(id)+".jpg")
count = 0
for id,data in enumerate(train_pic()):
    if count == 10:
        break
    saveImg(id, data)
    count = count+1




