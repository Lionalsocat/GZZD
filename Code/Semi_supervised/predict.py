
# 导入相关库
from keras.models import load_model
import numpy as np
import scipy.io as scio
from Code.Semi_supervised.utils import signal2pic_test
from PIL import Image
import os


# 预测函数
def predict(model_path, data_path, data_long, root_path):
    # ------------------------------------------------------------------
    # 参数：
    # mode_path: 模型保存路径
    # data_path: 数据保存路径
    # 返回：
    # clas: 类别
    # class_pro: 类别对应概率
    # -------------------------------------------------------------------

    # 加载模型，音频样本以及标签词典
    model = load_model(model_path)
    data = scio.loadmat(data_path)
    x_before = data['x'].T
    x = []
    j = 0
    while j < (x_before.size - data_long):
        x.append(x_before[0, j:j + data_long])
        j += data_long
    x = np.array(x)
    signal2pic_test(x, root_path)

    imgs = []
    test_files = [file for file in os.listdir(os.path.join(root_path, 'test')) if file.endswith('.jpg')]
    for img in test_files:
        img = np.array(Image.open(os.path.join(root_path, 'test', img)), dtype=np.float32)
        img /= 255.0
        imgs.append(img)
    imgs = np.array(imgs)

    # 预测加载的音频
    out = model.predict(imgs)
    class_num = out.shape[1]
    class_pro = []
    for i in range(class_num):
        temp = (out[:, i].sum()) / out.shape[0]
        class_pro.append(temp)
    clas = np.argmax(class_pro)

    return clas, round(class_pro[int(clas)], 2)

