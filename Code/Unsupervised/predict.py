"""
实现对输入音频的预测，并返回识别的类别及对应概率
"""

# 导入相关库
from keras.models import load_model
import numpy as np
import scipy.io as scio
from Code.Unsupervised.train import deal_x


# 预测函数
def predict(model_path, data_path, data_length):
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
    while j < (x_before.size - data_length):
        x.append(x_before[0, j:j + data_length])
        j += data_length
    x = np.array(x)
    x = deal_x(x)

    # 预测加载的音频
    out = model.predict(x)
    class_num = out.shape[1]
    class_pro = []
    for i in range(class_num):
        temp = (out[:, i].sum()) / out.shape[0]
        class_pro.append(temp)
    clas = np.argmax(class_pro)

    return clas, round(class_pro[int(clas)], 2)

