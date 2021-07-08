import numpy as np
import scipy.io as scio
import os
from PIL import Image
from Code.Supervised.model import MobileNetv2
from Code.Supervised.utils import signal2pic_test


def predict(model_path, data_path, data_length, num_classes, code_root):

    model = MobileNetv2((416, 416, 3), num_classes)
    model.load_weights(model_path)
    data = scio.loadmat(data_path)
    x_before = data['x'].T
    x = []
    j = 0
    while j < (x_before.size - data_length):
        x.append(x_before[0, j:j+data_length])
        j += data_length
    x = np.array(x)
    signal2pic_test(x, save_path=os.path.join(code_root, 'test'))

    imgs = []
    test_files = [file for file in os.listdir(os.path.join(code_root, 'test')) if file.endswith('.jpg')]
    for img in test_files:
        img = np.array(Image.open(os.path.join(os.path.join(code_root, 'test'), img)), dtype=np.float32)
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



