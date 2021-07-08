import scipy.io as scio
import numpy as np
import os


def convert(from_name_path, to_name_path):
    from_file_folders = [tmp for tmp in os.listdir(from_name_path) if os.path.isdir(os.path.join(from_name_path, tmp))]
    for file_floder in from_file_folders:
        if not os.path.exists(os.path.join(to_name_path, file_floder)):
            os.makedirs(os.path.join(to_name_path, file_floder))
        files = [tmp for tmp in os.listdir(os.path.join(from_name_path, file_floder)) if tmp.endswith('.mat')]
        for file in files:
            file_path = os.path.join(from_name_path, file_floder, file)
            to_path = os.path.join(to_name_path, file_floder, file)
            data = scio.loadmat(file_path)
            x = data['x'][:, 0]
            x_fft = np.fft.fft(x)
            x_fft = x_fft.reshape(len(x_fft), 1)
            scio.savemat(to_path, {'x': x_fft})
    print('done!')


# convert('./data_shi', './data_pin')

