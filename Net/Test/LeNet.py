from Net.BaseNet.LeNet.train import train
from Config import Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from Slice.MaxSlice.MaxSlice_Resize_Zero import MaxSlice_Resize_Zero
from Slice.MaxSlice.MaxSlice_R_Z_AVG import MaxSlice_R_Z_AVG
from tensorflow.examples.tutorials.mnist import input_data


if __name__ == '__main__':
    # dataset = MaxSlice_Resize(Config)
    # dataset = MaxSlice_Resize_Zero(Config)
    dataset = MaxSlice_R_Z_AVG(Config)
    train(dataset, load_model=True)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)