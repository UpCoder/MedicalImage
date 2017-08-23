from Net.BaseNet.LeNet.train import train
from Config import Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from Slice.MaxSlice.MaxSlice_Resize_Zero import MaxSlice_Resize_Zero
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    # dataset = MaxSlice_Resize(Config)
    dataset = MaxSlice_Resize_Zero(Config)
    train(dataset)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)