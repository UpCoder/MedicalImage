from Net.BaseNet.LeNet.train import train
from Config import Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from tensorflow.examples.tutorials.mnist import input_data

if __name__ == '__main__':
    dataset = MaxSlice_Resize(Config)
    train(dataset)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)