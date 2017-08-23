from Net.MyNet.MultiScaleNet.train import train
from Config import Config
from Slice.MaxSlice.MaxSlice_Multi_Scale import MaxSlice_Multi_Scale


if __name__ == '__main__':
    # dataset = MaxSlice_Resize(Config)
    # dataset = MaxSlice_Resize_Zero(Config)
    dataset = MaxSlice_Multi_Scale(Config)
    train(dataset, load_model=True)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)