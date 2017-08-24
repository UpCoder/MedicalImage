from Net.MyNet.MultiScaleNet.train import train
from Config import Config
from Slice.MaxSlice.MaxSlice_Multi_Scale import MaxSlice_Multi_Scale
from Slice.MaxSlice.MaxSlice_Multi_Scale_Zero import MaxSlice_Multi_Scale_Zero


if __name__ == '__main__':
    # dataset = MaxSlice_Multi_Scale(Config)
    dataset = MaxSlice_Multi_Scale_Zero(Config)
    train(dataset, load_model=False)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)