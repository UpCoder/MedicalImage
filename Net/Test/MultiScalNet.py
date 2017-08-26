from Net.MyNet.MultiScaleNet.train import train
from Config import Config
from Slice.MaxSlice.MaxSlice_Multi_Scale import MaxSlice_Multi_Scale
from Slice.MaxSlice.MaxSlice_Multi_Scale_Zero import MaxSlice_Multi_Scale_Zero
from Slice.New_Max_Slice.Slice_Base_Tumor import Slice_Base_Tumor
from Slice.New_Max_Slice.Config import Config as new_config
from Slice.New_Max_Slice.Slice_Base_Liver import Slice_Base_Liver

if __name__ == '__main__':
    # dataset = MaxSlice_Multi_Scale(Config)
    # dataset = MaxSlice_Multi_Scale_Zero(Config)
    # dataset = Slice_Base_Tumor(new_config)
    dataset = Slice_Base_Liver(new_config)
    train(dataset, load_model=False)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)