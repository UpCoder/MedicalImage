from Net.MyNet.RNN.train import train
from Config import Config
from Slice.MaxSlice.MaxSlice_Resize import MaxSlice_Resize
from Slice.MaxSlice.MaxSlice_Resize_Zero import MaxSlice_Resize_Zero
from Slice.MaxSlice.MaxSlice_R_Z_AVG import MaxSlice_R_Z_AVG
from tensorflow.examples.tutorials.mnist import input_data
from Slice.MaxSlice_Liver.MaxSlice_Liver_Resize import MaxSlice_Liver_Resize
from Slice.MaxSlice_Liver.MaxSlice_Liver_Resize_Zero import MaxSlice_Liver_Resize_Zero
from Slice.New_Max_Slice.Slice_Base_Tumor import Slice_Base_Tumor
from Slice.New_Max_Slice.Config import Config as new_config
from Slice.New_Max_Slice.Slice_Base_Liver import Slice_Base_Liver
from Slice.New_Max_Slice.Slice_Base_Liver_Tumor import Slice_Base_Liver_Tumor
from Slice.New_Max_Slice.Slice_Base_Liver_Tumor import Liver_Tumor_Operations

if __name__ == '__main__':
    # dataset = MaxSlice_Resize(Config)
    # dataset = MaxSlice_Resize_Zero(Config)
    # dataset = MaxSlice_R_Z_AVG(Config)
    # dataset = MaxSlice_Liver_Resize(Config)
    # dataset = MaxSlice_Liver_Resize_Zero(Config)
    # dataset = Slice_Base_Tumor(new_config)
    # dataset = Slice_Base_Liver(new_config)
    dataset = Slice_Base_Liver_Tumor(new_config, Liver_Tumor_Operations.tumor_linear_enhancement, size=[45, 45])
    # dataset = Slice_Base_Tumor(new_config)
    train(dataset, load_model=False)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)