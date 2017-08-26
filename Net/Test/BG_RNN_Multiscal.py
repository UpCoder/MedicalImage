from Net.MyNet.BG_RNN_MultiScale.train import train
from Config import Config
from Slice.MaxSlice.MaxSlice_Multi_Scale import MaxSlice_Multi_Scale
from Slice.MaxSlice.MaxSlice_Multi_Scale_Zero import MaxSlice_Multi_Scale_Zero
from Slice.New_Max_Slice.Slice_Base_Tumor import Slice_Base_Tumor
from Slice.New_Max_Slice.Using_BG.Config import Config as new_config
from Slice.New_Max_Slice.Using_BG.Slice_Base_Tumor_Liver import SliceBaseTumorLiverDataset, Liver_Tumor_Operations

if __name__ == '__main__':
    # dataset = MaxSlice_Multi_Scale(Config)
    # dataset = MaxSlice_Multi_Scale_Zero(Config)
    # dataset = Slice_Base_Tumor(new_config)
    dataset = SliceBaseTumorLiverDataset(new_config, operation=Liver_Tumor_Operations.tumor_linear_enhancement)
    train(dataset, load_model=False)
    # mnist = input_data.read_data_sets("../data", one_hot=True)
    # train(mnist)