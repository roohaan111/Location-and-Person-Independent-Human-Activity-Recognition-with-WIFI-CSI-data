# from dataset import *
# from UT_HAR_model import *
# from NTU_Fi_model import *
# from widar_model import *
# from self_supervised_model import *
# import torch

# def load_data_n_model(dataset_name, model_name, root):
#     classes = {'UT_HAR_data':7,'NTU-Fi-HumanID':14,'NTU-Fi_HAR':6,'Widar':22}
#     if dataset_name == 'UT_HAR_data':
#         print('using dataset: UT-HAR DATA')
#         data = UT_HAR_dataset(root)
#         train_set = torch.utils.data.TensorDataset(data['X_train'],data['y_train'])
#         test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))
#         train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True, drop_last=True) # drop_last=True
#         test_loader = torch.utils.data.DataLoader(test_set,batch_size=256,shuffle=False)
#         if model_name == 'MLP':
#             print("using model: MLP")
#             model = UT_HAR_MLP()
#             train_epoch = 200
#         elif model_name == 'LeNet':
#             print("using model: LeNet")
#             model = UT_HAR_LeNet()
#             train_epoch = 200 #40
#         elif model_name == 'ResNet18':
#             print("using model: ResNet18")
#             model = UT_HAR_ResNet18()
#             train_epoch = 200 #70
#         elif model_name == 'ResNet50':
#             print("using model: ResNet50")
#             model = UT_HAR_ResNet50()
#             train_epoch = 200 #100
#         elif model_name == 'ResNet101':
#             print("using model: ResNet101")
#             model = UT_HAR_ResNet101()
#             train_epoch = 200 #100
#         elif model_name == 'RNN':
#             print("using model: RNN")
#             model = UT_HAR_RNN()
#             train_epoch = 3000
#         elif model_name == 'GRU':
#             print("using model: GRU")
#             model = UT_HAR_GRU()
#             train_epoch = 200
#         elif model_name == 'LSTM':
#             print("using model: LSTM")
#             model = UT_HAR_LSTM()
#             train_epoch = 200
#         elif model_name == 'BiLSTM':
#             print("using model: BiLSTM")
#             model = UT_HAR_BiLSTM()
#             train_epoch = 200
#         elif model_name == 'CNN+GRU':
#             print("using model: CNN+GRU")
#             model = UT_HAR_CNN_GRU()
#             train_epoch = 200 #20
#         elif model_name == 'ViT':
#             print("using model: ViT")
#             model = UT_HAR_ViT()
#             train_epoch = 200 #100
#         return train_loader, test_loader, model, train_epoch
    
    
#     elif dataset_name == 'NTU-Fi-HumanID':
#         print('using dataset: NTU-Fi-HumanID')
#         num_classes = classes['NTU-Fi-HumanID']
#         train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/test_amp/'), batch_size=64, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/train_amp/'), batch_size=64, shuffle=False)
#         if model_name == 'MLP':
#             print("using model: MLP")
#             model = NTU_Fi_MLP(num_classes)
#             train_epoch = 50 #15
#         elif model_name == 'LeNet':
#             print("using model: LeNet")
#             model = NTU_Fi_LeNet(num_classes)
#             train_epoch = 50 #20
#         elif model_name == 'ResNet18':
#             print("using model: ResNet18")
#             model = NTU_Fi_ResNet18(num_classes)
#             train_epoch = 50 #30
#         elif model_name == 'ResNet50':
#             print("using model: ResNet50")
#             model = NTU_Fi_ResNet50(num_classes)
#             train_epoch = 50 #40
#         elif model_name == 'ResNet101':
#             print("using model: ResNet101")
#             model = NTU_Fi_ResNet101(num_classes)
#             train_epoch = 50
#         elif model_name == 'RNN':
#             print("using model: RNN")
#             model = NTU_Fi_RNN(num_classes)
#             train_epoch = 75
#         elif model_name == 'GRU':
#             print("using model: GRU")
#             model = NTU_Fi_GRU(num_classes)
#             train_epoch = 50 #40
#         elif model_name == 'LSTM':
#             print("using model: LSTM")
#             model = NTU_Fi_LSTM(num_classes)
#             train_epoch = 50
#         elif model_name == 'BiLSTM':
#             print("using model: BiLSTM")
#             model = NTU_Fi_BiLSTM(num_classes)
#             train_epoch = 50
#         elif model_name == 'CNN+GRU':
#             print("using model: CNN+GRU")
#             model = NTU_Fi_CNN_GRU(num_classes)
#             train_epoch = 200 #20
#         elif model_name == 'ViT':
#             print("using model: ViT")
#             model = NTU_Fi_ViT(num_classes=num_classes)
#             train_epoch = 50
#         return train_loader, test_loader, model, train_epoch
    
    
#     elif dataset_name == 'NTU-Fi_HAR':
#         print('using dataset: NTU-Fi_HAR')
#         num_classes = classes['NTU-Fi_HAR']
#         train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/'), batch_size=64, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/'), batch_size=64, shuffle=False)
#         if model_name == 'MLP':
#             print("using model: MLP")
#             model = NTU_Fi_MLP(num_classes)
#             train_epoch = 30 #10
#         elif model_name == 'LeNet':
#             print("using model: LeNet")
#             model = NTU_Fi_LeNet(num_classes)
#             train_epoch = 30 #10
#         elif model_name == 'ResNet18':
#             print("using model: ResNet18")
#             model = NTU_Fi_ResNet18(num_classes)
#             train_epoch = 30
#         elif model_name == 'ResNet50':
#             print("using model: ResNet50")
#             model = NTU_Fi_ResNet50(num_classes)
#             train_epoch = 30 #40
#         elif model_name == 'ResNet101':
#             print("using model: ResNet101")
#             model = NTU_Fi_ResNet101(num_classes)
#             train_epoch = 30
#         elif model_name == 'RNN':
#             print("using model: RNN")
#             model = NTU_Fi_RNN(num_classes)
#             train_epoch = 70
#         elif model_name == 'GRU':
#             print("using model: GRU")
#             model = NTU_Fi_GRU(num_classes)
#             train_epoch = 30 #20
#         elif model_name == 'LSTM':
#             print("using model: LSTM")
#             model = NTU_Fi_LSTM(num_classes)
#             train_epoch = 30 #20
#         elif model_name == 'BiLSTM':
#             print("using model: BiLSTM")
#             model = NTU_Fi_BiLSTM(num_classes)
#             train_epoch = 30 #20
#         elif model_name == 'CNN+GRU':
#             print("using model: CNN+GRU")
#             model = NTU_Fi_CNN_GRU(num_classes)
#             train_epoch = 100 #20
#         elif model_name == 'ViT':
#             print("using model: ViT")
#             model = NTU_Fi_ViT(num_classes=num_classes)
#             train_epoch = 30
#         return train_loader, test_loader, model, train_epoch

#     elif dataset_name == 'Widar':
#         print('using dataset: Widar')
#         num_classes = classes['Widar']
#         train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/train/'), batch_size=64, shuffle=True)
#         test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/test/'), batch_size=128, shuffle=False)
#         if model_name == 'MLP':
#             print("using model: MLP")
#             model = Widar_MLP(num_classes)
#             train_epoch = 30 #20
#         elif model_name == 'LeNet':
#             print("using model: LeNet")
#             model = Widar_LeNet(num_classes)
#             train_epoch = 100 #40
#         elif model_name == 'ResNet18':
#             print("using model: ResNet18")
#             model = Widar_ResNet18(num_classes)
#             train_epoch = 100
#         elif model_name == 'ResNet50':
#             print("using model: ResNet50")
#             model = Widar_ResNet50(num_classes)
#             train_epoch = 100 #40
#         elif model_name == 'ResNet101':
#             print("using model: ResNet101")
#             model = Widar_ResNet101(num_classes)
#             train_epoch = 100
#         elif model_name == 'RNN':
#             print("using model: RNN")
#             model = Widar_RNN(num_classes)
#             train_epoch = 500
#         elif model_name == 'GRU':
#             print("using model: GRU")
#             model = Widar_GRU(num_classes)
#             train_epoch = 200 
#         elif model_name == 'LSTM':
#             print("using model: LSTM")
#             model = Widar_LSTM(num_classes)
#             train_epoch = 200 #20
#         elif model_name == 'BiLSTM':
#             print("using model: BiLSTM")
#             model = Widar_BiLSTM(num_classes)
#             train_epoch = 200
#         elif model_name == 'CNN+GRU':
#             print("using model: CNN+GRU")
#             model = Widar_CNN_GRU(num_classes)
#             train_epoch = 200 #20
#         elif model_name == 'ViT':
#             print("using model: ViT")
#             model = Widar_ViT(num_classes=num_classes)
#             train_epoch = 200

#         elif model_name == 'custom_model':
#             print("using model: Custom model")
#             model = Widar_custom(num_classes)
#             train_epoch = 30

#         return train_loader, test_loader, model, train_epoch


# def load_unsupervised_data_n_model(model_name,root):
#     HAR_train_dataset=CSI_Dataset(root+'NTU-Fi_HAR/train_amp/')
#     HAR_test_dataset=CSI_Dataset(root+'NTU-Fi_HAR/test_amp/')
#     unsupervised_train_dataset = torch.utils.data.ConcatDataset([HAR_train_dataset,HAR_test_dataset])
#     unsupervised_train_loader = torch.utils.data.DataLoader(dataset=unsupervised_train_dataset, batch_size=64, shuffle=True)
#     supervised_train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root+'NTU-Fi-HumanID/test_amp/'), batch_size=64, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root+'NTU-Fi-HumanID/train_amp/'), batch_size=64, shuffle=False)
#     if model_name == 'MLP':
#         print("using model: MLP_Parrallel")
#         model = MLP_Parrallel()
#     elif model_name == 'LeNet':
#         print("using model: CNN_Parrallel")
#         model = CNN_Parrallel()
#     elif model_name == 'ResNet18':
#         print("using model: ResNet18_Parrallel")
#         model = ResNet18_Parrallel()
#     elif model_name == 'ResNet50':
#         print("using model: ResNet50_Parralle")
#         model = ResNet50_Parrallel()
#     elif model_name == 'ResNet101':
#         print("using model: ResNet101_Parrallel")
#         model = ResNet101_Parrallel()
#     elif model_name == 'RNN':
#         print("using model: RNN_Parrallel")
#         model = RNN_Parrallel()
#     elif model_name == 'GRU':
#         print("using model: GRU_Parrallel")
#         model = GRU_Parrallel()
#     elif model_name == 'LSTM':
#         print("using model: LSTM_Parrallel")
#         model = LSTM_Parrallel()
#     elif model_name == 'BiLSTM':
#         print("using model: BiLSTM_Parrallel")
#         model = BiLSTM_Parrallel()
#     elif model_name == 'CNN+GRU':
#         print("using model: CNN_GRU_Parrallel")
#         model = CNN_GRU_Parrallel()
#     elif model_name == 'ViT':
#         print("using model: ViT_Parrallel")
#         model = ViT_Parrallel()
#     return unsupervised_train_loader, supervised_train_loader, test_loader, model


from dataset import *
from widar_model import RecognitionModel, StateMachineModel
import torch
class Yousefi_Dataset(Dataset):
    def __init__(self, x_path, y_path):
        """
        Initialize the dataset with paths to X and Y data files
        Args:
            x_path (str): Path to the X data file
            y_path (str): Path to the Y data file
        """
        # Load the data
        self.x_data = np.genfromtxt(x_path, delimiter=',')
        self.y_data = np.genfromtxt(y_path, delimiter=',')
        
        # Calculate normalization parameters from your data
        self.x_mean = np.mean(self.x_data)
        self.x_std = np.std(self.x_data)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get the data point
        x = self.x_data[idx]
        y = self.y_data[idx]
        
        # Normalize the data
        x = (x - self.x_mean) / self.x_std
        
        # Reshape to match model's expected input size (22, 20, 20)
        # Take the first 22*20*20 = 8800 values and reshape
        x = x[:8800].reshape(22, 20, 20)
        
        # Convert to tensor
        x = torch.FloatTensor(x)
        y = torch.argmax(torch.FloatTensor(y))  # Convert one-hot to index

        return x, y
    

def load_data_n_model(dataset_name, model_name, root):
    classes = {'UT_HAR_data':7, 'NTU-Fi-HumanID':14, 'NTU-Fi_HAR':6, 'Widar':22,"Yousefi":7}
    
    if dataset_name == 'Widar':
        print('Using dataset: Widar')
        # Load Widar dataset
        num_classes = classes['Widar']
        train_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/train/'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=Widar_Dataset(root + 'Widardata/test/'), batch_size=128, shuffle=False)
        print("size:", len(train_loader))
        
        # Initialize models if model_name is custom_model for Recognition and State Machine
        if model_name == 'custom_model':
            print("Using custom model with RecognitionModel and StateMachineModel")
            recognition_model = RecognitionModel()
            state_machine_model = StateMachineModel(sequence_length=10, feature_dim=256, num_states=22)
            train_epoch = 10  # Set desired number of training epochs
            return train_loader, test_loader, (recognition_model, state_machine_model), train_epoch
        else:
            raise ValueError(f"Model '{model_name}' not supported for Widar dataset in this context.")
    if dataset_name == 'Yousefi':
        print('Using dataset: Yousefi')
        # Load modified Widar dataset
        num_classes = classes['Yousefi']
        
        train_dataset = Yousefi_Dataset(
            x_path=root + 'Yousefi/train/X.csv',
            y_path=root + 'Yousefi/train/Y.csv'
        )
        test_dataset = Yousefi_Dataset(
            x_path=root + 'Yousefi/test/X.csv',
            y_path=root + 'Yousefi/test/Y.csv'
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, 
            batch_size=64, 
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, 
            batch_size=128, 
            shuffle=False
        )
        
        if model_name == 'custom_model':
            print("Using custom model with RecognitionModel and StateMachineModel")
            recognition_model = RecognitionModel()
            state_machine_model = StateMachineModel(sequence_length=10, feature_dim=256, num_states=22)
            train_epoch = 25
            return train_loader, test_loader, (recognition_model, state_machine_model), train_epoch
        else:
            raise ValueError(f"Model '{model_name}' not supported for Yousefi dataset in this context.")

    # If the dataset is not Widar, raise an exception or handle other datasets as necessary
    raise ValueError(f"Dataset '{dataset_name}' is not supported by this setup.")
