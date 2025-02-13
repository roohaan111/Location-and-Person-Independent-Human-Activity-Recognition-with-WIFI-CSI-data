import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        
        # normalize
        x = (x - 42.3199)/4.9802
        
        # sampling: 2000 -> 500
        x = x[:,::4]
        x = x.reshape(3, 114, 500)
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.FloatTensor(x)

        return x,y


class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')
        
        # normalize
        x = (x - 0.0025)/0.0119
        
        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)

        return x,y


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
    

class HumanID_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]        
        try:
            mat_contents = scipy.io.loadmat(sample_dir)
            # Assuming the data is stored under a specific key, e.g., 'data'
            # You need to replace 'data' with the actual key used in your .mat files
            x = mat_contents['CSIamp']  # Replace 'data' with the correct key
        except KeyError:
            raise KeyError(f"'data' key not found in {sample_dir}. Available keys: {mat_contents.keys()}")
        except Exception as e:
            raise RuntimeError(f"Error loading {sample_dir}: {e}")
        
        # Ensure x is a numpy array and flatten if necessary
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        else:
            raise TypeError(f"Data loaded from {sample_dir} is not a numpy array.")
        
        x = (x - 0.0025) / 0.0119
        
        # Reshape data: Ensure the target shape matches the data's size
        # For example, if original shape is (342, 2000), we can reshape to (1, 342, 2000)
        # Change this to the shape you actually need.
        try:
            x = x.reshape(1, 342, 2000)
        except ValueError as e:
            raise ValueError(f"Reshape error for file {sample_dir}: {e}")
        
        # Convert to torch tensor
        x = torch.from_numpy(x)

        
        return x, y