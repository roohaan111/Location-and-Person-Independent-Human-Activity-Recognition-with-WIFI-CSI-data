import torch
import torch.nn as nn
import torch.nn.functional as F


class HumanID_RecognitionModel(nn.Module):
    def __init__(self):
        super(HumanID_RecognitionModel, self).__init__()
        
        # Starting shape: (N, 1, 342, 2000)
        
        # Use a convolution with stride to reduce width and height quickly
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        # After conv1: (N,32, ceil(342/2), ceil(2000/2)) = (N,32,171,1000)
        # Pool again
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool1: (N,32,85,500)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # After conv2: (N,64, ceil(85/2), ceil(500/2)) = (N,64,43,250)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # After pool2: (N,64,21,125)  (rounding down)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # After conv3: (N,128, ceil(21/2), ceil(125/2)) = (N,128,11,63)
        
        # At this point, we still have a fairly large feature map (11x63).
        # Use adaptive pooling to force it to a fixed size (e.g., 5x5):
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5,5))
        # After adaptive_pool: (N,128,5,5)
        
        self.fc1 = nn.Linear(128 * 5 * 5, 256)  # (128*5*5=3200)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))    # (N,32,171,1000)
        x = self.pool1(x)            # (N,32,85,500)
        
        x = F.relu(self.conv2(x))    # (N,64,43,250)
        x = self.pool2(x)            # (N,64,21,125)
        
        x = F.relu(self.conv3(x))    # (N,128,11,63)
        x = self.adaptive_pool(x)    # (N,128,5,5)
        
        x = x.view(x.size(0), -1)    # (N,3200)
        x = F.relu(self.fc1(x))      # (N,256)
        return x
    

class HumanID_StateMachineModel(nn.Module):
    def __init__(self, sequence_length=10, feature_dim=256, num_states=64):
        super(HumanID_StateMachineModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # After pool, sequence_length=10 becomes 5
        # Final shape after conv2: (N,128,5)
        self.fc1 = nn.Linear(128 * 5, num_states)

    def forward(self, x):
        # x shape: (N, 256, 10)
        x = F.relu(self.conv1(x))   # (N,64,10)
        x = self.pool(x)            # (N,64,5)
        x = F.relu(self.conv2(x))   # (N,128,5)
        
        x = x.view(x.size(0), -1)   # (N,640)
        x = self.fc1(x)             # (N,num_states=64)
        return x