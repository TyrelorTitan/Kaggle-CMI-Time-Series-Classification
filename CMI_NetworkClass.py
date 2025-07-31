# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:47:39 2025

@author: agilj
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Network sub-blocks
class SE_Block(nn.Module):
    def __init__(self, numFeats, reduction=8, drop=0.3, atten=False):
        super(SE_Block, self).__init__()
        
        self.atten = atten
        self.fullyConn = nn.Linear(numFeats,1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
                        nn.Linear(numFeats, numFeats // reduction, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Linear(numFeats // reduction, numFeats, bias=False),
                        nn.Sigmoid()
                        )
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        # Squeeze
        if self.atten:
            scores = self.fullyConn(x)
            scores = torch.tanh(scores)
            scores = F.softmax(scores.squeeze(-1), dim=1)
            z = torch.sum(x * scores.unsqueeze(-1), dim=1)
        else:
            y = x.permute(0,2,1) # Batch, Channels, Seq
            z = self.squeeze(y).view(y.size()[0],y.size()[1])
            
        # Excitation
        z = self.excitation(z).unsqueeze(1)
        
        # Apply weights
        y = x * z
        y = self.dropout(y)
        x = y + x
        
        return x
        
class CNN_Block(nn.Module):
    def __init__(self, numFeatsIn, numFeatsOut, kSize=3):
        super().__init__()
        self.conv0 = nn.Conv1d(numFeatsIn, numFeatsOut,
                               kernel_size=kSize,
                               stride=1,
                               padding='same')
        self.conv1 = nn.Conv1d(numFeatsIn, 2*numFeatsIn, 
                               kernel_size=kSize,
                               stride=1,
                               padding='same')
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(2*numFeatsIn)
        self.SE1 = SE_Block(2*numFeatsIn)
        self.conv2 = nn.Conv1d(2*numFeatsIn, numFeatsOut,
                               kernel_size=kSize,
                               stride=1,
                               padding='same')
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(numFeatsOut)
        self.SE2 = SE_Block(numFeatsOut)

    def forward(self, x):
        x = x.permute(0,2,1)
        # For residual connection
        x_in = self.conv0(x)
        
        # Get features.
            # Block 1
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.SE1(x.permute(0,2,1)).permute(0,2,1)
            # Block 2
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.SE2(x.permute(0,2,1)).permute(0,2,1)
        
        # Residual
        x = x + x_in
        x = x.permute(0,2,1)
        
        return x
        
"""
Class to define the neural network we are going to use.
""" 
class CMI_Network_IMU(nn.Module):
    def __init__(self, numFeats=19, numClasses=18,
                       kSize=(3,1), stride=1):
        super().__init__()        
        self.thmBlock_init(numFeats=5)
        self.tofBlock_init(numFeats=320)
        
        # Batch Norm
        self.bn0 = nn.BatchNorm1d(344)
        
        # SE Block
        self.SE1 = SE_Block(19)
        
        # LSTM Stack
            # LSTM-1
        self.lstm1 = nn.LSTM(numFeats, 128, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
            # LSTM-2
        self.lstm2 = nn.LSTM(256, 128, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)
        
        nf = 832 # Number of features after fusing all sensors.
        
        # SE Block to find relevant info in the mass of features.
        self.SE2 = SE_Block(nf)

        # Attention
        self.fullyConn2 = nn.Linear(nf,1)
        
        # Now fully connected to map to number of classes.
        self.fullyConnClass = nn.Linear(nf, 
                                        numClasses)
        self.sigmoidClass = nn.ReLU(inplace=True)
        
        # Softmax
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.bn0(x)
        x = x.permute(0,2,1)
        
        # Separate data based on originating sensor.
        x_thm = x[:,:,19:19+5]
        x_tof = x[:,:,19+5:]
        x = x[:,:,:19]
        
        # Process the 'extra' data first.
        x_thm = self.thmBlock(x_thm)
        x_tof = self.tofBlock(x_tof)
        
        # SE Block
        x = self.SE1(x)
        
        # LSTM Stack
            # LSTM-1
        x, _ = self.lstm1(x)
        x = x.permute(0,2,1)
        x = self.bn1(x)
        x = x.permute(0,2,1)
        x = self.dropout2(x)
            # LSTM-2
        y, _ = self.lstm2(x)
        x = y + x
        x = x.permute(0,2,1)
        x = self.bn2(x)
        x = x.permute(0,2,1)
        x = self.dropout3(x)

        x = torch.cat((x,x_thm,x_tof),dim=2) # 256+64+512 = 832

        x = self.SE2(x)
        
        # Attention to remove the time dimension.
        scores = self.fullyConn2(x)
        scores = torch.tanh(scores)
        scores = F.softmax(scores.squeeze(-1), dim=1)
        x = torch.sum(x * scores.unsqueeze(-1), dim=1)
        
        # Fully Connected
        x = self.fullyConnClass(x)
        x = self.sigmoidClass(x)
        
        # Softmax for class probabilities
        x = self.softmax(x)
        
        return x

    def thmBlock_init(self, numFeats=5):
        # Feature expansion
        self.thm_fullyConn = nn.Linear(numFeats, 32)
        
        # SE Block
        self.thm_SE = SE_Block(32)
        
        # LSTMs
            # LSTM 1
        self.thm_lstm1 = nn.LSTM(32, 32, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.thm_bn1 = nn.BatchNorm1d(64)
        self.thm_dropout2 = nn.Dropout(0.3)
            # LSTM 2
        self.thm_lstm2 = nn.LSTM(64, 32, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.thm_bn2 = nn.BatchNorm1d(64)
        self.thm_dropout3 = nn.Dropout(0.3)
        
    def thmBlock(self, thm_x):
        # Expand number of features.
        thm_x = self.thm_fullyConn(thm_x)
        
        # Squeeze-Excitation block.
        thm_x = self.thm_SE(thm_x)
        
        # LSTM Stack
            # LSTM-1
        thm_x, _ = self.thm_lstm1(thm_x)
        thm_x = thm_x.permute(0,2,1)
        thm_x = self.thm_bn1(thm_x)
        thm_x = thm_x.permute(0,2,1)
        thm_x = self.thm_dropout2(thm_x)
            # LSTM-2
        y, _ = self.thm_lstm2(thm_x)
        thm_x = thm_x + y
        thm_x = thm_x.permute(0,2,1)
        thm_x = self.thm_bn2(thm_x)
        thm_x = thm_x.permute(0,2,1)
        thm_x = self.thm_dropout3(thm_x)    
        
        return thm_x
        
    def tofBlock_init(self, numFeats=320):
        # Idea is to reshape the data and use 2d convs over each sensor over
        # each timestep, then reshape the sensors/positions to pass them into
        # a fully connected layer, which gets features. These go into a squeeze
        # excitation block, and then into LSTMs.
        
        # Convs
        self.tof_conv1 = nn.Conv2d(5, 64, kernel_size=(3,3), padding='same')
        self.tof_nl1 = nn.ReLU()
        self.tof_pool1 = nn.MaxPool2d(kernel_size=(3,3),stride=3)
        self.tof_conv2 = nn.Conv2d(64, 128, kernel_size=(3,3), padding='same')
        self.tof_nl2 = nn.ReLU()
        self.tof_pool2 = nn.MaxPool2d(kernel_size=(3,3),stride=3)
        # Squeeze-Excitation
        self.tof_SE1 = SE_Block(128)
        # LSTM Stack
            # LSTM-1
        self.tof_lstm1 = nn.LSTM(128, 256, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.tof_bn1 = nn.BatchNorm1d(512)
        self.tof_dropout2 = nn.Dropout(0.3)
            # LSTM-2
        self.tof_lstm2 = nn.LSTM(512, 256, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.tof_bn2 = nn.BatchNorm1d(512)
        self.tof_dropout3 = nn.Dropout(0.3)
        
    def tofBlock(self, tof_x):
        # Convolutions.
        b,s,_ = tof_x.size()
        tof_x = tof_x.reshape(b*s,
                              8,
                              8,
                              5)
        tof_x = tof_x.permute(0,3,1,2) # B*S, C, W, H
        tof_x = F.pad(tof_x,(0,1,
                             0,1,
                             0,0,
                             0,0))
        tof_x = self.tof_conv1(tof_x)
        tof_x = self.tof_nl1(tof_x)
        tof_x = self.tof_pool1(tof_x)
        tof_x = self.tof_conv2(tof_x)
        tof_x = self.tof_nl2(tof_x)
        tof_x = self.tof_pool2(tof_x)
        tof_x = tof_x.permute(0,2,3,1) # B*S, W, H, C
        tof_x = tof_x.reshape(b, s, 128) # B, S, C
        
        # Squeeze-Excitation Block
        tof_x = self.tof_SE1(tof_x)
        
        # LSTMs
            # LSTM-1
        tof_x, _ = self.tof_lstm1(tof_x)
        tof_x = tof_x.permute(0,2,1)
        tof_x = self.tof_bn1(tof_x)
        tof_x = tof_x.permute(0,2,1)
        tof_x = self.tof_dropout2(tof_x)
            # LSTM-2
        y, _ = self.tof_lstm2(tof_x)
        tof_x = tof_x + y
        tof_x = tof_x.permute(0,2,1)
        tof_x = self.tof_bn2(tof_x)
        tof_x = tof_x.permute(0,2,1)
        tof_x = self.tof_dropout3(tof_x)       
        
        return tof_x
