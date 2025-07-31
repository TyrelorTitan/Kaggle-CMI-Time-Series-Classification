# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:01:33 2025

@author: agilj
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import glob
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_warmup as warmup

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.transform import Rotation as R

import colorednoise as cn

# Define dataset
class CMI_Dataset(Dataset):
    def __init__(self, dataframe_path, demographicPath, 
                 indSet=None, device='cuda:0', subjToUse=None,
                 mean=None, std=None):
        self.device = device
        self.remove=False
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.orientations = [
            'Lie on Back',
            'Lie on Side - Non Dominant',
            'Seated Lean Non Dom - FACE DOWN',
            'Seated Straight'
        ]
        
        self.classes = self.target_gestures + self.non_target_gestures

        print('Processing Dataframe.')
        # No demographics to start with
        self.hasDemographics = False
        
        # Data for training.
        self.dataframe = pd.read_csv(dataframe_path)
        self.demographics = pd.read_csv(demographicPath)
        
        # Get the sequences (each seq is a data point).
        self.seqs = self.dataframe['sequence_id'].unique()
        
        # Pare down to subset of sequences.
        if subjToUse is None:
            self.dataframe = self.dataframe[self.dataframe['sequence_id']\
                                            .isin(self.seqs[indSet])]
        else:
            self.subjs = list(self.dataframe['subject'].unique())
            if subjToUse[0]=='REMOVE': # Use all but specified
                self.remove=True
                for sub in subjToUse[1:]:
                    self.subjs.remove(sub)
                self.dataframe = self.dataframe[self.dataframe['subject']\
                                                .isin(self.subjs)]
            else: # Use only specified
                self.dataframe = self.dataframe[self.dataframe['subject']\
                                                .isin(subjToUse)]
        
        # Correct for gravity on Z-IMU
        quats = self.dataframe[['rot_x','rot_y','rot_z','rot_w']]
        na = quats.isna().T.all()
        na = na.loc[na==True]
            # Remove data without rotation measurements.
        quats.loc[na.index,'rot_w'] = 1
        quats.loc[na.index,'rot_x'] = 0
        quats.loc[na.index,'rot_y'] = 0
        quats.loc[na.index,'rot_z'] = 0

        self.dataframe.loc[na.index,'rot_w'] = 1
        self.dataframe.loc[na.index,'rot_x'] = 0
        self.dataframe.loc[na.index,'rot_y'] = 0
        self.dataframe.loc[na.index,'rot_z'] = 0           
        # Rotate gravity vector into sensor coordinate system.
        rotation = R.from_quat(quats)
        gravity = np.ones((self.dataframe.shape[0],3)) * np.array([0,0,9.80665])
        gravity_sensor_frame = rotation.apply(gravity, inverse=True)
            # Remove gravity
        self.dataframe.loc[self.dataframe.index,['acc_x','acc_y','acc_z']] = \
            self.dataframe.loc[self.dataframe.index,
                               ['acc_x','acc_y','acc_z']]-gravity_sensor_frame
        
        # Specify the numerical features for classification.
        excludedCols = {
            'gesture', 'sequence_type', 'behavior', 'orientation',  # train-only
            'row_id', 'subject', 'phase',  # metadata
            'sequence_id', 'sequence_counter'}  # identifiers
        featuresUsed = [c for c in self.dataframe.columns \
                        if c not in excludedCols]
        self.allFeatures = featuresUsed
        self.imuFeatures = \
            [c for c in self.dataframe.columns if 'acc' in c or 'rot' in c]
        self.ptsPerSeq = 150 # For interpolating.
        
        self.featsToUse = self.imuFeatures
        
        # Quickly get mean and std.
        if mean is None:
            self.dfMean = self.dataframe[self.featsToUse].mean()
            self.dfMean = np.expand_dims(self.dfMean,axis=0)
            self.dfMean = torch.Tensor(self.dfMean).to('cuda:0')
        else:
            self.dfMean = mean
        if std is None:
            self.dfStd = self.dataframe[self.featsToUse].std()
            self.dfStd = np.expand_dims(self.dfStd,axis=0)
            self.dfStd = torch.Tensor(self.dfStd).to('cuda:0')
        else:
            self.dfStd = std
            
        # Now restrict the data used to only those points in the target set.
        # We train on all data here, so it is commented out.
        # self.dataframe = self.dataframe[self.dataframe['gesture']\
        #                                 .isin(self.target_gestures)]
        
        # Reset index.
        self.dataframe = self.dataframe.reset_index(drop=True)
        
        self.seqs = self.dataframe['sequence_id'].unique()

        # Get mapping between class labels and numbers.
        # self.numToClass = self.target_gestures + self.non_target_gestures
        self.numToClass = self.classes
        self.classToNum = {}
        for ind,c in enumerate(self.numToClass):    
            self.classToNum[c] = ind
        
        # Make list of sequences.
        self.Xset = []
        self.Yset = []
        self.Dset = []
        self.Tset = []
        self.Fset = []
        print('Building Dataset...')
        thmFeatNames = [c for c in self.dataframe.columns if 'thm' in c]       
        tofFeatNames = [c for c in self.dataframe.columns if 'tof' in c]       

        for seqInd, seq in enumerate(self.seqs):
            seqData = self.dataframe.loc[self.dataframe['sequence_id']==seq]

            # Get label
            try:
                tmp = self.classToNum[seqData['gesture'].iloc[0] + \
                                      ' - ' + \
                                      seqData['orientation'].iloc[0]]
            except:
                tmp = self.classToNum[seqData['gesture'].iloc[0]]
            # tmp = self.classToNum[seqData['gesture'].iloc[0]]
            truth = torch.LongTensor([tmp]).to(self.device)
            self.Yset.append(truth)
            
            # Now sort data by truth value.
            tmp = seqData[self.featsToUse]
            tmp = np.array(tmp)
            tmp = self._adjustSize(tmp)

            tmp = torch.Tensor(tmp).to(self.device)
            self.Xset.append(tmp)
                
            # ...Demographics
            # Get subject
            subj = seqData['subject'].iloc[0]
            # Correlate with list of subjects
            tmp = self.demographics.loc[self.demographics['subject']==subj]
            # Get values.
            info = np.array(tmp.values[0,1:],dtype=float)
            # Save.
            self.Dset.append(torch.Tensor(info).to(self.device))
            
            # ...Thermopile
            # Get features
            tmp = seqData[thmFeatNames]
            # Fill NA values with 0.
            tmp = np.array(tmp.fillna(0))
            # Pad
            tmp = self._adjustSize(tmp)
            # Append
            tmp = torch.Tensor(tmp).to(self.device)
            self.Tset.append(tmp)
            
            # ...Time of Flight
            # Get features
            tmp = seqData[tofFeatNames]
            # Fill NA values with 0.
            tmp = np.array(tmp.fillna(0))
            # Pad
            tmp = self._adjustSize(tmp)
            # Append
            tmp = torch.Tensor(tmp).to(self.device)
            self.Fset.append(tmp)
            
            if ((seqInd % 100) == 0) and (seqInd!=0):
                print(str(seqInd)+'/'+str(len(self.seqs))+\
                      ' datapoints completed.')
                    
        self.hasDemographics = True

    
    def __len__(self):
        return len(self.Xset)
    
    def __getitem__(self, idx):
        
        data = self.Xset[idx]
        truth = self.Yset[idx]
        demos = self.Dset[idx]
        thm = self.Tset[idx]
        tof = self.Fset[idx]
        
        data = torch.unsqueeze(data, dim=0)
        
        return data, truth, demos, thm, tof

    def _adjustSize(self, data):
        if data.shape[0] > self.ptsPerSeq:
            data = data[:self.ptsPerSeq,:]
        elif data.shape[0] < self.ptsPerSeq:
            padAmt = self.ptsPerSeq - data.shape[0]
            data = np.pad(data,((0,padAmt),(0,0)))
        return data

print('Cell Complete')

"""
Utility class to handle augmentations, data scaling, and feature extraction.
"""
class featureMaker():
    def __init__(self, seed=618567):
        self.rng = np.random.default_rng(seed)
        
    def processData(self, x, demo, tof=None, thm=None):
        last = torch.argmax(torch.all(x==0,dim=2).to(int),dim=1)
        last[last==0] = 149
        
        # Augment
        x, thm, tof = self.augment(x, last, tof=tof, thm=thm)
        
        # Scale. Found height-scaling does not improve recall.
        # x = self.scale(x, demo)
        
        # Adjust handedness. Found this does not improve recall.
        # x = self.adjustHandedness(x, demo)
        
        # Produce downstream features
        x = self.makeFeatures(x, last)
        
        return x, thm, tof
    
    def addPinkNoise(self, x, last):
        min_snr = 25
        max_snr = 30
        snr = np.random.uniform(min_snr, max_snr)
        a_signal = torch.sqrt((x ** 2).max(dim=0).values)  # shape: (N,)
        a_noise = a_signal / (10 ** (snr / 20))   # shape: (N,)
        pink_noise = torch.stack([
            torch.Tensor(cn.powerlaw_psd_gaussian(1, 
                                                  x.size()[0] * x.size()[2])
                         ).to('cuda:0') for _ in range(x.size()[1])], dim=1)
        pink_noise = pink_noise.reshape(x.size()[0],x.size()[2],x.size()[1])
        pink_noise = pink_noise.permute(0,2,1)
        a_pink = torch.sqrt((pink_noise ** 2).max(axis=0).values)  # shape: (N,)
        pink_noise_normalized = pink_noise * (a_noise / a_pink)
        augmented = (x + pink_noise_normalized).to(x.dtype)
        return augmented
    
    def timestretch(self, x, last, thm=None, tof=None):
        rate = torch.Tensor(rng.uniform(0.9, 1.1, (x.size()[0],))).to('cuda:0')

        L_new = (last / rate).to(int)
        L_new[L_new>=150] = 149
    
        xnew = [F.interpolate(x[i,:last[i],:].unsqueeze(0).permute(0,2,1),
                           L_new[i]).permute(0,2,1)[:,:150,:] \
             for i in range(x.size()[0])]
        x2 = torch.stack([F.pad(x,(0,0,0,150-x.size()[1],0,0)) for x in xnew])
        augmented = x2[:,0,:,:]

        if thm is not None:
            thmnew = [F.interpolate(thm[i,:last[i],:].unsqueeze(0).permute(0,2,1),
                               L_new[i]).permute(0,2,1)[:,:150,:] \
                 for i in range(thm.size()[0])]
            thm2 = torch.stack([F.pad(thm,(0,0,0,150-thm.size()[1],0,0)) \
                                for thm in thmnew])
            thm_aug = thm2[:,0,:,:]


        if tof is not None:
            tofnew = [F.interpolate(tof[i,:last[i],:].unsqueeze(0).permute(0,2,1),
                               L_new[i]).permute(0,2,1)[:,:150,:] \
                 for i in range(tof.size()[0])]
            tof2 = torch.stack([F.pad(tof,(0,0,0,150-tof.size()[1],0,0)) \
                                for tof in tofnew])
            tof_aug = tof2[:,0,:,:]
                
        if (tof is None) and (thm is None):
            return augmented, L_new
        elif (tof is None) and (thm is not None):
            return augmented, L_new, thm_aug
        elif (tof is not None) and (thm is None):
            return augmented, L_new, tof_aug
        elif (tof is not None) and (thm is not None):
            return augmented, L_new, thm_aug, tof_aug
            
    def drift(self, x, last):
        # Model this as a random walk.
        toZero = torch.tile(torch.all(x==0,dim=2).unsqueeze(2),(1,1,7))\
            .to('cuda:0')
        # Linear
            # Possible directions. Each row has unit 2-norm.
        moves = torch.Tensor([[0,0,0],
                              [1,0,0],
                              [0,1,0],
                              [0,0,1],
                              [1/np.sqrt(2),1/np.sqrt(2),0],
                              [1/np.sqrt(2),-1/np.sqrt(2),0],
                              [1/np.sqrt(2),0,1/np.sqrt(2)],
                              [1/np.sqrt(2),0,-1/np.sqrt(2)],
                              [0,1/np.sqrt(2),1/np.sqrt(2)],
                              [0,1/np.sqrt(2),-1/np.sqrt(2)],
                              [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
                              [-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
                              [1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)],
                              [1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)],
                              ]
                             ).to('cuda:0')
            # Select one direction per timestep
        moveOptions = torch.Tensor(self.rng.integers(0,14,size=(x.size()[0],
                                                               x.size()[1]))
                                   ).to('cuda:0')
            # Get the size of each step as a percentage of the measured
            # acceleration.
        stepSizes = torch.normal(0,torch.abs(x[:,:,:3])/10)
            # Now multiply and sum to get the displacement vector.
        steps = moves[moveOptions.flatten().to(int),:].reshape(x.size()[0],
                                                               x.size()[1],
                                                               3)
        steps = steps * stepSizes
        steps = torch.cumsum(steps,dim=1) # This is our displacement vector.
            # Take two derivatives to get the acceleration.
        steps_acc = torch.zeros_like(steps)
        steps_acc[:,2:,:] = torch.diff(torch.diff(steps,dim=1),dim=1)
        # Zero out border
        steps_acc[:,0,:] = 0
        steps_acc[:,1,:] = 0
        steps_acc[torch.arange(len(last)).unsqueeze(1), 
                  last.unsqueeze(1)] = 0
        steps_acc[torch.arange(len(last)).unsqueeze(1), 
                  last.unsqueeze(1)-1] = 0
        x[:,:,:3] = x[:,:,:3] + steps
        
            # Rotation (44 options)
        moves = torch.Tensor([[0,0,0,0],
                              [1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,1],
                              
                              [1/np.sqrt(2),1/np.sqrt(2),0,0],
                              [1/np.sqrt(2),-1/np.sqrt(2),0,0],
                              
                              [1/np.sqrt(2),0,1/np.sqrt(2),0],
                              [1/np.sqrt(2),0,-1/np.sqrt(2),0],
                              
                              [1/np.sqrt(2),0,0,1/np.sqrt(2)],
                              [1/np.sqrt(2),0,0,-1/np.sqrt(2)],
                              
                              [0,1/np.sqrt(2),1/np.sqrt(2),0],
                              [0,1/np.sqrt(2),-1/np.sqrt(2),0],
                              
                              [0,1/np.sqrt(2),0,1/np.sqrt(2)],
                              [0,1/np.sqrt(2),0,-1/np.sqrt(2)],
                              
                              [0,0,1/np.sqrt(2),1/np.sqrt(2)],
                              [0,0,1/np.sqrt(2),-1/np.sqrt(2)],
                              
                              [0,1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
                              [0,-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3)],
                              [0,1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3)],
                              [0,1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3)],

                              [1/np.sqrt(3),0,1/np.sqrt(3),1/np.sqrt(3)],
                              [-1/np.sqrt(3),0,1/np.sqrt(3),1/np.sqrt(3)],
                              [1/np.sqrt(3),0,-1/np.sqrt(3),1/np.sqrt(3)],
                              [1/np.sqrt(3),0,1/np.sqrt(3),-1/np.sqrt(3)],
                              
                              [1/np.sqrt(3),1/np.sqrt(3),0,1/np.sqrt(3)],
                              [-1/np.sqrt(3),1/np.sqrt(3),0,1/np.sqrt(3)],
                              [1/np.sqrt(3),-1/np.sqrt(3),0,1/np.sqrt(3)],
                              [1/np.sqrt(3),1/np.sqrt(3),0,-1/np.sqrt(3)],
                              
                              [1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),0],
                              [-1/np.sqrt(3),1/np.sqrt(3),1/np.sqrt(3),0],
                              [1/np.sqrt(3),-1/np.sqrt(3),1/np.sqrt(3),0],
                              [1/np.sqrt(3),1/np.sqrt(3),-1/np.sqrt(3),0],
                              
                              [1/2,1/2,1/2,1/2],
                              [-1/2,1/2,1/2,1/2],
                              [1/2,-1/2,1/2,1/2],
                              [1/2,1/2,-1/2,1/2],
                              [1/2,1/2,1/2,-1/2],
                              
                              [-1/2,-1/2,1/2,1/2],
                              [-1/2,1/2,-1/2,1/2],
                              [-1/2,1/2,1/2,-1/2],
                              [1/2,-1/2,-1/2,1/2],
                              [1/2,-1/2,1/2,-1/2],
                              [1/2,1/2,-1/2,-1/2]
                              ]
                             ).to('cuda:0')
        
            # Select one direction per timestep
        moveOptions = torch.Tensor(self.rng.integers(0,44,size=(x.size()[0],
                                                               x.size()[1]))
                                   ).to('cuda:0')
            # Get the size of each step
        stepSizes = torch.normal(0., 0.005, 
                                 size=(x.size()[0],x.size()[1],4))\
            .to( 'cuda:0')
            # Now multiply and sum to get the displacement vector. We can sum
            # the rotations because the step sizes are so small.
        steps = moves[moveOptions.flatten().to(int),:].reshape(x.size()[0],
                                                               x.size()[1],
                                                               4)
        steps = steps * stepSizes
            # We iteratively construct the new rotations by finding what the
            # change in rotation would have been, then we add on the 
            # displacement for that timestep and reproject onto the unit 
            # sphere.
        diffs = torch.zeros_like(x[:,:,3:])
        diffs[:,1:,:] = x[:,1:,3:] - x[:,:-1,3:]
        x2 = torch.zeros_like(x[:,:,3:])
        x2[:,0,:] = x[:,0,3:]
        for i in range(149):
            x2[:,i+1,:] = x2[:,i,:] + diffs[:,i+1,:] + steps[:,i+1,:]
            x2[:,i+1,:] = x2[:,i+1,:] / torch.norm(x2[:,i+1,:].unsqueeze(1),
                                                   dim=2)
        x[:,:,3:] = x2
        
        x[toZero] = 0
        
        return x
        
    def makeFeatures(self, x, last):
        x = self._getAngularVel(x, last)
        x = self._getAngularAcc(x, last)
        x = self._getAngularJerk(x, last)
        x = self._getSpatialJerk(x, last)
        x = self._getScalarFeatures(x, last)
        
        return x
    
    def adjustHandedness(self, x, demo):
        # Get indices where subject is left handed
        if torch.any(demo[:,3]==0):
            ex_toAdj = demo[:,3]==0
            # Flip x-axis
            x[ex_toAdj,:,0] = x[ex_toAdj,:,0] * -1
            # Get quats.
            quats = x[ex_toAdj,:,3:]
            b,l,_ = quats.size()
            quats = quats.view(b*l, 4)
            ref = torch.all(quats==torch.Tensor([0,0,0,0]).to('cuda:0'),dim=1)
            quats[ref,:] = torch.Tensor([1,0,0,0]).to('cuda:0')
            
            # Invert y-axis and z-axis rotations
            quats = R.from_quat(quats.cpu())
            rotvecs = R.as_rotvec(quats)
            rotvecs = rotvecs * [1,-1,-1]
            quats = R.from_rotvec(rotvecs)
            quats = torch.Tensor(R.as_quat(quats)).to('cuda:0')
            quats = quats.view(b,l,4)
            
            x[ex_toAdj,:,3:] = quats
        
        return x
    
    def scale(self, x, demo):
        x[:,:,:3] = x[:,:,:3] / demo[:,4].unsqueeze(1).unsqueeze(1)
        
        return x
    
    def augment(self, x, last, thm=None, tof=None):
        # Pink Noise
        noiseAug = self.rng.uniform(0,1,size=(x.size()[0]))
        if (noiseAug<0.1).any():
            x[noiseAug<0.5,:,:] = self.addPinkNoise(x[noiseAug<0.5,:,:],
                                                    last[noiseAug<0.5])
        # Time Stretching
        shiftAug = self.rng.uniform(0,1,size=(x.size()[0]))
        if (shiftAug<0.5).any():
            if (thm is not None) and (tof is not None):
                x[shiftAug<0.5,:,:], \
                last[shiftAug<0.5], \
                thm[shiftAug<0.5,:,:], \
                tof[shiftAug<0.5,:,:] = \
                    self.timestretch(x[shiftAug<0.5,:,:], 
                                     last[shiftAug<0.5],
                                     thm=thm[shiftAug<0.5,:,:],
                                     tof=tof[shiftAug<0.5,:,:])
            elif (thm is not None) and (tof is None):
                x[shiftAug<0.5,:,:], \
                last[shiftAug<0.5], \
                thm[shiftAug<0.5,:,:]= \
                    self.timestretch(x[shiftAug<0.5,:,:], 
                                     last[shiftAug<0.5],
                                     thm=thm[shiftAug<0.5,:,:])
            elif (thm is None) and (tof is not None):
                x[shiftAug<0.5,:,:], \
                last[shiftAug<0.5], \
                tof[shiftAug<0.5,:,:] = \
                    self.timestretch(x[shiftAug<0.5,:,:], 
                                     last[shiftAug<0.5],
                                     tof=tof[shiftAug<0.5,:,:])
            elif (thm is None) and (tof is None):
                x[shiftAug<0.5,:,:], \
                last[shiftAug<0.5] = \
                    self.timestretch(x[shiftAug<0.5,:,:], 
                                     last[shiftAug<0.5])
                    
        # # Random Walk Drift in Acc and Rot
        # # Found that this does not improve generalization, so removed it.
        # driftAug = self.rng.uniform(0,1,size=(x.size()[0]))
        # if (driftAug<0.5).any():
        #     x[driftAug<0.5,:,:] = self.drift(x[driftAug<0.5,:,:],
        #                                      last[driftAug<0.5])
        
        # Use tof/thm?
        if tof is not None:
            useExtra = self.rng.uniform(0,1,size=(tof.size()[0]))
            tof[useExtra<0.5,:,:] = 0
            thm[useExtra<0.5,:,:] = 0
        
        # Rot / Acc sensor dropout
        dropAug = self.rng.uniform(0,1,size=(x.size()[0]))
        if (dropAug<0.1).any():
            whichSensor = self.rng.uniform(0,1,(dropAug<0.1).astype(int).sum())
                # Acc: <0.5
            tmp = x[dropAug<0.1,:,:]
            tmp[whichSensor<0.5,:,:3] = 0
                # Rot: >=0.5
            tmp[whichSensor>=0.5,:,3:] = 0
            x[dropAug<0.1,:,:] = tmp
        
        # Zero regions.
        x[torch.arange(len(last)).unsqueeze(1), 
          last.unsqueeze(1),
          :] = 0
        thm[torch.arange(len(last)).unsqueeze(1), 
          last.unsqueeze(1),
          :] = 0
        tof[torch.arange(len(last)).unsqueeze(1), 
          last.unsqueeze(1),
          :] = 0
        
        return x, thm, tof
    
    """
    Compute change in angular position by calculating A^-1 * B, where A and B
    are rotations (elements of SO(3)), ^-1 denotes the multiplicative inverse
    over SO(3), and * is the multiplication in SO(3). This makes use of the 
    fact that rotating backward is the inverse of rotating forward, so this
    has the effect of 'subtracting' one rotation from another.
    """
    def _getAngularVel(self, x, last):        
        rots = x[:,:,3:] # Scalar last
        rots = rots[:,:,[1,2,3,0]] # Scalar First
        q1 = torch.zeros_like(rots)
        q2 = torch.zeros_like(rots)
        q1[:,1:,:] = rots[:,:-1,:]
        q2[:,1:,:] = rots[:,1:,:]
        dt = 0.1
        wx = (q1[:,:,0]*q2[:,:,1] - q1[:,:,1]*q2[:,:,0] - \
              q1[:,:,2]*q2[:,:,3] + q1[:,:,3]*q2[:,:,2]) * (2/dt)
        wy = (q1[:,:,0]*q2[:,:,2] + q1[:,:,1]*q2[:,:,3] - \
              q1[:,:,2]*q2[:,:,0] - q1[:,:,3]*q2[:,:,1]) * (2/dt)
        wz = (q1[:,:,0]*q2[:,:,3] - q1[:,:,1]*q2[:,:,2] + \
              q1[:,:,2]*q2[:,:,1] - q1[:,:,3]*q2[:,:,0]) * (2/dt)
            
        # Zero out border
        wx[:,0] = 0
        wx[:,-1] = 0 # Probably already 0, but fixes an edge case.
        wx[torch.arange(len(last)).unsqueeze(1), last.unsqueeze(1)] = 0
        wy[:,0] = 0
        wy[:,-1] = 0
        wy[torch.arange(len(last)).unsqueeze(1), last.unsqueeze(1)] = 0
        wz[:,0] = 0
        wz[:,-1] = 0
        wz[torch.arange(len(last)).unsqueeze(1), last.unsqueeze(1)] = 0
        
        x = torch.cat((x,
                       wx.unsqueeze(2),
                       wy.unsqueeze(2),
                       wz.unsqueeze(2)),dim=2)
        
        return x
        
    def _getAngularAcc(self, x, last):
        vels = x[:,:,7:]
        accs = torch.zeros_like(vels)
        accs[:,1:,:] = torch.diff(vels,dim=1)
        
        # Zero out border
        accs[:,0] = 0
        accs[:,1] = 0
        accs[torch.arange(len(last)).unsqueeze(1), last.unsqueeze(1)] = 0
        accs[torch.arange(len(last)).unsqueeze(1), last.unsqueeze(1)-1] = 0
        
        x = torch.cat((x,accs),dim=2)
        
        return x
        
    def _getAngularJerk(self, x, last):
        accs = x[:,:,10:]
        jerk = torch.zeros_like(accs)
        jerk[:,1:,:] = torch.diff(accs,dim=1)
        
        jerk[:,0] = 0
        jerk[:,1] = 0
        jerk[:,2] = 0
        jerk[torch.arange(len(last)).unsqueeze(1), last.unsqueeze(1)] = 0
        jerk[torch.arange(len(last)).unsqueeze(1), last.unsqueeze(1)-1] = 0
        jerk[torch.arange(len(last)).unsqueeze(1), last.unsqueeze(1)-2] = 0

        x = torch.cat((x, jerk), dim=2)
        
        return x
        
    """
    Partial derivative in each spatial direction.
    """
    def _getSpatialJerk(self, x, last):
        accs = x[:,:,:3]
        jerk = torch.zeros_like(accs)
        jerk[:,1:,:] = torch.diff(accs,dim=1)
        
        # Zero out border
        jerk[:,0] = 0
        jerk[:,-1] = 0
        jerk[torch.arange(len(last)).unsqueeze(1), last.unsqueeze(1)] = 0
        
        x = torch.cat((x[:,:,:3], jerk, x[:,:,3:]), dim=2)
        
        return x
    
    """
    Scalar features (energy, rotation angle)
    """
    def _getScalarFeatures(self, x, last):
        accMag = torch.sqrt(torch.sum(x[:,:,:3]**2, dim=2))
        accMag = torch.unsqueeze(accMag,dim=2)
        dAccMag = torch.zeros_like(accMag)
        dAccMag[:,1:,:] = torch.diff(accMag, dim=1)
        
        rotAngle = 2 * torch.acos(x[:,:,9])
        rotAngle[rotAngle.isnan()] = 0
        rotAngle = torch.unsqueeze(rotAngle,dim=2)
        dRotAngle = torch.zeros_like(rotAngle)
        dRotAngle[:,1:,:] = torch.diff(rotAngle, dim=1)
        
        x = torch.cat((x, accMag, dAccMag, rotAngle, dRotAngle), dim=2)
        
        return x

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
        
    
# Define neural network we want to use.
class CMI_Network_IMU(nn.Module):
    def __init__(self, numFeats=348, numClasses=33):
        super(CMI_Network_IMU, self).__init__()
        reduction = 8
        self.SEatten = False
        
        # Batch Norm
        self.bn0 = nn.BatchNorm1d(348)
        
        # Extra Feature Branches
        self.thmBlock_init(numFeats=5, numFeatsOut=10) # Out must be even
        self.tofBlock_init(numFeats=320, numFeatsOut=320) # Out must be even

        # CNN Block for IMU Data
        self.CNN1 = CNN_Block(23, 64)
        self.CNN2 = CNN_Block(64, 128)
        
        numFeats = 10+320+128 # Manually tuned.
        self.SE1 = SE_Block(numFeats)
        
        # Triplet Processing. Goal is for each branch to learn different
        # features. Attention then decides which are important.
            # LSTM
        self.lstm1 = nn.LSTM(numFeats, int(numFeats/2), num_layers=1,
                             batch_first=True, bidirectional=True)
        self.bn1 = nn.BatchNorm1d(numFeats)
        self.dropout1 = nn.Dropout(0.3)
            # 1d Conv
        self.conv1 = nn.Conv1d(numFeats, 2*numFeats, kernel_size=3,
                               stride=1, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(2*numFeats, numFeats, kernel_size=3,
                               stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(numFeats)
        self.dropout2 = nn.Dropout(0.3)
            # Dense
        self.fullyConn1 = nn.Linear(numFeats, numFeats)
        self.elu1 = nn.ELU()
        self.bn3 = nn.BatchNorm1d(numFeats)
        self.dropout3 = nn.Dropout(0.3)

        # Concatenate

        # Attention
        self.fullyConn2 = nn.Linear(3*numFeats, 1)
        self.bn4 = nn.BatchNorm1d(3*numFeats)

        # Reduce
        self.fullyConnRed1 = nn.Linear(3*numFeats, 
                                       numFeats)
        self.fullyConnRed2 = nn.Linear(numFeats,
                                       int(numFeats/2))
        
        # Now fully connected to map to number of classes.
        self.fullyConnClass = nn.Linear(int(numFeats/2), 
                                        numClasses)
        self.sigmoidClass = nn.Tanh()
        # Softmax
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):        
        x = x.permute(0,2,1)
        x = self.bn0(x)
        x = x.permute(0,2,1)
        
        x_thm = x[:,:,23:23+5]
        x_tof = x[:,:,23+5:]
        x = x[:,:,:23]
        
        # Extra Feature Processing
        x_thm = self.thmBlock(x_thm)
        # x_thm = y + x_thm
        x_tof = self.tofBlock(x_tof)
        # x_tof = y + x_tof
        
        # SE Block for IMU Data
        imuData = self.CNN1(x)
        if imuData.isnan().any():
            print('Ahh!')
        imuData = self.CNN2(imuData)
        
        # Concatenate
        x = torch.cat((imuData, x_thm, x_tof), dim=2)
    
        # SE Block
        x = self.SE1(x)
        
        # Triplet Processing
            # LSTM
        x1, _ = self.lstm1(x)
        x1 = x1 + x
        x1 = self.bn1(x1.permute(0,2,1)).permute(0,2,1)
        x1 = self.dropout1(x1)
            # 1D Conv
        x2 = self.conv1(x.permute(0,2,1)).permute(0,2,1)
        x2 = self.relu1(x2)
        x2 = self.conv2(x2.permute(0,2,1)).permute(0,2,1)
        x2 = x2 + x
        x2 = self.bn2(x2.permute(0,2,1)).permute(0,2,1)
        x2 = self.dropout2(x2)
            # Dense
        x3 = self.fullyConn1(x)
        x3 = self.elu1(x3)
        x3 = x3 + x
        x3 = self.bn3(x3.permute(0,2,1)).permute(0,2,1)
        x3 = self.dropout3(x3)
        
        # Concatenate
        x = torch.cat((x1, x2, x3), dim=2)
        
        # Attention
        scores = self.fullyConn2(x)
        scores = torch.tanh(scores)
        scores = F.softmax(scores.squeeze(-1), dim=1)
        x = torch.sum(x * scores.unsqueeze(-1), dim=1)
        x = self.bn4(x)
        
        # Reduce Dimensionality
        x = self.fullyConnRed1(x)
        x = self.fullyConnRed2(x)
        
        # Fully Connected
        x = self.fullyConnClass(x)
        x = self.sigmoidClass(x)
        # Softmax for class probabilities
        x = self.softmax(x)
        return x
    
    def thmBlock_init(self, numFeats=5, numFeatsOut=64):
        reduction = numFeats
        self.thm_fullyConn = nn.Linear(numFeats, 2*numFeats)
        # LSTMs
            # LSTM 1
        self.thm_lstm1 = nn.LSTM(2*numFeats, numFeats, num_layers=1,
                             batch_first=True, bidirectional=True)
        self.thm_bn1 = nn.BatchNorm1d(2*numFeats)
        self.thm_dropout2 = nn.Dropout(0.3)
        # SE Block
        self.thm_SE2 = SE_Block(2*numFeats)
        # Linear
        self.thm_fullyConn2 = nn.Linear(2*numFeats, numFeatsOut)

    def thmBlock(self, thm_x):
        thm_x = self.thm_fullyConn(thm_x)
        # LSTMs    
            # LSTM 1
        y, _ = self.thm_lstm1(thm_x)
        thm_x = thm_x + y
        thm_x = thm_x.permute(0,2,1)
        thm_x = self.thm_bn1(thm_x)
        thm_x = thm_x.permute(0,2,1)
        thm_x = self.thm_dropout2(thm_x)
        # SE Block
        thm_x = self.thm_SE2(thm_x)
        # Linear
        thm_x = self.thm_fullyConn2(thm_x)
        return thm_x
        
    def tofBlock_init(self, numFeats=320, numFeatsOut=512):
        # Idea is to reshape the data and use 2d convs over each sensor over
        # each timestep, then reshape the sensors/positions to pass them into
        # a fully connected layer, which gets features. These go into a squeeze
        # excitation block, and then into LSTMs.
        reduction = 8
            # Convs
        self.tof_conv1 = nn.Conv2d(5, 64, kernel_size=(3,3), padding='same')
        self.tof_nl1 = nn.Tanh()
        self.tof_pool1 = nn.MaxPool2d(kernel_size=(3,3),stride=3)
        self.tof_conv2 = nn.Conv2d(64, int(numFeatsOut/4), kernel_size=(3,3), padding='same')
        self.tof_nl2 = nn.Tanh()
        self.tof_pool2 = nn.MaxPool2d(kernel_size=(3,3),stride=3)
        
            # Squeeze-Excitation
        self.tof_SE1 = SE_Block(int(numFeatsOut/4))
        
        self.tof_fullyConn = nn.Linear(int(numFeatsOut/4), numFeatsOut)

            # LSTMs
        self.tof_lstm1 = nn.LSTM(numFeatsOut, int(numFeatsOut/2), 
                                 num_layers=1, batch_first=True, 
                                 bidirectional=True)
                                 
        self.tof_bn1 = nn.BatchNorm1d(numFeatsOut)
        self.tof_dropout2 = nn.Dropout(0.3)
            # Decoder
        self.tof_lstm2 = nn.LSTM(numFeatsOut, int(numFeatsOut/2), num_layers=1,
                             batch_first=True, bidirectional=True)
        self.tof_bn2 = nn.BatchNorm1d(numFeatsOut)
        
        self.tof_SE2 = SE_Block(numFeatsOut)
        
    def tofBlock(self, tof_x):
        # Convolutions
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
        tof_x = tof_x.reshape(b, s, tof_x.size()[3]) # B, S, C
        # Squeeze-Excitation
        tof_x = self.tof_SE1(tof_x)
        # Dense
        tof_x = self.tof_fullyConn(tof_x)
        # LSTMs
        y, _ = self.tof_lstm1(tof_x)
        tof_x = tof_x + y
        tof_x = tof_x.permute(0,2,1)
        tof_x = self.tof_bn1(tof_x)
        tof_x = tof_x.permute(0,2,1)
        tof_x = self.tof_dropout2(tof_x)
        # Decode
        y, _ = self.tof_lstm2(tof_x)
        tof_x = tof_x + y
        tof_x = tof_x.permute(0,2,1)
        tof_x = self.tof_bn2(tof_x)
        tof_x = tof_x.permute(0,2,1)
        # Squeeze-Excitation
        tof_x = self.tof_SE2(tof_x)
        return tof_x
    
#%%
if __name__ == '__main__':
    #%%
    # Specify data path.
    dataframe_path = './train.csv'
    demo_path = './train_demographics.csv'
    device = 'cuda:0'
    numFolds=5
    
    # Get sequences
    numSubj = 81
    numSeqs = 8151
    ptsPerSeq = 50
    
    # Split sequences into training and testing (say 80-10-10).
    trainSplit = 0.90
    valSplit = 0.1
    # testSplit = 1 - trainSplit - valSplit
    rng = np.random.default_rng(0)
    
    # Make training dataset.
    # Get subject names
    subjNames = pd.read_csv(demo_path)['subject']
    # Pull out set for training/testing the SVM
    svmtrain = rng.permutation(numSubj)
    svmNames = subjNames.iloc[svmtrain[:20]]
    subjNames = subjNames.iloc[svmtrain[20:]]
    svmtest = list(svmNames.iloc[:10])
    svmtrain = list(svmNames.iloc[10:])
    
    numSubj=61
    for k in range(numFolds):
        # Randomly pick 10 subjects to use as validation
        picks = rng.permutation(numSubj)[:10]
        subjToUse = list(subjNames.iloc[picks])
        subjToRemove = ['REMOVE'] + subjToUse + list(svmNames)
        print('Assembling training dataset.')
        trainData = CMI_Dataset(dataframe_path, demo_path,
                                subjToUse=subjToRemove)
    
        # Make validation dataset.
        print('Assembling validation dataset.')
        valData = CMI_Dataset(dataframe_path, demo_path,
                              subjToUse=subjToUse,
                              mean=trainData.dfMean, std=trainData.dfStd)
        
        #%%
        torch.backends.cudnn.deterministic = True
        numClasses = len(trainData.classes)
        # Add demographic info
        # Make dataloaders.
        
        batchSize = 64
        trainDataloader = DataLoader(trainData, batch_size=batchSize, 
                                     shuffle=True, num_workers=0, drop_last=False)
        valDataloader = DataLoader(valData, batch_size=batchSize, 
                                   shuffle=False, num_workers=0, drop_last=False)
        
        # Instantiate network, specify optimizer, specify loss.
        netStartingPoint = 0
        print('Building classification network.')
        net = CMI_Network_IMU(numClasses=numClasses).to(device)
        if netStartingPoint:
            net.load_state_dict(torch.load('./outputs/GClass/'+str(k)+\
                                               'model_'+\
                                               str(netStartingPoint)+\
                                               '.pth', 
                                           weights_only=True))
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim,
            T_0=100,
            T_mult=2)
        warmup_scheduler = warmup.UntunedLinearWarmup(optim)
        loss_fun = nn.CrossEntropyLoss()
        net.train()
            
        # Training loop.
        numEpochs = 100 + netStartingPoint
        numPts = len(trainDataloader.dataset)
            # Create log file.
        logfile_path = './outputs/GClass/'+str(k)+'log.txt'
        if not netStartingPoint:
            mode = 'w'
        else:
            mode = 'a'
        with open(logfile_path, mode) as f:
            f.write('Validation Loss\n')
            # Set validation loss requirement for saving.
        lastValF1 = -torch.inf
        print('Beginning Training!')
        fm = featureMaker()
        for ep in range(numEpochs):
            epMod = ep + netStartingPoint
            acc = 0
            acc_ori = 0
            lossAvg = 0
            t1 = time.time()
            for batchnum, (X, truth, 
                           demo, thm, tof) in enumerate(trainDataloader):
                truth = truth.squeeze()

                X = X[:,0,:,:]

                X, thm, tof = fm.processData(X, demo, thm=thm, tof=tof)
    
                # Concatenate extended features.
                X = torch.cat((X,thm,tof), dim=2)
                
                # Predict class
                optim.zero_grad()
                pred = net(X)
                if X.size()[0] == 1:
                    truth = truth.unsqueeze(0)
                loss = loss_fun(pred, truth)
                # Backprop
                loss.backward()
                optim.step()
                acc_ori += float((torch.argmax(pred,dim=1) == truth).sum().cpu())
                acc += float((torch.floor(torch.argmax(pred,dim=1)).to(int) \
                              == torch.floor(truth).to(int)).sum().cpu())
                lossAvg += float(loss)
                
                with warmup_scheduler.dampening():
                    pass
                lr_scheduler.step(ep + batchnum / len(trainDataloader))
                
            # Check val performance every epoch.
            lossVal1 = 0
            lossVal2 = 0
            accVal1 = 0
            accVal2 = 0
            yPredMC = np.empty((valData.seqs.shape[0],),dtype=int)
            yTrueMC = np.empty((valData.seqs.shape[0],),dtype=int)
            yPredMC2 = np.empty((valData.seqs.shape[0],),dtype=int)
            yTrueMC2 = np.empty((valData.seqs.shape[0],),dtype=int)
            net.eval()
            with torch.no_grad():
                for seqNum, (Xv, truthv, 
                             demov, thmv, tofv) in enumerate(valDataloader):
                    truthv = truthv.squeeze()
                    Xv = Xv[:,0,:,:]
                    
                    # Zero the first value in the vels and first 2 in the accs.
                    last = torch.argmax(torch.all(Xv==0,dim=2).to(int),dim=1)
                    last[last==0] = 149
                    
                    # Xv = fm.adjustHandedness(Xv, demov)
                    Xv = fm.makeFeatures(Xv, last)
                    
                    # Compute dual F1 scores.
                    Xv = torch.cat((Xv,thmv,tofv), dim=2)
    
                    # Predict class
                    predVal = net(Xv)
                    predClass = torch.argmax(predVal,dim=1)
                    if Xv.size()[0] == 1:
                        truthv = truthv.unsqueeze(0)
                    lossVal1 += float(loss_fun(predVal, truthv))
                    # accVal += float((predClass == truthv).sum().cpu())
                    truthvred = truthv
                    accVal1 += float((predClass == truthvred).sum().cpu())
                    # F1 Score Step
                    if (seqNum+1)*batchSize < len(yPredMC):
                        yPredMC[seqNum*batchSize:(seqNum+1)*batchSize] = \
                            np.array(predClass.cpu(), copy=None).astype(int)
                        yTrueMC[seqNum*batchSize:(seqNum+1)*batchSize] = \
                            np.array(truthvred.cpu(), copy=None).astype(int)
                    else:
                        yPredMC[seqNum*batchSize:] = \
                            np.array(predClass.cpu(), copy=None).astype(int)
                        yTrueMC[seqNum*batchSize:] = \
                            np.array(truthvred.cpu(), copy=None).astype(int)
    
                    # Now without extra info
                    Xv[:,:,19:] = 0
                    # Predict class
                    predVal = net(Xv)
                    predClass = torch.argmax(predVal,dim=1)
                    if Xv.size()[0] == 1:
                        truthv = truthv.unsqueeze(0)
                    lossVal2 += float(loss_fun(predVal, truthv))
                    accVal2 += float((predClass == truthv).sum().cpu())
                    # F1 Score Step
                    if (seqNum+1)*batchSize < len(yPredMC):
                        yPredMC2[seqNum*batchSize:(seqNum+1)*batchSize] = \
                            np.array(predClass.cpu(), copy=None).astype(int)
                        yTrueMC2[seqNum*batchSize:(seqNum+1)*batchSize] = \
                            np.array(truthv.cpu(), copy=None).astype(int)
                    else:
                        yPredMC2[seqNum*batchSize:] = \
                            np.array(predClass.cpu(), copy=None).astype(int)
                        yTrueMC2[seqNum*batchSize:] = \
                            np.array(truthv.cpu(), copy=None).astype(int)
                        
                lossVal = (lossVal1 + lossVal2) / (2*valData.__len__())
                accVal = (accVal1 + accVal2) / (2*valData.__len__())
                
                # Binary F1 Score.
                yTrueBin = yTrueMC<8
                yPredBin = yPredMC<8
                yTrueBin2 = yTrueMC2<8
                yPredBin2 = yPredMC2<8
    
                f1_binary1 = f1_score(
                    yTrueBin,
                    yPredBin,
                    pos_label=True,
                    zero_division=0,
                    average='binary'
                )   
                f1_binary2 = f1_score(
                    yTrueBin2,
                    yPredBin2,
                    pos_label=True,
                    zero_division=0,
                    average='binary'
                )
                f1_binary = (f1_binary1 + f1_binary2) / 2
                
                yPredMC = [x if x < 8 else 8 for x in yPredMC]
                yPredMC2 = [x if x < 8 else 8 for x in yPredMC2]
                yTrueMC = [x if x < 8 else 8 for x in yTrueMC]
                yTrueMC2 = [x if x < 8 else 8 for x in yTrueMC2]
    
                # Macro F1 Score.
                f1_macro1 = f1_score(
                    yTrueMC,
                    yPredMC,
                    average='macro',
                    zero_division=0
                )
                f1_macro2 = f1_score(
                    yTrueMC2,
                    yPredMC2,
                    average='macro',
                    zero_division=0
                )
                f1_macro = (f1_macro1 + f1_macro2) / 2
                
                f1_total = (f1_macro + f1_binary) / 2
                # Write val loss value to file.
                with open(logfile_path,'a') as f:
                    f.write(f"{lossVal:>7f}\t{accVal:>3f}\t{f1_total:>3f}\n")
                if f1_macro > lastValF1:
                    lastValF1 = f1_macro
                    for file in glob.glob("./outputs/GClass/"+str(k)+\
                                          "model_*"):
                        os.remove(file)

                    torch.save(net.state_dict(), 
                               "./outputs/GClass/"+str(k)+"model_"+\
                                   str(epMod)+".pth")
            net.train()
                
            # Print progress
            t2 = time.time()
            totalTime = t2-t1
            lossAvg = lossAvg / trainData.__len__()
            accCurr = acc / trainData.__len__()
            print(f"(Epoch {epMod:d}) Loss: {lossAvg:f} // {lossVal:f} " + \
                  f"-- Acc: {accCurr:>3f} // {accVal:>3f} " + \
                  f"-- F1: {f1_total:>3f} " + \
                  f"-- Time: {totalTime:2f}s.")

    #%% Build SVM Infrastructure
    # The goal here is that each of the cross-validation networks are likely
    # good at different things (since they have different scores), so we train
    # an SVM on *a different dataset* to use the networks' output probabilities
    # to predict the true class.
    
    from sklearn import svm
    # import xgboost as xgb

    """
    Accepts a list X, where each element in the list is a different class. Each
    list should be N_i x F where N_i is the number of examples in that class and F
    is the number of features.
    Returns a tuple (Xout,y) where Xout is an NxF ndarray, where N is the total
    number of examples in the list, and where y is a length-N array of class
    labels.
    """
    def format_SVM_data(X):
        y = []
        Xout = []
        for idx, cl in enumerate(X):
            label = idx * np.ones((cl.shape[0],))
            y.extend(label)
            Xout.extend(cl)
        Xout = np.array(Xout)
        y = np.array(y)
        return (Xout,y)
    
    # Load network ensemble.
    folder = './outputs/GClass/'
    nets = []
    for i in range(numFolds):
        tmp = CMI_Network_IMU(numClasses=numClasses).to(device)
        file = [f for f in os.listdir(folder) \
                if f.startswith(str(i)+'model')][0]
        tmp.load_state_dict(torch.load('./outputs/GClass/'+file,
                                       weights_only=True))
        tmp.eval()
        nets.append(tmp)
    
    # Build SVM datset
    svmTrainDataset = CMI_Dataset(dataframe_path, demo_path,
                            subjToUse=svmtrain)
    svmValDataset = CMI_Dataset(dataframe_path, demo_path,
                            subjToUse=svmtest)
    
    #%% Train SVM
    from scipy.stats import mode
    from sklearn import neighbors, ensemble

    print('Building SVM Dataset...')
    svmTrainData = []
    svmTrainData_imu = []
    predClasses = []
    predClasses_imu = []
    trueClasses = []
    trueClasses_imu = []
    for i in range(numClasses):
        args = [j for j in range(len(svmTrainDataset)) \
                if svmTrainDataset.Yset[j]==torch.Tensor([i]).to('cuda:0')]
        # Pass data into network ensemble.
        Xdata = torch.stack([svmTrainDataset.Xset[j] for j in args],dim=0)
        Tdata = torch.stack([svmTrainDataset.Tset[j] for j in args],dim=0)
        Fdata = torch.stack([svmTrainDataset.Fset[j] for j in args],dim=0)
        Ddata = torch.stack([svmTrainDataset.Dset[j] for j in args],dim=0)
        
        # Zero the first value in the vels and first 2 in the accs.
        last = torch.argmax(torch.all(Xdata==0,dim=2).to(int),dim=1)
        last[last==0] = 149

        Xdata = fm.makeFeatures(Xdata, last)
        
        data = torch.cat((Xdata, Tdata, Fdata), dim=2)
        
        tmp = []
        tmp_imu = []
        for j, net in enumerate(nets):
            with torch.no_grad():
                pred = net(data)
                tmp.append(pred)
                # tmp.append(torch.argmax(pred,dim=1).unsqueeze(dim=1))
                predClasses.append(torch.argmax(pred,dim=1).cpu())
                trueClasses.append(i*torch.ones_like(predClasses[-1]))
                
                # IMU Only
                data[:,:,19:] = 0
                pred = net(data)
                tmp_imu.append(pred)
                # tmp_imu.append(torch.argmax(pred,dim=1).unsqueeze(dim=1))
                predClasses_imu.append(torch.argmax(pred,dim=1).cpu())
                trueClasses_imu.append(i*torch.ones_like(predClasses_imu[-1]))
                
        tmp = np.array(torch.cat(tmp,dim=1).cpu())
        tmp_imu = np.array(torch.cat(tmp_imu,dim=1).cpu())

        svmTrainData.append(tmp)
        svmTrainData_imu.append(tmp_imu)


    # Build SVM-compatible dataset
    data_forSVM, labels_forSVM = format_SVM_data(svmTrainData)
    predClasses = np.array(torch.cat(predClasses,dim=0),dtype=int)
    predClasses = np.reshape(predClasses, (svmTrainDataset.__len__(), 
                                           numFolds))
    predClasses = mode(predClasses, axis=1)[0]
    trueClasses = np.array(torch.cat(trueClasses,dim=0),dtype=int)
        # IMU Only
    data_forSVM_imu, labels_forSVM_imu = format_SVM_data(svmTrainData_imu)
    predClasses_imu = np.array(torch.cat(predClasses_imu,dim=0),\
                               dtype=int)
    predClasses_imu = np.reshape(predClasses_imu, (svmTrainDataset.__len__(), 
                                                   numFolds))
    predClasses_imu = mode(predClasses_imu, axis=1)[0]
    trueClasses_imu = np.array(torch.cat(trueClasses_imu,dim=0),\
                               dtype=int)
    trueClasses_imu = np.reshape(trueClasses_imu, (svmTrainDataset.__len__(), 
                                                   numFolds))
    trueClasses_imu = mode(trueClasses_imu, axis=1)[0]

        
    # Train an SVM
    print('Training SVM...')
    secondary_classifier = svm.SVC(kernel='rbf',gamma='scale')
    secondary_classifier.fit(data_forSVM, labels_forSVM)
    
    secondary_classifier_imu = svm.SVC(kernel='rbf',gamma='scale')
    secondary_classifier_imu.fit(data_forSVM_imu, labels_forSVM_imu)

    # Separation?
    print('Checking SVM Separation...')
    preds = secondary_classifier.predict(data_forSVM)
    gotCorrect = preds==labels_forSVM
    acc = np.mean(gotCorrect)*100
    print('Separation: '+str(acc)+'%.')
    cm = confusion_matrix(labels_forSVM, preds)
    disp = ConfusionMatrixDisplay(cm).plot()
    disp.ax_.set_title('Separation: with Extra Features')
        # IMU Only
    print('Checking SVM Separation with only IMU Data...')
    preds = secondary_classifier_imu.predict(data_forSVM_imu)
    gotCorrect = preds==labels_forSVM_imu
    acc = np.mean(gotCorrect)*100
    print('Separation, IMU Only: '+str(acc)+'%.')
    cm = confusion_matrix(labels_forSVM_imu, preds)
    disp = ConfusionMatrixDisplay(cm).plot()
    disp.ax_.set_title('Separation: IMU Only')
        
    gotCorrect = predClasses_imu==trueClasses_imu
    accDefault = np.mean(gotCorrect)*100
    print('No SVM Separation, IMU Only: '+str(accDefault)+'%.')

    #% Validate SVM
    svmValData = []
    predClassesVal = []
    trueClassesVal = []
    
    svmValData_imu = []
    predClassesVal_imu = []
    trueClassesVal_imu = []
    sex = []
    for i in range(numClasses):
        args = [j for j in range(len(svmValDataset)) \
                if svmValDataset.Yset[j]==torch.Tensor([i]).to('cuda:0')]
        # Pass data into network ensemble.
        Xdata = torch.stack([svmValDataset.Xset[j] for j in args],dim=0)
        Tdata = torch.stack([svmValDataset.Tset[j] for j in args],dim=0)
        Fdata = torch.stack([svmValDataset.Fset[j] for j in args],dim=0)
        Ddata = torch.stack([svmValDataset.Dset[j] for j in args],dim=0)
        
        # Zero the first value in the vels and first 2 in the accs.
        last = torch.argmax(torch.all(Xdata==0,dim=2).to(int),dim=1)
        last[last==0] = 149
        
        # Xdata = fm.adjustHandedness(Xdata, Ddata)
        Xdata = fm.makeFeatures(Xdata, last)
        
        data = torch.cat((Xdata, Tdata, Fdata), dim=2)
        sex.append(Ddata[:,3])


        tmp = []
        tmp_imu = []
        for j, net in enumerate(nets):
            with torch.no_grad():
                predVal = net(data)
                tmp.append(predVal)
                # tmp.append(torch.argmax(predVal,dim=1).unsqueeze(dim=1))
                predClassesVal.append(torch.argmax(predVal,dim=1).cpu())
                trueClassesVal.append(i*torch.ones_like(predClassesVal[-1]))

                # No extra data
                data[:,:,19:] = 0
                predVal = net(data)
                tmp_imu.append(predVal)
                # tmp_imu.append(torch.argmax(predVal,dim=1).unsqueeze(dim=1))
                predClassesVal_imu.append(torch.argmax(predVal,dim=1).cpu())
                trueClassesVal_imu.append(i*torch.ones_like(predClassesVal_imu[-1]))
            
        tmp = np.array(torch.cat(tmp,dim=1).cpu())
        tmp_imu = np.array(torch.cat(tmp_imu,dim=1).cpu())
        svmValData.append(tmp)
        svmValData_imu.append(tmp_imu)

    # Build SVM-compatible dataset
    data_forSVM, labels_forSVM = format_SVM_data(svmValData)
    predClassesVal = np.array(torch.cat(predClassesVal,dim=0),\
                              dtype=int)
    predClassesVal = np.reshape(predClassesVal, (svmValDataset.__len__(), 
                                                 numFolds))
    predClassesVal = mode(predClassesVal, axis=1)[0]    
    trueClassesVal = np.array(torch.cat(trueClassesVal,dim=0),\
                              dtype=int)
    trueClassesVal = np.reshape(trueClassesVal, (svmValDataset.__len__(), 
                                                 numFolds))
    trueClassesVal = mode(trueClassesVal, axis=1)[0]
    
    data_forSVM_imu, labels_forSVM_imu = format_SVM_data(svmValData_imu)
    predClassesVal_imu = np.array(torch.cat(predClassesVal_imu,dim=0),\
                                  dtype=int)
    predClassesVal_imu = np.reshape(predClassesVal_imu, (svmValDataset.__len__(), 
                                                 numFolds))
    predClassesVal_imu = mode(predClassesVal_imu, axis=1)[0]    
    trueClassesVal_imu = np.array(torch.cat(trueClassesVal_imu,dim=0),\
                                  dtype=int)
    trueClassesVal_imu = np.reshape(trueClassesVal_imu, (svmValDataset.__len__(), 
                                                         numFolds))
    trueClassesVal_imu = mode(trueClassesVal_imu, axis=1)[0]
    
    # Validate the SVM
    print('Validating SVM with Extra Features...')
    preds = secondary_classifier.predict(data_forSVM)
    gotCorrect = preds==labels_forSVM
    acc = np.mean(gotCorrect)*100
    print('Generalization: '+str(acc)+'%.')
    gotCorrect = predClassesVal==trueClassesVal
    accDefault = np.mean(gotCorrect)*100
    print('No SVM Generalization: '+str(accDefault)+'%.')
    cm = confusion_matrix(labels_forSVM, preds)
    disp = ConfusionMatrixDisplay(cm).plot()
    disp.ax_.set_title('Generalization: with Extra Features')
    print('Validating SVM with only IMU Data...')
    preds = secondary_classifier_imu.predict(data_forSVM_imu)
    gotCorrect = preds==labels_forSVM_imu
    acc = np.mean(gotCorrect)*100
    print('Generalization with only IMU Data: '+str(acc)+'%.')
    gotCorrect = predClassesVal_imu==trueClassesVal_imu
    accDefault = np.mean(gotCorrect)*100
    print('No SVM Generalization with only IMU Data: '+str(accDefault)+'%.')
    cm = confusion_matrix(labels_forSVM_imu, preds)
    disp = ConfusionMatrixDisplay(cm).plot()
    disp.ax_.set_title('Generalization: IMU Only')

    
    #%% Binary F1 Score.
    predClassesVal_SVM = secondary_classifier.predict(data_forSVM)
    predClassesVal_imu_SVM = secondary_classifier_imu.predict(data_forSVM_imu)

    yTrueBin = labels_forSVM<8
    yPredBin = predClassesVal_SVM<8
    yTrueBin2 = labels_forSVM_imu<8
    yPredBin2 = predClassesVal_imu_SVM<8

    f1_binary1 = f1_score(
        yTrueBin,
        yPredBin,
        pos_label=True,
        zero_division=0,
        average='binary'
    )   
    f1_binary2 = f1_score(
        yTrueBin2,
        yPredBin2,
        pos_label=True,
        zero_division=0,
        average='binary'
    )
    f1_binary = (f1_binary1 + f1_binary2) / 2
    
    yPredMC = [x if x < 8 else 8 for x in predClassesVal_SVM]
    yPredMC2 = [x if x < 8 else 8 for x in predClassesVal_imu_SVM]
    yTrueMC = [x if x < 8 else 8 for x in labels_forSVM]
    yTrueMC2 = [x if x < 8 else 8 for x in labels_forSVM_imu]

    # Macro F1 Score.
    f1_macro1 = f1_score(
        yTrueMC,
        yPredMC,
        average='macro',
        zero_division=0
    )
    f1_macro2 = f1_score(
        yTrueMC2,
        yPredMC2,
        average='macro',
        zero_division=0
    )
    f1_macro = (f1_macro1 + f1_macro2) / 2
    
    f1_total = (f1_macro + f1_binary) / 2
    
    print('Generalization F1-Score is: '+str(f1_total))
    
    # Save SVMs
    # import joblib
    # joblib.dump(svm_classifier_imu,'./outputs/GClass/SVMv4_model_imu_joblib.pkl')
    # joblib.dump(svm_classifier,'./outputs/GClass/SVMv4_model_joblib.pkl')