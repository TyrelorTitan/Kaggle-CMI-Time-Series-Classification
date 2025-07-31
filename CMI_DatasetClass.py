# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:34:31 2025

@author: agilj
"""
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from scipy.spatial.transform import Rotation as R

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

    """
    Method to crop or pad the data to 150 time samples per example.
    """
    def _adjustSize(self, data):
        if data.shape[0] > self.ptsPerSeq:
            data = data[:self.ptsPerSeq,:]
        elif data.shape[0] < self.ptsPerSeq:
            padAmt = self.ptsPerSeq - data.shape[0]
            data = np.pad(data,((0,padAmt),(0,0)))
        return data