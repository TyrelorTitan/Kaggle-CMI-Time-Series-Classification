# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 16:35:55 2025

@author: agilj

"""

import numpy as np

import torch
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R

import colorednoise as cn

"""
Utility class to handle augmentations, data scaling, and feature extraction.
"""
class featureMaker():
    def __init__(self, seed=618567):
        self.rng = np.random.default_rng(seed)
        
    """
    Method to run the augmentation, scaling, and handedness adjustment, and 
    then to make derivative features from the resulting data.
    """
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
    
    """
    Method to add a pink noise to the data.
    """
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
    
    """
    Method to augment the data by making it appear as though the action was
    performed faster or slower.
    """
    def timestretch(self, x, last, thm=None, tof=None):
        rate = torch.Tensor(self.rng.uniform(0.9, 
                                             1.1, 
                                             (x.size()[0],))).to('cuda:0')

        L_new = (last / rate).to(int)
        L_new[L_new>=150] = 149
    
        xnew = [F.interpolate(x[i,:last[i],:].unsqueeze(0).permute(0,2,1),
                           L_new[i]).permute(0,2,1)[:,:150,:] \
             for i in range(x.size()[0])]
        x2 = torch.stack([F.pad(x,(0,0,0,150-x.size()[1],0,0)) for x in xnew])
        augmented = x2[:,0,:,:]

        if thm is not None:
            new = [F.interpolate(thm[i,:last[i],:].unsqueeze(0).permute(0,2,1), 
                                    L_new[i]).permute(0,2,1)[:,:150,:] \
                      for i in range(thm.size()[0])]
            thm2 = torch.stack([F.pad(thm,(0,0,0,150-thm.size()[1],0,0)) \
                                for thm in new])
            thm_aug = thm2[:,0,:,:]


        if tof is not None:
            new = [F.interpolate(tof[i,:last[i],:].unsqueeze(0).permute(0,2,1),
                               L_new[i]).permute(0,2,1)[:,:150,:] \
                 for i in range(tof.size()[0])]
            tof2 = torch.stack([F.pad(tof,(0,0,0,150-tof.size()[1],0,0)) \
                                for tof in new])
            tof_aug = tof2[:,0,:,:]
                
        if (tof is None) and (thm is None):
            return augmented, L_new
        elif (tof is None) and (thm is not None):
            return augmented, L_new, thm_aug
        elif (tof is not None) and (thm is None):
            return augmented, L_new, tof_aug
        elif (tof is not None) and (thm is not None):
            return augmented, L_new, thm_aug, tof_aug
        
    """
    Method to add a drift augmentation, modeled as a random walk, to the data.
    """
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
        
    """
    Method to construct all derivative features from the IMU and angular data.
    """
    def makeFeatures(self, x, last):
        x = self._getAngularVel(x, last)
        x = self._getAngularAcc(x, last)
        x = self._getAngularJerk(x, last)
        x = self._getSpatialJerk(x, last)
        x = self._getScalarFeatures(x, last)
        
        return x
    
    """
    Method to adjust the features to make left-handed data look similar to 
    right-handed data.
    """
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
    
    """
    Method to scale the data by the height of the individual.
    """
    def scale(self, x, demo):
        x[:,:,:3] = x[:,:,:3] / demo[:,4].unsqueeze(1).unsqueeze(1)
        
        return x
    
    """
    Method to perform data augmentation on the batch of data.
    """
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
    Compute change in angular position by using quaternion -> angular velocity
    formula.
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
    
    """
    Partial derivative in each angular direction.
    """ 
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
    
    """
    Partial derivative in each angular direction.
    """
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