# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:01:33 2025

@author: agilj

The idea here is to use an ensemble of neural networks trained with different
subsets of the training data. We then train an SVM to classify based on the
outputs of the different neural networks. This is something like a majority
poll, but using an SVM.
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

from CMI_DatasetClass import CMI_Dataset
from CMI_FeaturemakerClass import featureMaker
from CMI_NetworkClass import CMI_Network_IMU
    
#%%
if __name__ == '__main__':
    #%%
    rng = np.random.default_rng(0)

    # Specify data path.
    dataframe_path = './train.csv'
    demo_path = './train_demographics.csv'
    device = 'cuda:0'
    numFolds=5
    
    # Get sequences
    numSubj = 81
    numSeqs = 8151
    ptsPerSeq = 50

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
    torch.backends.cudnn.deterministic = True
    for k in range(numFolds):
        # Randomly pick 10 subjects to use as validation
        picks = rng.permutation(numSubj)[:10]
        subjToUse = list(subjNames.iloc[picks])
        subjToRemove = ['REMOVE'] + subjToUse + list(svmNames)
        print('Assembling training dataset.')
        trainData = CMI_Dataset(dataframe_path, demo_path,
                                subjToUse=subjToRemove)
        numClasses = len(trainData.classes)
        
        # Make validation dataset.
        print('Assembling validation dataset.')
        valData = CMI_Dataset(dataframe_path, demo_path,
                              subjToUse=subjToUse,
                              mean=trainData.dfMean, std=trainData.dfStd)
        # Make dataloaders.
        batchSize = 64
        trainDataloader = DataLoader(trainData, batch_size=batchSize, 
                                     shuffle=True, num_workers=0, drop_last=False)
        valDataloader = DataLoader(valData, batch_size=batchSize, 
                                   shuffle=False, num_workers=0, drop_last=False)
        
        # Instantiate network, specify optimizer, specify loss.
        print('Building classification network.')
        net = CMI_Network_IMU(numClasses=numClasses).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim,
            T_0=100,
            T_mult=2)
        warmup_scheduler = warmup.UntunedLinearWarmup(optim)
        loss_fun = nn.CrossEntropyLoss()
        net.train()
            
        # Training loop per network.
        numEpochs = 100
        numPts = len(trainDataloader.dataset)
            # Create log file.
        logfile_path = './outputs/GClass/'+str(k)+'log.txt'
        mode = 'w'
        with open(logfile_path, mode) as f:
            f.write('Validation Loss\n')
            # Set validation loss requirement for saving.
        lastValF1 = -torch.inf
        print('Beginning Training!')
        fm = featureMaker()
        for ep in range(numEpochs):
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
                                   str(ep)+".pth")
            net.train()
                
            # Print progress
            t2 = time.time()
            totalTime = t2-t1
            lossAvg = lossAvg / trainData.__len__()
            accCurr = acc / trainData.__len__()
            print(f"(Epoch {ep:d}) Loss: {lossAvg:f} // {lossVal:f} " + \
                  f"-- Acc: {accCurr:>3f} // {accVal:>3f} " + \
                  f"-- F1: {f1_total:>3f} " + \
                  f"-- Time: {totalTime:2f}s.")

    #%% Build SVM Infrastructure
    # The goal here is that each of the cross-validation networks are likely
    # good at different things (since they have different scores), so we train
    # an SVM on *a different dataset* to use the networks' output probabilities
    # to predict the true class.
    
    from sklearn import svm

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
    
    #%% Build SVM Training Dataset
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
                predClasses.append(torch.argmax(pred,dim=1).cpu())
                trueClasses.append(i*torch.ones_like(predClasses[-1]))
                
                # IMU Only
                data[:,:,19:] = 0
                pred = net(data)
                tmp_imu.append(pred)
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

    #%% Train an SVM
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

    #%% Build SVM Validation Dataset
    svmValData = []
    predClassesVal = []
    trueClassesVal = []
    
    svmValData_imu = []
    predClassesVal_imu = []
    trueClassesVal_imu = []
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
        
        # Make features
        Xdata = fm.makeFeatures(Xdata, last)
        
        # Concatenate data from all sources.
        data = torch.cat((Xdata, Tdata, Fdata), dim=2)
        tmp = []
        tmp_imu = []
        for j, net in enumerate(nets):
            with torch.no_grad():
                # With all data.
                predVal = net(data)
                tmp.append(predVal)
                predClassesVal.append(torch.argmax(predVal,dim=1).cpu())
                trueClassesVal.append(i*torch.ones_like(predClassesVal[-1]))

                # No extra data (IMU only).
                data[:,:,19:] = 0
                predVal = net(data)
                tmp_imu.append(predVal)
                predClassesVal_imu.append(torch.argmax(predVal,dim=1).cpu())
                trueClassesVal_imu.append(i*torch.ones_like(predClassesVal_imu[-1]))
        
        # Store results.
        tmp = np.array(torch.cat(tmp,dim=1).cpu())
        tmp_imu = np.array(torch.cat(tmp_imu,dim=1).cpu())
        svmValData.append(tmp)
        svmValData_imu.append(tmp_imu)

    # Build SVM-compatible dataset
    data_forSVM, labels_forSVM = format_SVM_data(svmValData)
    predClassesVal = np.array(torch.cat(predClassesVal,dim=0),dtype=int)
    predClassesVal = np.reshape(predClassesVal, (svmValDataset.__len__(), 
                                                 numFolds))
    predClassesVal = mode(predClassesVal, axis=1)[0]    
    trueClassesVal = np.array(torch.cat(trueClassesVal,dim=0),dtype=int)
    trueClassesVal = np.reshape(trueClassesVal, (svmValDataset.__len__(), 
                                                 numFolds))
    trueClassesVal = mode(trueClassesVal, axis=1)[0]
    
    data_forSVM_imu, labels_forSVM_imu = format_SVM_data(svmValData_imu)
    predClassesVal_imu = np.array(torch.cat(predClassesVal_imu,dim=0),dtype=int)
    predClassesVal_imu = np.reshape(predClassesVal_imu, 
                                    (svmValDataset.__len__(),numFolds))
    predClassesVal_imu = mode(predClassesVal_imu, axis=1)[0]    
    trueClassesVal_imu = np.array(torch.cat(trueClassesVal_imu,dim=0),\
                                  dtype=int)
    trueClassesVal_imu = np.reshape(trueClassesVal_imu,
                                    (svmValDataset.__len__(),numFolds))
    trueClassesVal_imu = mode(trueClassesVal_imu, axis=1)[0]
    
    #%% Validate the SVM
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