'''
Created on 2 Mar 2022
@author: filippomasi
'''

import numpy as np

class PreProcessing():
    
    def __init__(self,train_percentage):
    
        self.train_percentage=train_percentage
        
        return
        
    def split_dataset(self,x,percentage=False):
        n_samples = x.shape[0]
        train_p = int(n_samples * self.train_percentage)
        val_p = int(n_samples * 0.5 *(1.- self.train_percentage))
        test_p = val_p
        if percentage==True:
            return train_p,val_p,test_p
        else:
            return x[:(train_p + val_p)],x[(train_p + val_p):(train_p + val_p + test_p)]
            
    def GetParams(self,x,normalize=True,normalize_01=False):
        '''
        Find normalization parameters for data x
        '''
        if normalize == True:
            if normalize_01 == False:
                A = 0.5*(np.amax(x)-np.amin(x));
                B = 0.5*(np.amax(x)+np.amin(x))
            else: A = np.amax(x); B = 0.
            
        elif normalize == False:
            A = np.std(x,axis=0); B = np.mean(x,axis=0);
            if A.shape[0]>1: A[np.abs(A)<1.e-12]=1     
        return np.array([np.float32(A),
                         np.float32(B)])
                
    def Normalize(self,inputs,prm,train_val=False):
        '''
        Normalize features
        :inputs : data
        :prm : normalization parameters
        '''
        if train_val==True:
            inputs_tv,inputs_test=inputs
            return np.divide(np.add(inputs_tv, -prm[1]), prm[0]),np.divide(np.add(inputs_test, -prm[1]), prm[0])
        else:
            return np.divide(np.add(inputs, -prm[1]), prm[0])
    
    def DeNormalize(self,outputs,prm):
        '''
        Denormalize features
        :output : dimensionless data
        :prm : normalization parameters
        '''
        return np.add(np.multiply(outputs, prm[0]), prm[1])
    
    def computeEnergy_elasticity(self,strain,stress):
        energy = 0.5*(strain[0]*stress[0]+strain[1]*stress[1]+strain[5]*stress[5])
        return energy
            
            
        
        
