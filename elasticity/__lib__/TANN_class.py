'''
Created on 2 Mar 2022
@author: filippomasi
'''

import numpy as np
import tensorflow as tf
import pickle
import os

class TANN(tf.keras.Model):
    """ Thermodynamics-based Artificial Neural Networks """

    def __init__(self,
                 umat_dim,
                 material,
                 hidden_NN_Energy,
                 activation_NN_Energy,
                 hidden_NN_Evolution=0,
                 activation_NN_Evolution='',
                 name="Thermodynamics-based_Artificial_Neural_Networks", **kwargs):
        
        super(TANN, self).__init__(name=name, **kwargs)
        
        ''' elastic or inelastic '''
        self.material = material
        
        ''' Size and dimensions '''
        self.dim = umat_dim
        self.hidden_NN_Energy = hidden_NN_Energy
        self.activation_NN_Energy = activation_NN_Energy
        self.hidden_NN_Evolution = hidden_NN_Evolution
        self.activation_NN_Evolution = activation_NN_Evolution
        
        ''' Neural Networks (NN) initialization '''
        self.NN_Energy = self.NN_Energy_Init(int(4*self.dim),
                                             hidden_NN_Energy,
                                             activation_NN_Energy)
        
        if self.material == 'inelastic':
            self.NN_Evolution = self.NN_Evolution_Init(int(24*self.dim),
                                                       hidden_NN_Evolution,
                                                       activation_NN_Evolution)
                                                   
        self.prm_E = np.array([1,0]); self.prm_EDot = np.array([1,0]);
        self.prm_Z = np.array([1,0]); self.prm_ZDot = np.array([1,0])
        self.prm_S = np.array([1,0]); self.prm_F = np.array([1,0]); self.prm_D = np.array([1,0])
        
    
    def SetParams(self,params):
        '''
        Make normalization parameters attributes of TANN class
        :params : normalization parameters
        '''
        if self.material=='elastic':
            self.prm_E,self.prm_S,self.prm_F = params   
            
        else:
            self.prm_E,self.prm_S,self.prm_Z,self.prm_F,self.prm_D,self.prm_EDot,self.prm_ZDot = params   
        
        return
    
    def NN_Evolution_Init(self, nodes_Z, number_hidden, activation_NN, train=True):
        '''
        Initializator of the evolution law NN
        '''
        if activation_NN == 'fast_gelu':
            activation=self.fast_gelu
        else:
            activation=activation_NN
        
        NNZ = tf.keras.Sequential(name='NN_Evolution')
        for i in range(number_hidden):
            NNZ.add(tf.keras.layers.Dense(nodes_Z,activation=activation,trainable=train))
        
        NNZ.add(tf.keras.layers.Dense(self.dim,trainable=train))
        return NNZ
    
    def NN_Energy_Init(self, nodes_F, number_hidden, activation_NN, train=True):
        '''
        Initializator of the energy NN
        '''
        if activation_NN == 'fast_gelu':
            activation=self.fast_gelu
        else:
            activation=activation_NN
        
        NNF = tf.keras.Sequential(name='NN_Energy')
        for i in range(number_hidden):
            NNF.add(tf.keras.layers.Dense(nodes_F,activation=activation,trainable=train))
        
        NNF.add(tf.keras.layers.Dense(1,trainable=train,name="output2"))
        return NNF

    def Normalize(self,inputs,prm):
        '''
        Normalize features
        :inputs : data
        :prm : normalization parameters
        '''
        return tf.divide(tf.add(inputs, -prm[1]), prm[0])
    
    def DeNormalize(self,outputs,prm):
        '''
        Denormalize features
        :output : dimensionless data
        :prm : normalization parameters
        '''
        return tf.add(tf.multiply(outputs, prm[0]), prm[1])
    
        
    def computeStress(self,Energy,Strain,norm=True):
        '''
        Compute stress from gradients
        :Energy : energy from NN_Energy
        :Strain : strain
        '''
        Stress = tf.gradients(Energy,Strain)[0]
        nStress = self.Normalize(Stress,self.prm_S)
        if norm==True: return nStress
        else: return Stress
        
    def computeDissipation(self,Energy,SVars,SVarsDot,
                    norm=True):
        '''
        Compute dissipation rate from gradients
        :Energy : energy from NN_Energy
        :SVars : internal svars
        :SVarsDot : internal svars rate
        '''
        ThForce = -tf.gradients(Energy,SVars)[0]
        Dissipation = tf.linalg.einsum('ij,ij->i',ThForce,SVarsDot)
        nDissipation = self.Normalize(Dissipation,self.prm_D)
        if norm==True: return nDissipation
        else: return Dissipation
        
    def L1L2_loss(self,y_true,y_pred):
        '''
        L1L2 Loss function
        '''
        delta = 1.e-3; x = y_pred-y_true
        loss = tf.where(tf.abs(x)>delta,tf.abs(x),tf.pow(x,2)/(2*delta)+delta/2)-0.5*delta
        return tf.reduce_mean(loss,axis=-1)  
    
    def fast_gelu(self,x): return x*tf.nn.sigmoid(1.702*x)
    
    def call(self, inputs):
        if self.material == 'elastic':
            nStrain=inputs
            uStrain = self.DeNormalize(nStrain,self.prm_E)
            NStrain = self.Normalize(uStrain,self.prm_E)
            
            state = tf.concat([NStrain],1)
            nEnergy = self.NN_Energy(state)
            uEnergy = self.DeNormalize(nEnergy,self.prm_F)
            
            nStress = self.computeStress(uEnergy,uStrain)
            
            return [nEnergy,nStress]
            
        else:    
            nSVars,nStrain,nStrainDot = inputs
            uStrain = self.DeNormalize(nStrain,self.prm_E)
            NStrain = self.Normalize(uStrain,self.prm_E)
            
            uSVars = self.DeNormalize(nSVars,self.prm_Z)
            NSVars = self.Normalize(uSVars,self.prm_Z)
            
            state = tf.concat([NStrain,NSVars],1)
            nEnergy = self.NN_Energy(state)
            uEnergy = self.DeNormalize(nEnergy,self.prm_F)
            
            nStress = self.computeStress(uEnergy,uStrain) 
            var = tf.concat([nStrainDot,nSVars,nStrain],1)
            nSVarsDot = self.NN_Evolution(var)
            uSVarsDot = self.DeNormalize(nSVarsDot,self.prm_ZDot)
            
            nDissipation = self.computeDissipation(uEnergy,uSVars,uSVarsDot)
    
            return [nSVarsDot,
                    nEnergy,
                    nStress,
                    nDissipation]
        
    def compileTANN(self):
        
        rate = 1e-3
        optimizer = tf.keras.optimizers.Nadam(learning_rate=rate,beta_1=0.9999,beta_2=0.9999,epsilon=1e-10)
        
        self.compile(optimizer = optimizer,
                     loss = [self.L1L2_loss,self.L1L2_loss,self.L1L2_loss,self.L1L2_loss,self.L1L2_loss],
                     loss_weights=[1.,1.,1.,1.,1.])
        
    def save_export(self,filename,saveDataPath=''):
        # define the name of the directory to be created
        path = saveDataPath+filename

        try:
            os.mkdir(path)
        except OSError:
            print ("Creation of the directory '%s' failed." % path)
        else:
            print ("Successfully created the directory '%s' " % path)
        
        params = [self.prm_E,self.prm_S,self.prm_Z,self.prm_F,self.prm_D,self.prm_EDot,self.prm_ZDot,
                  self.dim,self.material,
                  self.hidden_NN_Energy,self.activation_NN_Energy,
                  self.hidden_NN_Evolution,self.activation_NN_Evolution]
        file_params = path + '/_params'
 
        with open(file_params, 'wb') as f_obj:
            pickle.dump(params, f_obj)
               
        file_weights = path + '/_weights'
        self.save_weights(file_weights, save_format='tf') 
        print("Successfully exported model.")
        return
    
    def import_compile(self,file,input_shape):
        
        filename = file+'/_params'
        with open(filename, 'rb') as f_obj:
            params = pickle.load(f_obj)
        
        self.prm_E,self.prm_S,self.prm_Z,self.prm_F,self.prm_D,self.prm_EDot,self.prm_ZDot,self.dim,self.material,self.hidden_NN_Energy,self.activation_NN_Energy,self.hidden_NN_Evolution,self.activation_NN_Evolution = params
                  
#         self.prm_E,self.prm_S,self.prm_Z,self.prm_F,self.prm_D,self.prm_EDot,self.prm_ZDot,self.dim,self.material,self.hidden_NN_Energy,self.activation_NN_Energy,self.hidden_NN_Evolution,self.activation_NN_Evolution=params
        
        print('File',file,'refers to the following model:'
              '\numat_dim:',self.dim,
              '\nmaterial:',self.material,
              '\nhidden_NN_Energy:',self.hidden_NN_Energy,
              '\nactivation_NN_Energy:',self.activation_NN_Energy,
              '\nhidden_NN_Evolution:',self.hidden_NN_Evolution,
              '\nactivation_NN_Evolution:',self.activation_NN_Evolution)
        
        self.build(input_shape)

        
        filename = file+'/_weights'
        self.load_weights(filename)
        
        return 
