import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
import numpy as np
import threading
import time
#from gym_malware.envs.utils.MalConv import MalConv
from MalConv import MalConv

from ember import predict_sample
import warnings
import random
warnings.filterwarnings("error")


EMBER_MODEL_PATH = './gym_malware/envs/utils/detectors/ember_model.txt'
MALCONV_MODEL_PATH = './gym_malware/envs/utils/malconv.checkpoint'
NONNEG_MODEL_PATH = './gym_malware/envs/utils/nonneg_2nd.checkpoint'

outputs = None

class MalConvModel(object):
    def __init__(self, model_path, thresh=0.5, name='malconv'): 
        self.model = MalConv(channels=256, window_size=512, embd_size=8).train()
        weights = torch.load(model_path,map_location='cpu')
        self.model.load_state_dict(weights['model_state_dict'])
        self.thresh = thresh
        self.__name__ = name

    def predict(self, bytez):
        _inp = torch.from_numpy( np.frombuffer(bytez,dtype=np.uint8)[np.newaxis,:] )
        with torch.no_grad():
            outputs = F.softmax( self.model(_inp), dim=-1)

        return outputs.detach().numpy()[0,1] > self.thresh

    def trainit(self, bytez):
        _inp = torch.from_numpy( np.frombuffer(bytez,dtype=np.uint8)[np.newaxis,:] )
        with torch.no_grad():
            outputs = F.softmax( self.model(_inp), dim=-1)

        return outputs

class EmberModel(object):
    # ember_threshold = 0.8336 # resulting in 1% FPR
    def __init__(self, model_path=EMBER_MODEL_PATH, thresh=0.8336, name='ember'):
        # load lightgbm model
        self.model = lgb.Booster(model_file=model_path)
        self.thresh = thresh
        self.__name__ = 'ember'

    def predict(self,bytez):
        return predict_sample(self.model, bytez) > self.thresh



def check(model):
    with torch.no_grad():
        global outputs
        outputs = F.softmax(model(_inp), dim=-1)
        outputs.detach().numpy()[0,1] > 0.5 ## threshold for malconv decision = 0.5, for NonNeg change threshold to 0.35, For LightGBM the threshold should be 0.9
    #time.sleep(1)
    

def test_Model(models,exe_path):
    sum_positive=0
    sum_negative=0
    threshold=0.5
    exe_data = os.listdir(exe_path)[:100]

    for i in exe_data:
        #print(i)
        with open (exe_path+i,"rb") as f:
            bytez = f.read()
            #outputs=model.predict(bytez)
            _inp = torch.from_numpy( np.frombuffer(bytez,dtype=np.uint8)[np.newaxis,:] )
            #outputs = F.softmax(model_mal(_inp), dim=-1)
            
            try:
                outputs = F.softmax(models(_inp), dim=-1).detach().numpy()[0]
                #print(outputs)
                if outputs[0]>threshold:
                    sum_positive+=1
                else:
                    sum_negative+=1
            except:
                #print(i)
                continue
    return sum_positive,sum_negative

if __name__ == '__main__':
  
    import sys
    import os
    exe_path = '/Data/chaiyidong/pythonCode/Mo85-radar_new/evaded/blackbox_2nd/'
    #exe_path = '/home/chaiyd/pythonCode/adar/defense/VT_original_Functional/Virus/'
    #exe_path = '/home/chaiyd/pythonCode/adar/defense/VT_original_Functional/Dropper/'
    exe_path_neg_train='/Data/chaiyidong/pythonCode/adar/Dataset/Benign_test/'    
    
    model_mal = MalConv(channels=256, window_size=512, embd_size=8).train()
    weights = torch.load(MALCONV_MODEL_PATH,map_location='cpu')
    #model_mal.load_state_dict( weights['model_state_dict'])
    model_mal.load_state_dict(weights)

    
    opt = torch.optim.SGD(model_mal.parameters(), lr=1e-4)
    exe_data = os.listdir(exe_path)
    exe_data_neg = os.listdir(exe_path_neg_train)

    train=True
    if train==True:
        for epoch_i in range(2):
            sampled_list = random.sample(exe_data, 1)
            try:
                for i in sampled_list:
                    with open (exe_path+i,"rb") as f:
                        bytez = f.read()
                        #outputs=model.predict(bytez)
                        _inp = torch.from_numpy( np.frombuffer(bytez,dtype=np.uint8)[np.newaxis,:] )
                        output = F.softmax(model_mal(_inp), dim=-1)
                        y_true=torch.tensor([1])
                        loss = nn.CrossEntropyLoss()(output,y_true)

                sampled_neg_list = random.sample(exe_data_neg, 3)
                for i in sampled_neg_list:
                    print(i)
                    with open (exe_path_neg_train+i,"rb") as f:
                        bytez = f.read()
                        #outputs=model.predict(bytez)
                        _inp = torch.from_numpy( np.frombuffer(bytez,dtype=np.uint8)[np.newaxis,:] )
                        output = F.softmax(model_mal(_inp), dim=-1)

                        y_true=torch.tensor([0])

                        loss = loss+nn.CrossEntropyLoss()(output,y_true)
                        if opt:
                            opt.zero_grad()
                            loss.backward()
                            opt.step()
            except:
                continue
        torch.save(model_mal.state_dict(), '/Data/chaiyidong/pythonCode/Mo85-radar_new/gym_malware/envs/utils/nonneg_3nd.checkpoint')
        torch.save(model_mal, '/Data/chaiyidong/pythonCode/Mo85-radar_new/gym_malware/envs/utils/test_model')
        model_mal = torch.load('/Data/chaiyidong/pythonCode/Mo85-radar_new/gym_malware/envs/utils/test_model')
    else:
        model_mal = torch.load('/Data/chaiyidong/pythonCode/Mo85-radar_new/gym_malware/envs/utils/test_model')
    
    
    exe_path_test ='/Data/chaiyidong/pythonCode/Mo85-radar_new/evaded/blackbox_2nd/'
    sum_positive,sum_negative=test_Model(model_mal,exe_path_test)
    print(sum_positive,sum_negative)
    print('Total: %d, correct: %d, sensitivity: %.4f'%(sum_positive+sum_negative,sum_positive,1.0*sum_positive/(sum_positive+sum_negative)))
    
    exe_path_neg_test='/Data/chaiyidong/pythonCode/adar/Dataset/Benign_test/'
    sum_positive,sum_negative=test_Model(model_mal,exe_path_neg_test)
    print(sum_positive,sum_negative)
    print('Total: %d, correct: %d, specificity: %.4f'%(sum_positive+sum_negative,sum_negative,1.0*sum_negative/(sum_positive+sum_negative)))

    '''
    for i in exe_data:
        print(i)
        with open (exe_path+i,"rb") as f:
            bytez = f.read()
            for m in models:
                #outputs=model.predict(bytez)
                print(f'{m.__name__}: {m.predict(bytez)}')
    '''