import numpy as np
import pickle
import copy
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from cave_flowshop import calculateTotalTime
from cave_flowshop import VAE
from cave_flowshop import get_config
from cave_flowshop import decode

def MCMC_metropolis(N,T,instance,calculateTotalTime,VAEModel,config,decode):
    
    time_start=datetime.datetime.now()
    print("start:",time_start)
    
    #initial pop
    pop=np.random.uniform(-config.search_space_bound, config.search_space_bound, (1, config.search_space_size))
    VAEModel.reset_decoder(1, config)
    
    instance=instance.unsqueeze(0)
    args=(VAEModel, config, instance, calculateTotalTime)
    
    seq, score = decode(pop, *args)
    
    for i in range(0,N):  
        if i%100==0:
            print(i)
        pop_new = copy.copy(pop)
        
        #switch two random points
        a = random.randint(0, config.search_space_size-1)
        b = random.randint(0, config.search_space_size-1)
        pop_new[0][a], pop_new[0][b] = pop_new[0][b], pop_new[0][a]
        
        seq_new, score_new = decode(pop_new, *args)       
        
        # compare new and old sequence
        delta=score_new[0]-score[0]
        if delta<0:
            # accept new solution
            seq=seq_new 
            score=score_new
            pop=pop_new
        else:
            # the worse new seq could still be accepted 
            if np.random.uniform(0, 1) < np.exp(-delta/T):
                seq=seq_new 
                score=score_new
                pop=pop_new

    time_end=datetime.datetime.now()
    duration=time_end-time_start
    print("end:",time_end)
    print("duration:",duration)

    return seq,score,duration

if __name__ == "__main__":
    
    model_file = open(r"/home/nagashijun/cvae/runs/run_22.3.2022_60484/models/model_60484_15.pt",'rb')
    checkpoint = torch.load(model_file)
    config = get_config()
    config.search_space_bound=checkpoint['Z_bound']

    VAEModel =  VAE(config).to(config.device)
    VAEModel.load_state_dict(checkpoint['parameters']) 
    
    pickle_file = open(r"/home/nagashijun/cvae/flowdata/process_num=20x6(100000)/instances.pkl",'rb')
    instances=pickle.load(pickle_file)
    pickle_file.close()
    
    evaluate_data_instances=instances[-20:]
    evaluate_data_instances_tensor = torch.Tensor(evaluate_data_instances)
    evaluate_data_instances_tensor = evaluate_data_instances_tensor.to(config.device)
    
    N=200
    T=0.01
    
    for i in range(len(evaluate_data_instances_tensor)):
        print("evaluate_instance_NO.",i+1)
        seq,score,duration=MCMC_metropolis(N,T,evaluate_data_instances_tensor[i],calculateTotalTime,VAEModel,config,decode)
        print("score:",score)
        print("solution",seq)
