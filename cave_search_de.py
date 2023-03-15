import pickle
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from cave_flowshop import VAE
from cave_flowshop import get_config
from cave_flowshop import minimize
from cave_flowshop import solve_instance_de
from cave_flowshop import calculateTotalTime

def evaluate(model, config, instances):
    model.eval()
    cost_fn=calculateTotalTime

    cost_values = []
    runtime_values = []
    solutions=[]
    for i, instance in enumerate(instances):
        print("evaluate NO.",i+1)
        start_time = time.time()        
        objective_value,solution = solve_instance_de(model, instance, config, cost_fn)      
        cost_values.append(objective_value)
        print("Cost " + str(objective_value))   
        runtime = time.time() - start_time
        runtime_values.append(runtime)  
        solutions.append(solution)
        print("solution",solution)
        
    return runtime_values, cost_values

# Main
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
    
    evaluate(VAEModel, config, evaluate_data_instances)
