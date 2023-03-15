import pandas as pd
import pickle
import zipfile
import numpy as np
import time
import datetime
import logging
import os
import sys
import argparse
import random
import copy
from random import sample
from pandas import DataFrame

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_config(args=None):
    parser = argparse.ArgumentParser(
        description="CVAE-Opt Training")

    parser.add_argument('--problem', type=str, default='Flowshop 20*6')
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--process_num', default=6, type=int)
    parser.add_argument("--problem_size", type=int, default=20)
    parser.add_argument('--batch_size', default=128, type=int)
    #parser.add_argument('--epoch_size', type=int, default=9344, help='Number of instances used for training')
    parser.add_argument('--epoch_size', type=int, default=28032, help='Number of instances used for training')
    parser.add_argument('--nb_epochs', default=500, type=int)
    parser.add_argument('--search_validation_size', default=100, type=int)
    parser.add_argument('--network_validation_size', default=640, type=int)
    parser.add_argument('--KLD_weight', default=0.01, type=float)  # Beta in the paper
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--q_percentile', default=99, type=float)


    # Differential Evolution
    parser.add_argument('--search_space_size', default=20, type=int)    
    parser.add_argument('--de_mutate', default=0.3, type=float)
    parser.add_argument('--de_recombine', default=0.95, type=float)
    parser.add_argument('--de_popsize', default=128, type=float)
    parser.add_argument('--search_timelimit', default=30, type=int)
    parser.add_argument('--search_iterations', default=3, type=int)    
    
    config = parser.parse_args(args=[])

    config.device = torch.device(config.device)

    return config
    

class FlowshopDataset(Dataset):
    def __init__(self, size, problem_size, config, data):
        self.size = size
        self.problem_size = problem_size
        self.instances = data[0]
        self.solutions = data[1]
        self.config = config

        assert len(self.instances) == len(self.solutions)
        assert len(self.instances) >= size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        instances = self.instances[idx]
        solutions = np.array(self.solutions[idx])

        instances = torch.from_numpy(instances).to(self.config.device).float()
        solutions = torch.from_numpy(solutions).long().to(self.config.device)
      
        return instances, solutions, solutions


# MODEL
class Embedding(nn.Module):
    """Encodes the coordinate states using 1D Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Embedding, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)

    def forward(self, input_data):
        output_data = self.embed(input_data)
        return output_data
        
        
##ENCODER        
class Encoder(nn.Module):
    def __init__(self, instance_embedding, reference_embedding, encoder_attn, rnn, search_space_size,
                 hidden_size):
        super(Encoder, self).__init__()
        self.instance_embedding = instance_embedding
        self.reference_embedding = reference_embedding
        self.encoder_attn = encoder_attn
        self.gru_decoder = nn.GRU(hidden_size * 2, hidden_size, 1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, search_space_size)
        self.fc2 = nn.Linear(hidden_size, search_space_size)
        self.rnn = rnn #rnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, dropout=0)
    
    def forward(self, instance, solution, instance_hidden, config):
        batch_size, sequence_size, input_size, = instance.size() 
        reference_input = instance[torch.arange(batch_size), solution[:, 0], :].unsqueeze(1).detach() 
    
        last_hh = None  # hidden state(num_layers * num_directions, batch, hidden_size)
        last_hh_2 = None # hidden state(num_layers * num_directions, batch, hidden_size)
        
        reference_hidden = self.reference_embedding(reference_input) 
        
        for j in range(1, solution.shape[1]):
            
            rnn_out, last_hh = self.rnn(reference_hidden, last_hh)
            
            # Given a summary of the output, find an input context
            enc_attn = self.encoder_attn(instance_hidden, rnn_out)
            context = enc_attn.permute(0, 2, 1).bmm(instance_hidden) 
            
            ptr = solution.t()[j].long() 
            
            reference_input = torch.gather(instance, 1, ptr.view(-1, 1, 1).expand(-1, 1, input_size))
            reference_hidden = self.reference_embedding(reference_input)

            rnn_input = torch.cat((reference_hidden, context), dim=2)
            rnn_out_2, last_hh_2 = self.gru_decoder(rnn_input, last_hh_2)            
        
        mu = self.fc1(last_hh_2.squeeze(0))
        log_var = self.fc2(last_hh_2.squeeze(0))
        return self.reparameterise(mu, log_var), mu, log_var       

    @staticmethod
    def reparameterise(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, hidden_size, 1), requires_grad=True)) 
        self.W = nn.Parameter(torch.zeros((1, 2 * hidden_size, 1 * hidden_size), requires_grad=True))
    def forward(self, instance_hidden, rnn_out):
        batch_size, _, hidden_size = instance_hidden.size() 

        hidden = rnn_out.expand_as(instance_hidden)  
        hidden = torch.cat((instance_hidden, hidden), 2) 

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, -1, -1) 
        W = self.W.expand(batch_size, -1, -1) 

        ret = torch.bmm(hidden, W)  
        attns = torch.bmm(torch.relu(ret), v) 
        attns = F.softmax(attns, dim=1)  
        return attns

    
## DECODER
def update_mask(mask, dynamic, chosen_idx):
    """Marks the Processed process, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask

### version 2 Remove the restriction that it can only start from the first starting point,which is different from orginal paper
class Decoder(nn.Module):
    def __init__(self, instance_embedding, reference_embedding, encoder_attn, rnn, hidden_size,
                 search_space_size, mask_fn):
        super(Decoder, self).__init__()

        # Define the encoder & decoder models
        self.pointer = Pointer(encoder_attn, rnn, hidden_size, search_space_size)
        self.instance_embedding = instance_embedding
        self.reference_embedding = reference_embedding
        self.encoder_attn = encoder_attn
        self.mask_fn = mask_fn

        self.rnn = rnn
        
    def forward(self, instance, solution, Z, instance_hidden, config, teacher_forcing, last_hh_new=None):
        batch_size, sequence_size, input_size, = instance.size() 
        
        max_steps = sequence_size if self.mask_fn is None else 10000  #10000
        reference_input = instance[torch.arange(batch_size), solution[:, 0], :].unsqueeze(1).detach()

        tour_logp =[]
        tour_prob= []
        
        mask = torch.ones(batch_size, sequence_size, device=config.device) 
        
        if teacher_forcing:    #(True)
            tour_idx=[solution[:, [0]]]
            mask[torch.arange(batch_size), solution[:, 0]] = 0
            for j in range(1, max_steps):
                if not mask.byte().any():
                    break   # mask all -->0 then break
        
                reference_hidden = self.reference_embedding(reference_input)
                probs, last_hh_new = self.pointer(instance_hidden, reference_hidden, Z, last_hh_new)
                probs = F.softmax(probs + mask.log(), dim=1) 
                
                # Select the actions based on the training solutions (during training)
                ptr = solution.t()[j].long() 
                t = mask[torch.arange(len(mask)), ptr] 
                assert t.eq(1).all()
                logp = torch.log(probs[torch.arange(batch_size), ptr])
                # probs torch.Size([128, 20])
                _, predicted_ptr = torch.max(probs, 1) # – the result: tuple of two output tensors (max, max_indices)

                tour_idx.append(predicted_ptr.data.unsqueeze(1)) 
                if self.mask_fn is not None:
                    mask = self.mask_fn(mask, instance[:, :, 2:], ptr).detach() #visited position`s index-->
                
                reference_input = torch.gather(instance, 1, ptr.view(-1, 1, 1).expand(-1, 1, input_size)) 
                tour_prob.append(probs)
                tour_logp.append(logp.unsqueeze(1))
                
        else:
            tour_idx=[]
            for j in range(0, max_steps):
                if not mask.byte().any():
                    break   
                
                reference_hidden = self.reference_embedding(reference_input)
                probs, last_hh_new = self.pointer(instance_hidden, reference_hidden, Z, last_hh_new)
                probs = F.softmax(probs + mask.log(), dim=1)               
                
                # Select actions greedily (during the search)
                prob, ptr = torch.max(probs, 1)
                logp = prob.log()
                tour_idx.append(ptr.data.unsqueeze(1))
                
                if self.mask_fn is not None:
                    mask = self.mask_fn(mask, instance[:, :, 2:], ptr).detach() 
                
                reference_input = torch.gather(instance, 1, ptr.view(-1, 1, 1).expand(-1, 1, input_size))
            
                tour_prob.append(probs)
                tour_logp.append(logp.unsqueeze(1))
            
        tour_idx = torch.cat(tour_idx, dim=1)
        tour_logp = torch.cat(tour_logp, dim=1)

        return None, tour_idx, tour_logp

class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""
    def __init__(self, encoder_attn, rnn, hidden_size, search_space_size):
        super(Pointer, self).__init__()
        
        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, hidden_size, 1), requires_grad=True))
        self.W = nn.Parameter(torch.zeros((1, 2 * hidden_size, hidden_size) , requires_grad=True))
        
        # Used to compute a representation of the current decoder output
        self.fc1 = nn.Linear(2 * hidden_size + search_space_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, hidden_size)
        self.encoder_attn = encoder_attn
        self.rnn = rnn #rnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, dropout=0)
  
    def forward(self, instance_hidden, reference_hidden, Z, last_hh):
        rnn_out, last_hh = self.rnn(reference_hidden, last_hh)
        rnn_out = rnn_out

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(instance_hidden, rnn_out)
        context = enc_attn.permute(0, 2, 1).bmm(instance_hidden)  # (B, 1, num_feats)

        fc_input = torch.cat((context.squeeze(1), Z, reference_hidden.squeeze(1)), dim=1)  # (B, num_feats, seq_len)
        fc_output = self.fc1(fc_input)
        fc_output = self.fc2(fc_output).unsqueeze(1)
        fc_output = fc_output.expand(-1, instance_hidden.size(1), -1)
        fc_output = torch.cat((instance_hidden, fc_output), dim=2)

        v = self.v.expand(instance_hidden.size(0), -1, -1)
        W = self.W.expand(instance_hidden.size(0), -1, -1)
        probs = torch.bmm(torch.tanh(torch.bmm(fc_output, W)), v).squeeze(2)
        return probs, last_hh   

class VAE(nn.Module): 
    def __init__(self, config):
        super(VAE, self).__init__()

        input_size = config.process_num
        mask_fn = update_mask           
        hidden_size = 128    
    
        self.instance_embedding = Embedding(input_size, hidden_size)
        reference_embedding = Embedding(input_size, hidden_size)
        encoder_attn = Attention(hidden_size)
        rnn = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, dropout=0)    
        
        self.encoder = Encoder(self.instance_embedding, reference_embedding, encoder_attn, rnn,
                               config.search_space_size, hidden_size)
        self.decoder = Decoder(self.instance_embedding, reference_embedding, encoder_attn, rnn, hidden_size,
                               config.search_space_size, mask_fn)    
    
        self.instance_hidden = None
        self.dummy_solution = None    
        
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, instance, solution_1, solution_2, config):
        instance_hidden = self.instance_embedding(instance)
        output_e = self.encoder(instance, solution_1, instance_hidden, config)

        Z, mu, log_var = output_e

        output_prob, tour_idx, tour_logp = self.decoder(instance, solution_2, Z, instance_hidden, config,True)
        return  output_prob, mu, log_var, Z, tour_idx, tour_logp
        
    def decode(self, instance, Z, config):
        if self.instance_hidden is None:
            self.instance_hidden = self.instance_embedding(instance)
        output_prob, tour_idx, tour_logp = self.decoder(instance, self.dummy_solution, Z, self.instance_hidden, config,
                                                        False)
        return output_prob, tour_idx, tour_logp    
     
    def reset_decoder(self, batch_size, config):
        self.instance_hidden = None
        self.dummy_solution = torch.zeros(batch_size, 1).long().to(config.device)   #torch.Size([600, 1])  


# LOSS FUNCTION
def calculate_RC_loss(tour_logp):
    RC = - tour_logp.sum()
    return RC
def calculate_KLD_loss(mean, log_var):
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return KLD
    
    
# TRAIN
def train_epoch(model, config, epoch_idx, training_dataloader, optimizer):
    
    model.train() #set model to train mode

    start_time = time.time()
    
    loss_RC_values = []
    loss_KLD_values = []
    
    for batch_id, batch in enumerate(training_dataloader):    
        print("Batch {}/{}".format(batch_id, int(config.epoch_size / config.batch_size)))#Batch 729/730    
    
        # Get an instance and two symmetric solutions (see the paragraph symmetry breaking in the paper)
        instances, solutions_1, solutions_2 = batch    
        
        # Forward pass
        output, mean, log_var, Z, tour_idx, tour_logp = VAEModel(instances, solutions_1,solutions_2, config)
        
        # Calculate weighted loss
        loss_RC = calculate_RC_loss(tour_logp)
        loss_KLD = calculate_KLD_loss(mean, log_var)
        loss = loss_RC + loss_KLD * config.KLD_weight    
        
        # Update network weights 
        optimizer.zero_grad()    # Sets the gradients of all optimized torch.Tensors to zero
        assert not torch.isnan(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()      # Performs a single optimization step Update network weights 
        
        loss_RC_values.append(loss_RC.item())
        loss_KLD_values.append(loss_KLD.item())
        
    print(f' Loss_RC: {loss_RC.item()} Loss_KLD: {loss_KLD.item()}')
        
    epoch_duration = time.time() - start_time
    
    logging.info(
        "Finished epoch {}, took {} s".format(epoch_idx, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))
    logging.info("Epoch Loss_RC {}, Epoch Loss_KLD {}".format(np.mean(loss_RC_values), np.mean(loss_KLD_values)))
    #return loss_RC.item(), loss_KLD.item()
    loss_RC_mean=np.mean(loss_RC_values)
    loss_KLD_mean=np.mean(loss_KLD_values)
    return loss_RC_mean, loss_KLD_mean

def evaluate_network(config, model, validation_dataloader, epoch_idx):
    
    model.eval() 
    loss_RC_values = []
    loss_KLD_values = []
    abs_Z_values = []
    
    for batch_id, batch in enumerate(validation_dataloader):  
        instances, solutions_1, solutions_2 = batch
    
        with torch.no_grad():
            output, mean, log_var, Z, tour_idx, tour_logp = VAEModel(instances, solutions_1, solutions_2, config)
        
        loss_RC = calculate_RC_loss(tour_logp)
        loss_KLD = calculate_KLD_loss(mean, log_var)
        
        loss_RC_values.append(loss_RC.item())
        loss_KLD_values.append(loss_KLD.item())        
        
        abs_Z = torch.abs(Z)  # Absolute coordinates of points in latent space (Z)
        abs_Z_values.append(abs_Z.cpu().numpy())    # from tensor to array form, delete device: cuda:0

    abs_Z_values = np.array(abs_Z_values).flatten() #(128,20)

    # The bounds of the search space are defined as a percentile of the absolute latent variable coordinates
    new_bound = np.percentile(abs_Z_values, config.q_percentile).item()   
        
    logging.info(f'[Network Validation] Loss_RC: {np.mean(loss_RC_values)} Loss_KLD: {np.mean(loss_KLD_values)}')
    
    model.train()

    return new_bound # sampling boundary
    
def train(model, training_data, validation_data, config):
    assert config.nb_epochs > 2
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    training_dataset = FlowshopDataset(config.epoch_size, config.problem_size, config, training_data)
    validation_dataset = FlowshopDataset(config.network_validation_size, config.problem_size, config, validation_data)
    
    training_dataloader = DataLoader(training_dataset, batch_size=config.batch_size, num_workers=0, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config.batch_size, num_workers=0, shuffle=True)
    
    runtime_values=[]
    cost_values=[]
    predict_solutions=[]
    loss_RCs, loss_KLDs=[],[]
    for epoch_idx in range(1, config.nb_epochs + 1):
        print("epoch {}/{}".format(epoch_idx, config.nb_epochs))
        loss_RC, loss_KLD=train_epoch(model, config, epoch_idx, training_dataloader, optimizer)
        loss_RCs.append(loss_RC)
        loss_KLDs.append(loss_KLD)
        # Validate and save model every 20 epochs
        if epoch_idx==1 or epoch_idx % 20 == 0 or epoch_idx==config.nb_epochs:
            logging.info("Start validation")
            # Evaluate the network performance on the validation set  
            new_bound = evaluate_network(config, VAEModel, validation_dataloader, epoch_idx)
            
            config.search_space_bound = new_bound
            logging.info(f"Setting search space bounds to {new_bound:2.3f}")            
            # Evaluate the search performance of the current model on a small subset of the validation set
            """
         
            """
            runtime_value, cost_value,predict_solution = validate(VAEModel, config,
                                                 validation_data[0][: config.search_validation_size],
                                                 None)
            
            runtime_values.append(runtime_value)
            cost_values.append(cost_value)  
            predict_solutions.append(predict_solution)
           
            #  save the model every __ epoch            
            model_data = {
                    'parameters': model.state_dict(),
                    'code_version': VERSION,
                    'problem': config.problem,
                    'problem_size': config.problem_size,
                    'Z_bound': new_bound,
                    'training_epochs': epoch_idx,
                    'model': "VAE_final"
                    }     


            torch.save(model_data, os.path.join(config.output_path, "models",
                                                    "model_{}_{}.pt".format(run_id, epoch_idx)))
            
    # Save the last model after the end of the training
    torch.save(model_data, os.path.join(config.output_path, "models",
                                        "model_{0}_final.pt".format(run_id, epoch_idx)))
    #logging.info("Training finished")   
    return runtime_values, cost_values,predict_solutions,loss_RCs, loss_KLDs
    
# Validation
def calculateTotalTime(instances,seqs,config):
    instances_= torch.gather(instances, 1, seqs.unsqueeze(2).expand_as(instances))
    flow_time=[]
    n_row_cttable, n_col_cttable = instances_[0].shape
    for instance_id, instance in enumerate(instances_):
        for i in range(n_row_cttable-1):
            for j in range(n_col_cttable-1):
                ct_form, ct_latt = instance[i,j+1], instance[i+1,j]
                if ct_form >= ct_latt:
                    ct_latt = ct_form
                instance[i,j+1], instance[i+1,j] = ct_form, ct_latt    
        total_time= instance.sum(axis=0)[0] + instance.sum(axis=1)[n_row_cttable-1] - instance[n_row_cttable-1, 0]    
        
        #penalty objective function
        """
        penalty=0
        for i in range (len(seqs[instance_id])):
            penalty=pow(10,-i-2)*seqs[instance_id][i]+penalty
        total_time = total_time + penalty           
        """
        
        flow_time.append(total_time.item())
    return torch.Tensor(flow_time).to(config.device).detach()
    
def decode(Z, model, config, instance, cost_fn):
    Z = torch.Tensor(Z).to(config.device)
    with torch.no_grad():
        tour_probs, tour_idx, tour_logp = model.decode(instance, Z, config)
    costs = cost_fn(instance, tour_idx, config)
    return tour_idx, costs.tolist()
    
def validate(model, config, instances):
    model.eval()
    cost_fn=calculateTotalTime
    cost_values = []
    runtime_values = []
    predict_solutions=[]
    for i, instance in enumerate(instances):
        print("validate NO.",i+1)
        start_time = time.time()
        objective_value, solution = solve_instance_de(model, instance, config, cost_fn)
        runtime = time.time() - start_time
        cost_values.append(objective_value)
        predict_solutions.append(solution)
        print("Costs " + str(objective_value))
        runtime_values.append(runtime)
        print(cost_values)
        print(predict_solutions)

    return runtime_values, cost_values,predict_solutions
 
def solve_instance_de(model, instance, config, cost_fn):
    batch_size = config.de_popsize 
    instance = torch.Tensor(instance)
    instance = instance.unsqueeze(0).expand(batch_size, -1, -1)
    instance = instance.to(config.device)
    model.reset_decoder(batch_size, config)

    result_cost, result_tour = minimize(decode, (model, config, instance, cost_fn), config.search_space_bound,
                                        config.search_space_size, popsize=config.de_popsize,
                                        mutate=config.de_mutate, recombination=config.de_recombine,
                                        maxiter=config.search_iterations, maxtime=config.search_timelimit)
    solution = decode(np.array([result_tour] * batch_size), model, config, instance, cost_fn)[0][0].tolist()
    return result_cost, solution

## differential evolution
def minimize(cost_func, args, search_space_bound, search_space_size, popsize, mutate, recombination, maxiter, maxtime):
    
    time_start=datetime.datetime.now()
    print("start:",time_start)
    
    # --- INITIALIZE A POPULATION (step #1) ----------------+
    start_time = time.time()
    population_cost = np.ones((popsize)) * np.inf
    children = np.zeros((popsize, search_space_size))
    iterations_without_improvement = 0
    gen_best = np.inf

    population = np.random.uniform(-search_space_bound, search_space_bound,
                                   (popsize, search_space_size))

    # --- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    for i in range(1, maxiter + 1):
        print(i)
        if time.time() - start_time > maxtime:
            break

        # cycle through each individual in the population
        for j in range(0, popsize):
            # --- MUTATION (step #3.A) ---------------------+

            # select three random vector index positions [0, popsize), not including current vector (j)
            candidates = list(range(0, popsize))
            candidates.remove(j)
            random_index = sample(candidates, 3)

            # subtract x3 from x2, and create a new vector (x_diff)
            x_diff = population[random_index[1]] - population[random_index[2]]

            # multiply x_diff by the mutation factor (F) and add to x_1
            child = population[random_index[0]] + mutate * x_diff

            # --- RECOMBINATION (step #3.B) ----------------+
            crossover = np.random.uniform(0, 1, search_space_size)
            crossover = crossover > recombination
            child[crossover] = population[j][crossover]

            children[j] = child

        # Ensure bounds
        children = np.clip(children, -search_space_bound, search_space_bound)
        
        _, scores_trial = cost_func(children, *args)
        scores_trial = np.array(scores_trial)

        iterations_without_improvement += 1
        if min(population_cost) > min(scores_trial):
            iterations_without_improvement = 0

        improvement = population_cost > scores_trial
        population[improvement] = children[improvement]
        population_cost[improvement] = scores_trial[improvement]

        # --- SCORE KEEPING --------------------------------+
        gen_best = min(population_cost)  # fitness of best individual
   
    time_end=datetime.datetime.now()
    duration=time_end-time_start
    print("end:",time_end)
    print("duration:",duration)
    
    return gen_best, population[np.argmin(population_cost)]
    
# Main
if __name__ == "__main__":

    VERSION = "1.9.0"
    time_start=datetime.datetime.now()
    print("start:",time_start)
    run_id = np.random.randint(10000, 99999)
    now = datetime.datetime.now()
    config = get_config()
    if config.output_path == "":
        config.output_path = os.getcwd()
    config.output_path = os.path.join(config.output_path, "runs", "run_" + str(now.day) + "." + str(now.month) + "." + str(now.year) + "_" + str(run_id))

    os.makedirs(os.path.join(config.output_path, "models"))

    logging.basicConfig(
        filename=os.path.join(config.output_path, "log_" + str(run_id) + ".txt"), filemode='w',
        level=logging.INFO, format='[%(levelname)s]%(message)s')
    logging.info("Started Training Run")
    logging.info("Call: {0}".format(''.join(sys.argv)))
    logging.info("Version: {0}".format(VERSION))
    logging.info("PARAMETERS:")
    for arg in sorted(vars(config)):
        logging.info("{0}: {1}".format(arg, getattr(config, arg)))
    logging.info("----------")    
     
    pickle_file = open(r"/home/nagashijun/cvae/flowdata/process_num=20x6(100000)/instances.pkl",'rb')
    instances=pickle.load(pickle_file)
    pickle_file.close()
     
    pickle_file = open(r"/home/nagashijun/cvae/flowdata/process_num=20x6(100000)/solutions.pkl",'rb')
    solutions=pickle.load(pickle_file)
    pickle_file.close() 
     
    pickle_file = open(r"/home/nagashijun/cvae/flowdata/process_num=20x6(100000)/solution_values.pkl",'rb')
    solution_values=pickle.load(pickle_file)
    pickle_file.close() 
    
    training_data=[instances[:config.epoch_size],solutions[:config.epoch_size]]     
    validation_data=[instances[config.epoch_size:],solutions[config.epoch_size:]]  
     
    VAEModel = VAE(config).to(config.device)
    runtime_values, cost_values, predict_solutions,loss_RCs, loss_KLDs=train(VAEModel, training_data, validation_data, config)    
    
    time_end=datetime.datetime.now()
    print("end:",time_end) 
