import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models import OriginTransformer, ReparamTransformer
torch.manual_seed(0)
import numpy as np
np.random.seed(0)
import math

from datetime import datetime
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import os
from cycler import cycler

# Constants
vocab_size = 60  # Size of your vocabulary
d_model = 128       # Dimension of embeddings
batch_size = 512
H = 256
Q = 5
Y = 4

# Define the Dataset Class
class SentenceDataset(Dataset):
    def __init__(self, sentences, next_tokens, output_tokens):
        """
        Args:
            sentences: List of m sentences, each as a list of token indices
            next_tokens: List of m next tokens (as indices)
        """
        self.sentences = sentences
        self.next_tokens = next_tokens
        self.output_tokens = output_tokens
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sentences[idx]), torch.tensor(self.next_tokens[idx], dtype=torch.int64), torch.tensor(self.output_tokens[idx], dtype=torch.int64)

def generate_sentences(vocab_size, H, m, Q, Y, noise = False, alpha = 0.5, testing_generalization = False):
    sentences = []
    next_tokens = []
    output_tokens = []
    isNoisy = np.random.binomial(n = 1, p = alpha, size = m)
    
    if testing_generalization:
        ytest = np.random.randint(Q+Y, vocab_size-1, size = m)
    else:
        ytest = np.random.randint(Q, vocab_size-1, size = m)

    for sentence_id in range(m):        
        if noise == True:
            q = 0
        else:
            q = np.random.randint(0, Q)
            
        y = np.random.randint(Q, Q+Y)
        
        if testing_generalization == True:
            y = ytest[sentence_id] # replace y by a random token outside of the union of the output tokens set and the trigger tokens set
        
        sentence = np.random.randint(Q, vocab_size-1, (H,)).tolist()
        sentence[-1] = q
        
        pos_qy = np.random.randint(0, H-2)
        sentence[pos_qy] = q  
        sentence[pos_qy+1] = y
        
        if noise:
            pos_qtau = -1
            while True:
                pos_qtau = np.random.randint(0, H-2)
                if pos_qtau < pos_qy-1 or pos_qtau > pos_qy + 1:
                    break
            sentence[pos_qtau] = q
            sentence[pos_qtau+1] = vocab_size-1
             
        sentences.append(sentence)
        
        ytrue = 0
        if noise and isNoisy[sentence_id]:            
            ytrue = vocab_size-1
        else:
            ytrue = y
            
        next_tokens.append(ytrue)            
        # print(sentence, ytrue)
        
        output_tokens.append(y)
        
    return sentences, next_tokens, output_tokens
    
    
# Training and Testing Functions
def train_model(model, train_loader, criterion, optimizer, num_epochs, device, test_loader=None, noise = False):
    model.train()
    model.to(device)
    
    # for name, param in model.named_parameters():
        # if param.requires_grad:
            # print(name, param.data)
            
    train_losses = []
    test_losses = []
    test_probs_all = []
    
    brange = torch.arange(512)
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, (sentences, next_tokens, _) in enumerate(train_loader):
            sentences, next_tokens = sentences.to(device), next_tokens.to(device)           
            
            # Forward pass
            logits, xiA, xiF = model(sentences)  # shape: (batch_size, vocab_size, 1)
            if logits.ndim == 3:                
                logits = logits.squeeze(2)                
            # Compute loss            
            loss = criterion(logits, next_tokens)
            
            ## Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()            
            
            ## Normalize all gradients before step()
            for param in model.parameters():
                if param.grad is not None:
                    l2_norm = torch.norm(param.grad, p=2)
                    if l2_norm > 0:
                        param.grad.data.div_(l2_norm)
                        
            optimizer.step()
            
            total_loss += loss.item()
            
            ### Testing Generalization
            test_loss = None
            if test_loader is not None:
                for test_batch_idx, (test_sentences, test_next_tokens, _) in enumerate(test_loader):
                    assert(test_batch_idx == 0) # There should be only one batch while testing
                    cpy = test_next_tokens.detach().clone()
                    test_sentences, test_next_tokens = test_sentences.to(device), test_next_tokens.to(device)                   
                    
                    test_logits, tmp1, tmp2 = model(test_sentences)
                    if test_logits.ndim == 3:                
                        test_logits = test_logits.squeeze(2)                
                    # Compute loss            
                    test_loss = criterion(test_logits, test_next_tokens)
                    
                    batch_size = test_next_tokens.shape[0]
                    Ps = F.softmax(test_logits, dim = -1)
                    
                    if noise == False:
                        if cpy.ndim == 2:
                            cpy.squeeze(1)
                            
                        assert cpy.shape == (batch_size,), f"shape is wrong {cpy.shape}"
                            
                        output_probs = Ps[brange, cpy]
                        test_probs_all.append(output_probs.sum().item() / batch_size)
                    else:                        
                        output_probs = Ps[brange, -1]
                        test_probs_all.append(output_probs.sum().item() / batch_size)
                        
            
            # if batch_idx % 100 == 0:
            train_losses.append(loss.item())
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            if test_loss is not None:            
                test_losses.append(test_loss.item())
                print(f'Test Loss: {test_loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
    # print(model.F)
    # print(model.V)
    # print(model.W)

    return train_losses, test_losses, test_probs_all
    
    
def finite_train_model(model, train_loader, criterion, optimizer, num_epochs, device, test_loader=None, pop_loader=None, noise = False):
    model.train()
    model.to(device)
    
    # for name, param in model.named_parameters():
        # if param.requires_grad:
            # print(name, param.data)
            
    pop_losses = []
    test_losses = []
    list_xiAy = []
    list_xiAmax = []
    list_xiFtau = []
    list_xiFmax = []
    
    batch_size = 512
    brange = torch.arange(batch_size)
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for batch_idx, (sentences, next_tokens, _) in enumerate(train_loader):
            sentences, next_tokens = sentences.to(device), next_tokens.to(device)          
            
            # Forward pass
            logits, xiA, xiF = model(sentences)  # shape: (batch_size, vocab_size, 1)
            if logits.ndim == 3:                
                logits = logits.squeeze(2)                
            # Compute loss            
            loss = criterion(logits, next_tokens)
            
            ## Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()            
            
            ## Normalize all gradients before step()
            for param in model.parameters():
                if param.grad is not None:
                    l2_norm = torch.norm(param.grad, p=2)
                    if l2_norm > 0:
                        param.grad.data.div_(l2_norm)
                        
            optimizer.step()
            
            total_loss += loss.item()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')         
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
        ### Testing Generalization
        test_loss = None
        if test_loader is not None:
            test_loss = 0
            for test_batch_idx, (test_sentences, test_next_tokens, _) in enumerate(test_loader):
                assert(test_batch_idx == 0) # There should be only one batch while testing
                # cpy = test_next_tokens.detach().clone()
                test_sentences, test_next_tokens = test_sentences.to(device), test_next_tokens.to(device)                   
                
                test_logits, tmp1, tmp2 = model(test_sentences)
                if test_logits.ndim == 3:                
                    test_logits = test_logits.squeeze(2)                
                # Compute loss            
                test_loss += criterion(test_logits, test_next_tokens).item()
                
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            print(f'Test Loss: {test_loss:.4f}')
            
        ### Testing Population
        pop_loss = None
        if pop_loader is not None:
            pop_loss = 0
            
            xiAy_sum, xiAmax_sum, xiFtau_sum, xiFmax_sum = 0, 0, 0, 0
            
            for test_batch_idx, (pop_sentences, pop_next_tokens, output_tokens) in enumerate(pop_loader):                
                pop_sentences, pop_next_tokens = pop_sentences.to(device), pop_next_tokens.to(device)  

                assert pop_sentences.shape[0] == batch_size
                
                pop_logits, xiA, xiF = model(pop_sentences)
                if pop_logits.ndim == 3:                
                    pop_logits = pop_logits.squeeze(2)                
                # Compute loss            
                pop_loss += criterion(pop_logits, pop_next_tokens).item()
                
                if xiA.ndim == 3:
                    xiA = xiA.squeeze(2)
                    
                if xiF.ndim == 3:
                    xiF = xiF.squeeze(2)
                    
                if output_tokens.ndim == 2:
                    output_tokens.squeeze(1)
                
                xiAy = xiA[brange, output_tokens]
                xiAmax = xiA.max(dim=-1).values
                
                
                xiFtau = xiF[brange, -1]
                xiFmax = xiF.max(dim=-1).values
                
                xiAy_sum += xiAy.mean().item()
                xiAmax_sum += xiAmax.mean().item()
                xiFtau_sum += xiFtau.mean().item()
                xiFmax_sum += xiFmax.mean().item()
                
            pop_loss /= len(pop_loader)
            pop_losses.append(pop_loss)
            
            print(f'Pop Loss: {pop_loss:.4f}')
            
            xiAy_sum /= len(pop_loader)
            xiAmax_sum /= len(pop_loader)
            xiFtau_sum /= len(pop_loader)
            xiFmax_sum /= len(pop_loader)
            
            print(f'xiAy_sum: {xiAy_sum:.4f}, xiAmax_sum: {xiAmax_sum:.4f}, xiFtau_sum: {xiFtau_sum:.4f}, xiFmax_sum: {xiFmax_sum:.4f}')
            
            list_xiAy.append(xiAy_sum)
            list_xiAmax.append(xiAmax_sum)
            list_xiFtau.append(xiFtau_sum)
            list_xiFmax.append(xiFmax_sum)
            
            
    # print(model.F)
    # print(model.V)
    # print(model.W)
    
    
    return pop_losses, test_losses, list_xiAy, list_xiAmax, list_xiFtau, list_xiFmax
    

# Main Training Setup

def getdatafname(seed, noise, alpha, nsteps):
    train_sentences_fname = f'data/seed{seed}-noise{noise}-alpha{alpha:.1f}-nsteps{nsteps}-train-sentences.npy'
    train_next_tokens_fname = f'data/seed{seed}-noise{noise}-alpha{alpha:.1f}-nsteps{nsteps}-train-next-tokens.npy'
    train_output_tokens_fname = f'data/seed{seed}-noise{noise}-alpha{alpha:.1f}-nsteps{nsteps}-train-output-tokens.npy'
    
    test_sentences_fname = f'data/seed{seed}-noise{noise}-alpha{alpha:.1f}-nsteps{nsteps}-test-sentences.npy'
    test_next_tokens_fname = f'data/seed{seed}-noise{noise}-alpha{alpha:.1f}-nsteps{nsteps}-test-next-tokens.npy'
    test_output_tokens_fname = f'data/seed{seed}-noise{noise}-alpha{alpha:.1f}-nsteps{nsteps}-test-output-tokens.npy'
    
    return train_sentences_fname, train_next_tokens_fname, train_output_tokens_fname, test_sentences_fname, test_next_tokens_fname, test_output_tokens_fname
    
def getfname(seed, noise, model_type, att_type, kappa, alpha, learning_rate, nsteps = 0):
    train_losses_fname = f'logs/seed{seed}-train-losses-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}.npy'
    test_losses_fname = f'logs/seed{seed}-test-losses-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}.npy'
    model_fname = f'logs/seed{seed}-Model-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}.pth'
    test_probs_all_fname = f'logs/seed{seed}-test-probs-all-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}.npy'
    
    return train_losses_fname, test_losses_fname, model_fname, test_probs_all_fname
    
def getdatafinitefname(seed, noise, alpha, finite_nsteps):
    finite_train_sentences_fname = f'data/finite-seed{seed}-noise{noise}-alpha{alpha:.1f}-nsteps{finite_nsteps}-train-sentences.npy'
    finite_train_next_tokens_fname = f'data/finite-seed{seed}-noise{noise}-alpha{alpha:.1f}-nsteps{finite_nsteps}-train-next-tokens.npy'
    finite_train_output_tokens_fname = f'data/finite-seed{seed}-noise{noise}-alpha{alpha:.1f}-nsteps{finite_nsteps}-train-output-tokens.npy'
    
    return finite_train_sentences_fname, finite_train_next_tokens_fname, finite_train_output_tokens_fname
    
def getfnamefinite(seed, noise, model_type, att_type, kappa, alpha, learning_rate, finite_nsteps):
    pop_losses_fname = f'logs/finite-pop-losses-seed{seed}-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}-nsteps{finite_nsteps}.npy'
    test_losses_fname = f'logs/finite-test-losses-seed{seed}-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}-nsteps{finite_nsteps}.npy'
    model_fname = f'logs/finite-Model-seed{seed}-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}-nsteps{finite_nsteps}.pth'
    
    xiAy_fname = f'logs/finite-xiAy-seed{seed}-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}-nsteps{finite_nsteps}.npy'
    xiAmax_fname = f'logs/finite-xiAmax-seed{seed}-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}-nsteps{finite_nsteps}.npy'
    xiFtau_fname = f'logs/finite-xiFtau-seed{seed}-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}-nsteps{finite_nsteps}.npy'
    xiFmax_fname = f'logs/finite-xiFmax-seed{seed}-model{model_type}-att{att_type}-kappa{kappa}-noise{noise}-alpha{alpha:.1f}-lr{learning_rate:.1f}-nsteps{finite_nsteps}.npy'
    
    return pop_losses_fname, test_losses_fname, model_fname, xiAy_fname, xiAmax_fname, xiFtau_fname, xiFmax_fname
    
def plot_graph(list_values, list_names, xlabel = 'Iteration', ylabel = '', plot_title = '', save_path=None, dpi=300, file_format='pdf', ymax = None):
    """
    Create a line plot for each set of values in list_values, with legends from list_names.
    
    Args:
        list_values (list of lists): List containing N lists of values to plot
        list_names (list of str): List containing N legend names
    """
    plt.rcParams.update({
    'font.size': 12,                  # Base font size
    'axes.titlesize': 16,             # Title font size
    'axes.labelsize': 12,             # Axis label font size
    'xtick.labelsize': 11,            # X-axis tick label size
    'ytick.labelsize': 11,            # Y-axis tick label size
    'legend.fontsize': 12,            # Legend font size
    'figure.autolayout': True,        # Auto-adjust layout
    'axes.spines.top': False,         # Remove top frame line
    'axes.spines.right': False,       # Remove right frame line
    'savefig.bbox': 'tight',          # Tight bounding box
    'savefig.pad_inches': 0.05,       # Padding when saving
    'lines.linewidth': 1.5,           # Line thickness
    'axes.linewidth': 1.0,             # Axis line width
    "text.usetex": True,          # Enable LaTeX rendering
    "font.family": "serif",       # Use serif font (like LaTeX)
    })

    N = len(list_names)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))
    styles = ['-o', '--s', '-.^', ':D']  # Combined linestyle+marker
    
    # Plot each line
    for i in range(N):
        ax.plot(list_values[i], styles[i % len(styles)], label=list_names[i], markersize=0.25)
    
    # Add legend, labels, and grid
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ymax is not None:
        plt.ylim(0, ymax)
    
    ax.set_title(plot_title)
    ax.grid(True, linestyle='--', alpha=0.7) 
    
    
    
    # Add tight layout
    plt.tight_layout()
    
    if save_path:
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + '.' + file_format, 
                   dpi=dpi, 
                   format=file_format, 
                   bbox_inches='tight',
                   pad_inches=0.1)
        print(f"Saved high-res plot to: {save_path}")
    else:
        # Show the plot
        plt.show()
    plt.close()

    
def visualize_fig1(seed = 0, alpha = 0.5, learning_rate = 0.05, nsteps = 2000):    
    list_names = ['Reparam-Linear-FA', 'Reparam-Softmax-FA', 'Reparam-Linear-W-FA', 'Origin-Linear-FA', 'Origin-Softmax-F']
    fnames = [' '] * 5
    for noise in [False, True]:
        real_alpha = 0 if noise == False else alpha
        fnames[0] = getfname(seed = seed, noise = noise, model_type = 'reparam', att_type = 'linear', kappa = 1, alpha = real_alpha, learning_rate = learning_rate, nsteps = nsteps)
        fnames[1] = getfname(seed = seed, noise = noise, model_type = 'reparam', att_type = 'softmax', kappa = 1, alpha = real_alpha, learning_rate = learning_rate, nsteps = nsteps)
        fnames[2] = getfname(seed = seed, noise = noise, model_type = 'reparamW', att_type = 'linear', kappa = 1, alpha = real_alpha, learning_rate = learning_rate, nsteps = nsteps)
        fnames[3] = getfname(seed = seed, noise = noise, model_type = 'origin', att_type = 'linear', kappa = 1, alpha = real_alpha, learning_rate = learning_rate, nsteps = nsteps)
        fnames[4] = getfname(seed = seed, noise = noise, model_type = 'origin', att_type = 'softmax', kappa = 0, alpha = real_alpha, learning_rate = learning_rate, nsteps = nsteps)
    
        list_train_losses = []
        for i in range(5):
            train_losses = np.load(fnames[i][0])
            list_train_losses.append(train_losses)
            
        plot_graph(list_train_losses, list_names, ylabel = 'Loss', plot_title='Population Loss', save_path=f'fig-train-noise{noise}')
        
        list_test_losses = []
        for i in range(5):
            test_losses = np.load(fnames[i][1])
            list_test_losses.append(test_losses)
            
        plot_graph(list_test_losses, list_names, ylabel = 'Loss', plot_title='Unseen Output Test Loss', save_path=f'fig-test-noise{noise}')
        
        list_test_probs_all = []
        for i in range(2):
            test_probs_all = np.load(fnames[i][3])
            list_test_probs_all.append(test_probs_all)
            
        list_prob_names = list_names[:2]
        list_prob_names.append('Expected')
        
        if noise == False:
            list_test_probs_all.append([1] * nsteps)
            plot_graph(list_test_probs_all, list_prob_names, ylabel = 'Prob', plot_title='Pr(Output Token)', save_path=f'fig-prob-noise{noise}', ymax = 1)
        else:
            list_test_probs_all.append([real_alpha] * nsteps)           
            plot_graph(list_test_probs_all, list_prob_names, ylabel = 'Prob', plot_title='Pr(Noise Token)', save_path=f'fig-prob-noise{noise}', ymax = 1)
            
            
def visualize_fig2(seed = 0, alpha = 0.5, learning_rate = 0.05, finite_nsteps = 4):    
    noise = True
    
    list_names = ['Reparam-Relu-FA', 'Reparam-Relu-W-FA', 'Origin-ReLU-FA']
    fnames = [' '] * 3
    
    fnames[0] = getfnamefinite(seed = seed, noise = noise, model_type = 'reparam', att_type = 'relu', kappa = 1, alpha = alpha, learning_rate = learning_rate, finite_nsteps = finite_nsteps)
    fnames[1] = getfnamefinite(seed = seed, noise = noise, model_type = 'reparamW', att_type = 'relu', kappa = 1, alpha = alpha, learning_rate = learning_rate, finite_nsteps = finite_nsteps)
    fnames[2] = getfnamefinite(seed = seed, noise = noise, model_type = 'origin', att_type = 'relu', kappa = 1, alpha = alpha, learning_rate = learning_rate, finite_nsteps = finite_nsteps)
    
    list_pop_losses = []
    for i in range(3):
        pop_losses = np.load(fnames[i][0])
        list_pop_losses.append(pop_losses)
        
    plot_graph(list_pop_losses, list_names, xlabel='Epoch', ylabel = 'Loss', plot_title=r'Population Loss ($\alpha={}$)'.format(alpha), save_path=f'fig-finite-pop-noise{noise}-alpha{alpha:.1f}')
    
    list_test_losses = []
    for i in range(3):
        test_losses = np.load(fnames[i][1])
        list_test_losses.append(test_losses)
        
    plot_graph(list_test_losses, list_names, xlabel='Epoch', ylabel = 'Loss', plot_title=r'Unseen Output Test Loss ($\alpha={}$)'.format(alpha), save_path=f'fig-finite-test-noise{noise}-alpha{alpha:.1f}')
        
def visualize_each_model(seed, model_type, att_type, kappa, learning_rate, nsteps): 
    entropy05 = math.log(2)
    entropy02 = -(0.2 * math.log(0.2) + 0.8 * math.log(0.8))

    fnames = [''] * 4
    fnames[0] = getfname(seed = seed, noise = False, model_type = model_type, att_type = att_type, alpha = 0.5, kappa = kappa, learning_rate = learning_rate, nsteps = nsteps)
    fnames[1] = getfname(seed = seed, noise = True, model_type = model_type, att_type = att_type, alpha = 0.2, kappa = kappa, learning_rate = learning_rate, nsteps = nsteps)
    fnames[2] = getfname(seed = seed, noise = True, model_type = model_type, att_type = att_type, alpha = 0.5, kappa = kappa, learning_rate = learning_rate, nsteps = nsteps)
    fnames[3] = getfname(seed = seed, noise = True, model_type = model_type, att_type = att_type, alpha = 0.8, kappa = kappa, learning_rate = learning_rate, nsteps = nsteps)
    
    list_names = []
    kappaStr = ['F', 'FA']
    if model_type != 'reparamW':
        prefix = model_type.capitalize() + '-' + att_type.capitalize() + '-' + kappaStr[kappa]
    else:
        prefix = 'Reparam-Linear-W-FA'

    list_names = [''] * 6
    alphas = [0, 0.2, 0.5, 0.8]
    for i in range(4):
        list_names[i] = prefix + f'(noise = {alphas[i]:.1f})'        
    list_names[4] = 'Entropy of Ber(0.2)'
    list_names[5] = 'Entropy of Ber(0.5)'
    
    list_train_losses = []
    for i in range(4):
        train_losses = np.load(fnames[i][0])
        list_train_losses.append(train_losses)
        
    list_train_losses.append([entropy02] * nsteps)
    list_train_losses.append([entropy05] * nsteps)
        
    plot_graph(list_train_losses, list_names, ylabel = 'Loss', plot_title=f'Population Loss, learning rate = {learning_rate:.1f}', save_path=f'viz/fig-seed-{seed}-{model_type}-{att_type}-kappa{kappa}-lr{learning_rate:.1f}-{nsteps}-poploss')
    
    list_test_losses = []
    for i in range(4):
        test_losses = np.load(fnames[i][1])
        list_test_losses.append(test_losses)
        
    list_test_losses.append([entropy02] * nsteps)
    list_test_losses.append([entropy05] * nsteps)
        
    plot_graph(list_test_losses, list_names, ylabel = 'Loss', plot_title='Unseen Output Test Loss', save_path=f'viz/fig-seed-{seed}-{model_type}-{att_type}-kappa{kappa}-lr{learning_rate:.1f}-{nsteps}-testloss')
    
def visualize_layer_wise(seed, alphas, learning_rate = 0.1, finite_nsteps = 4):
    noise = True
    
    list_names = ['Reparam-Linear-FA', 'Reparam-Linear-W-FA', 'Origin-Linear-FA']
    fnames = [' '] * 3
    
    for alpha in alphas:
        fnames[0] = getfnamefinite(seed = seed, noise = noise, model_type = 'reparam', att_type = 'linear', kappa = 1, alpha = alpha, learning_rate = learning_rate, finite_nsteps = finite_nsteps)
        fnames[1] = getfnamefinite(seed = seed, noise = noise, model_type = 'reparamW', att_type = 'linear', kappa = 1, alpha = alpha, learning_rate = learning_rate, finite_nsteps = finite_nsteps)
        fnames[2] = getfnamefinite(seed = seed, noise = noise, model_type = 'origin', att_type = 'linear', kappa = 1, alpha = alpha, learning_rate = learning_rate, finite_nsteps = finite_nsteps)
        
        entropyalpha = -(alpha * math.log(alpha) + (1-alpha) * math.log(1-alpha))
        
        for j in range(3):
            pop_losses_fname, test_losses_fname, tmp, xiAy_fname, xiAmax_fname, xiFtau_fname, xiFmax_fname = fnames[j]
            pop_losses = np.load(pop_losses_fname)
            test_losses = np.load(test_losses_fname)
            
            list_xiAy = np.load(xiAy_fname)
            list_xiAmax = np.load(xiAmax_fname)
            list_xiFtau = np.load(xiFtau_fname)
            list_xiFmax = np.load(xiFmax_fname)
            
            plot_graph([list_xiAy, list_xiAmax], ['Attention Logits of Output Tokens', 'Maximum Value of Attention Logits'], plot_title=r'$\xi_{A,y}$ versus $\max_{j} \xi_{A,j}$', save_path=f'viz/fig-layerwise-finite-logits-attention-{list_names[j]}-alpha{alpha:.1f}')
            
            plot_graph([list_xiFtau, list_xiFmax], ['Feed-forward Logits of Noise Tokens', 'Maximum Value of Feed-forward Logits'], plot_title=r'$\xi_{F,y}$ versus $\max_{j} \xi_{F,j}$', save_path=f'viz/fig-layerwise-finite-logits-feedforward-{list_names[j]}-alpha{alpha:.1f}')
            
            plot_graph([pop_losses, test_losses, [entropyalpha] * len(pop_losses)], ['Population Loss', 'Unseen Output Test Loss', f'Entropy of Ber({alpha})'], plot_title='Population and Unseen Output Test Losses', save_path=f'viz/fig-layerwise-finite-train-test-loss-{list_names[j]}-alpha{alpha:.1f}')
    
    
def generate_data(seed, noise, alpha, nsteps = 0, finite_nsteps = 0):
    directory_path = './data/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")
    
    if nsteps > 0:
        train_sentences_fname, train_next_tokens_fname, train_output_tokens_fname, test_sentences_fname, test_next_tokens_fname, test_output_tokens_fname = getdatafname(seed, noise, alpha, nsteps)
        
        if os.path.exists(train_sentences_fname):
            print("Population and Test Data already exist")            
        else:        
            m = batch_size*nsteps  # Number of sentences        
            
            sentences, next_tokens, output_tokens = generate_sentences(vocab_size, H, m, Q, Y, noise, alpha)
            np.save(train_sentences_fname, sentences)
            np.save(train_next_tokens_fname, next_tokens)
            np.save(train_output_tokens_fname, output_tokens)
            
            test_sentences, test_next_tokens, test_output_tokens = generate_sentences(vocab_size, H, batch_size, Q, Y, noise, alpha, testing_generalization = True)
            np.save(test_sentences_fname, test_sentences)
            np.save(test_next_tokens_fname, test_next_tokens)
            np.save(test_output_tokens_fname, test_output_tokens)
    
    if finite_nsteps > 0:   
        finite_train_sentences_fname, finite_train_next_tokens_fname, finite_train_output_tokens_fname = getdatafinitefname(seed, noise, alpha, finite_nsteps)
        
        if os.path.exists(finite_train_sentences_fname):
            print("Finite Training Data already exist")
        else:       
            m = batch_size*finite_nsteps # Number of sentences in the finite set     
        
            sentences, next_tokens, output_tokens = generate_sentences(vocab_size, H, m, Q, Y, noise, alpha)
            np.save(finite_train_sentences_fname, sentences)
            np.save(finite_train_next_tokens_fname, next_tokens)
            np.save(finite_train_output_tokens_fname, output_tokens)

    
def main_population_loss(seed, noise, model_type, att_type, kappa, alpha = 0.5, learning_rate = 0.05, nsteps = 2000, num_epochs = 1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_sentences_fname, train_next_tokens_fname, train_output_tokens_fname, test_sentences_fname, test_next_tokens_fname, test_output_tokens_fname = getdatafname(seed, noise, alpha, nsteps)
    
    if os.path.exists(train_sentences_fname) and os.path.exists(train_next_tokens_fname) and os.path.exists(train_output_tokens_fname) and os.path.exists(test_sentences_fname) and os.path.exists(test_next_tokens_fname) and os.path.exists(test_output_tokens_fname):        
        sentences = np.load(train_sentences_fname)
        next_tokens = np.load(train_next_tokens_fname)
        output_tokens = np.load(train_output_tokens_fname)
        
        test_sentences = np.load(test_sentences_fname)
        test_next_tokens = np.load(test_next_tokens_fname)
        test_output_tokens = np.load(test_output_tokens_fname)
    else:
        assert False, 'Run train.py -g with nsteps > 0 to generate data first'
    
    
    # Create dataset and dataloader
    dataset = SentenceDataset(sentences, next_tokens, output_tokens)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = SentenceDataset(test_sentences, test_next_tokens, test_output_tokens)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss, and optimizer
    if model_type == 'origin':
        model = OriginTransformer(vocab_size, d_model, att_type = att_type, kappa=kappa)
    elif model_type == 'reparam' or model_type == 'reparamW':    
        gamma = 0
        if alpha > 0:
            gamma = math.log(alpha / (1.0 - alpha))            
            
        trainW = True if model_type == 'reparamW' else False
        if noise == True:            
            model = ReparamTransformer(vocab_size, d_model, att_type = att_type, trigger_tokens = [0], noise = noise, gamma = gamma, kappa=kappa, trainW = trainW)
        else:
            model = ReparamTransformer(vocab_size, d_model, att_type = att_type, trigger_tokens = list(range(Q)), noise = noise, kappa=kappa, trainW = trainW)
    else:
        assert False, f'{model_type} is invalid'
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Pure Gradient Descent
    
    # Train the model
    train_losses, test_losses, test_probs_all = train_model(model, train_loader, criterion, optimizer, num_epochs, device, test_loader = test_loader, noise = noise)
    
    # Save the losses and model   
    train_losses_fname, test_losses_fname, model_fname, test_probs_all_fname = getfname(seed, noise, model_type, att_type, kappa, alpha, learning_rate, nsteps)
    
    np.save(train_losses_fname, train_losses)
    np.save(test_losses_fname, test_losses)
    torch.save(model.state_dict(), model_fname)    
    
    np.save(test_probs_all_fname, test_probs_all)
    
def main_finite(seed, noise, model_type, att_type, kappa, alpha = 0.5, learning_rate = 0.05, nsteps = 2000, finite_nsteps = 4, num_epochs = 100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Population and Test Set
    full_sentences_fname, full_next_tokens_fname, full_output_tokens_fname, test_sentences_fname, test_next_tokens_fname, test_output_tokens_fname = getdatafname(seed, noise, alpha, nsteps)    

    m = batch_size*finite_nsteps # Number of sentences in the finite set     
    
    finite_train_sentences_fname, finite_train_next_tokens_fname, finite_train_output_tokens_fname = getdatafinitefname(seed, noise, alpha, finite_nsteps)
    
    if os.path.exists(full_sentences_fname) and os.path.exists(full_next_tokens_fname) and os.path.exists(full_output_tokens_fname) and os.path.exists(test_sentences_fname) and os.path.exists(test_next_tokens_fname) and os.path.exists(test_output_tokens_fname) and os.path.exists(finite_train_sentences_fname) and os.path.exists(finite_train_next_tokens_fname) and os.path.exists(finite_train_output_tokens_fname):    
    
        poplen = batch_size*nsteps // 50
        
        full_sentences = np.load(full_sentences_fname)[:poplen]
        full_next_tokens = np.load(full_next_tokens_fname)[:poplen]
        full_output_tokens = np.load(full_output_tokens_fname)[:poplen]
        
        test_sentences = np.load(test_sentences_fname)
        test_next_tokens = np.load(test_next_tokens_fname)
        test_output_tokens = np.load(test_output_tokens_fname)

        sentences = np.load(finite_train_sentences_fname)
        next_tokens = np.load(finite_train_next_tokens_fname)
        output_tokens = np.load(finite_train_output_tokens_fname)
        
    else:
        assert False, 'Run train.py -g with nsteps > 0 and finite_nsteps > 0 to generate data first'
    
    mtau = np.sum(next_tokens == vocab_size-1)
    alpha_hat = mtau * 1.0 / m
    
    # Create dataset and dataloader
    dataset = SentenceDataset(sentences, next_tokens, output_tokens)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)    
    
    test_dataset = SentenceDataset(test_sentences, test_next_tokens, test_output_tokens)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    pop_dataset = SentenceDataset(full_sentences, full_next_tokens, full_output_tokens)
    pop_loader = DataLoader(pop_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss, and optimizer
    if model_type == 'origin':
        model = OriginTransformer(vocab_size, d_model, att_type = att_type, kappa=kappa)
    elif model_type == 'reparam' or model_type == 'reparamW':   
        gamma = 0
        if alpha_hat > 0:
            gamma = math.log(alpha_hat / (1.0 - alpha_hat))            
            
        trainW = True if model_type == 'reparamW' else False
        if noise == True:            
            model = ReparamTransformer(vocab_size, d_model, att_type = att_type, trigger_tokens = [0], noise = noise, gamma = gamma, kappa=kappa, trainW = trainW)
        else:
            model = ReparamTransformer(vocab_size, d_model, att_type = att_type, trigger_tokens = list(range(Q)), noise = noise, kappa=kappa, trainW = trainW)
    else:
        assert False, f'{model_type} is invalid'
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_results = finite_train_model(model, train_loader, criterion, optimizer, num_epochs, device, test_loader = test_loader, pop_loader = pop_loader, noise = noise)
    pop_losses, test_losses = train_results[:2]
    list_xiAy, list_xiAmax, list_xiFtau, list_xiFmax = train_results[2:]

    # Save the losses and model    
    fnames = getfnamefinite(seed, noise, model_type, att_type, kappa, alpha, learning_rate, finite_nsteps)
    pop_losses_fname, test_losses_fname, model_fname = fnames[:3]
    
    xiAy_fname, xiAmax_fname, xiFtau_fname, xiFmax_fname = fnames[3:]
    
    np.save(pop_losses_fname, pop_losses)
    np.save(test_losses_fname, test_losses)
    
    np.save(xiAy_fname, list_xiAy)
    np.save(xiAmax_fname, list_xiAmax)
    np.save(xiFtau_fname, list_xiFtau)
    np.save(xiFmax_fname, list_xiFmax)
    
    # torch.save(model.state_dict(), model_fname)    
    # np.save(test_probs_all_fname, test_probs_all)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default = 0)
    parser.add_argument("-n", "--noise", type=int)
    parser.add_argument("-m", "--model_type", type=str)
    parser.add_argument("-a", "--att_type", type=str)
    parser.add_argument("-k", "--kappa", type=int)
    parser.add_argument("-alpha", "--alpha", type=float, default = 0)
    parser.add_argument("-lr", "--lr", type=float, default = 0.1)
    parser.add_argument("-g", "--generate", type=int, default=0)
    parser.add_argument("-nst", "--nsteps", type=int, default=2000)
    parser.add_argument("-fnst", "--finite_nsteps", type=int, default=4)
    parser.add_argument("-f", "--finite", type=int, default = 0)
    parser.add_argument("-v", "--visualize", type=int, default = 0)
    parser.add_argument("-pl", "--plot", type=int, default = -1)    

    args = parser.parse_args()
    
    start_time = datetime.now()
    
    # print(args.seed, args.noise, args.model_type, args.att_type, args.kappa)
    
    noise = True
    if args.noise == 0:
        noise = False    
        args.alpha = 0       
    else:
        assert args.alpha > 0 and args.alpha < 1
            
    if args.generate == 1:
        print(f'Generating data with nsteps = {args.nsteps} and finite_nsteps = {args.finite_nsteps}')
        generate_data(seed = args.seed, noise = noise, alpha = args.alpha, nsteps = args.nsteps, finite_nsteps = args.finite_nsteps)      
    elif args.visualize == 1:
        if args.plot == 0:
            visualize_fig1(alpha = args.alpha, learning_rate = args.lr)
        elif args.plot == 1:
            visualize_fig2(alpha = args.alpha, finite_nsteps = 4)
        elif args.plot == 2:
            visualize_each_model(seed = args.seed, model_type = args.model_type, att_type = args.att_type, kappa = args.kappa, learning_rate = args.lr, nsteps = args.nsteps)
        elif args.plot == 3:
            visualize_layer_wise(seed = args.seed, alphas = [0.2, 0.5, 0.8], learning_rate = 0.1, finite_nsteps = 4)
        else:
            assert False, 'Invalid visualize parameters'
    elif args.finite == 0:            
        main_population_loss(
        seed = args.seed, 
        noise = noise, 
        model_type = 
        args.model_type, 
        att_type = args.att_type, 
        kappa = args.kappa, 
        alpha = args.alpha, 
        learning_rate = args.lr, 
        nsteps = args.nsteps)
    else:
        main_finite(
        seed = args.seed, 
        noise = noise, 
        model_type = args.model_type, 
        att_type = args.att_type, 
        kappa = args.kappa, 
        alpha = args.alpha, 
        learning_rate = args.lr, 
        nsteps = args.nsteps)
        
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
        
    
    
    