import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
import math

class OriginTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, att_type = 'linear', kappa=0.0):
        """
        Args:
            vocab_size (int): Size of vocabulary (N)
            d_model (int): Must be 2*vocab_size (to accommodate basis vectors)
            kappa (float): Constant for weighted sum path. Set kappa = 0 for Chen et al 2015 and kappa = 1 for our baseline
        """
        super().__init__()
        assert d_model >= 2 * vocab_size, "d_model must be at least 2*vocab_size for basis vectors"
        
        # Constants
        self.kappa = kappa
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.att_type = att_type

        # Create basis vectors (non-trainable)
        basis = torch.eye(d_model)  # Shape: [2N, 2N]
        self.E = basis[:self.vocab_size]
        self.E_tilde = basis[self.vocab_size:]
        self.U = self.E.T
        
        # Trainable parameters
        self.F = nn.Parameter(torch.randn(d_model, d_model))
        self.V = nn.Parameter(torch.randn(d_model, d_model))
        self.W = nn.Parameter(torch.randn(d_model, d_model))
        
        # Initialize trainable parameters
        self._init_weights()
    
    
    def _init_weights(self):
        """Initialize trainable parameters"""
        for p in [self.F, self.V, self.W]:
            nn.init.zeros_(p)
            # if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
    
    def forward(self, z):
        """
        Args:
            z: Input token indices, shape [batch_size, H]
        Returns:
            xi: Logits for next token prediction, shape [batch_size, vocab_size]
        """
        batch_size, H = z.shape
        
        # Get E(z_h) using fixed basis
        E_z = self.E[z]  # [batch_size, H, d_model]
        z_shifted = torch.roll(z, shifts=1, dims=1)
        z_shifted[:, 0] = 0  # Padding
        E_tilde_z_prev = self.E_tilde[z_shifted]
        
        
        # Compute x_h = E(z_h) + E_tilde(z_{h-1}) 
        # BUT x_0 (first token) should be just E(z_0)
        x = E_z + E_tilde_z_prev
        x[:, 0, :] = E_z[:, 0, :]  # Override first position
        
        # Focus only on the last position (x_H)
        x_H = x[:, -1, :]  # [batch_size, d_model]
        #print("x_H = ", x_H, "; shape = ", x_H.shape)
        x_1_to_H = x  # All positions [batch_size, H, d_model]
        
        #print("self.W = ", self.W)
        
        # Compute attention scores: x_H^T W x_h
        W_x = torch.matmul(x_1_to_H, self.W.T)  # [batch_size, H, d_model]
        scores = torch.matmul(x_H.unsqueeze(1), W_x.transpose(1, 2))  # [batch_size, 1, H]
        
        # Weighted sum (over sequence length)
        attn_weights = None
        if self.att_type is None:
            assert 'Attention Type must be specificed'
        if self.att_type == 'linear':
            attn_weights = F.linear(scores, torch.eye(H, H))  # σ(x) = x for linear attention 
        elif self.att_type == 'softmax':
            attn_weights = F.softmax(scores, dim=-1)  # for softmax
        elif self.att_type == 'relu':
            attn_weights = F.relu(scores + 1e-8) # for ReLU attention   
        else:
            assert 'Attention Type must be either linear, softmax or relu'
        
        weighted_sum = torch.matmul(attn_weights, x_1_to_H).squeeze(1)  # [batch_size, d_model]
        
        # Compute outputs (for last position only)        
        phi = torch.matmul(self.V, weighted_sum.unsqueeze(2))  # [batch_size, d_model, 1]
        #print(phi.shape, self.U.shape)
        xi_A = torch.matmul(self.U.T, phi)  # [batch_size, vocab_size, 1]
        
        combined = x_H + self.kappa * weighted_sum
        Fc = torch.matmul(self.F, combined.unsqueeze(2))
        xi_F = torch.matmul(self.U.T, Fc)  # [batch_size, vocab_size, 1]
        
        xi = xi_A + xi_F  # [batch_size, vocab_size, 1]
        # ##print("size of xi is ", xi.shape)
        return xi, xi_A, xi_F
        
       
class ReparamTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, trigger_tokens, noise = False, att_type = 'linear', kappa=1.0, gamma = 1.0, trainW = False):
        """
        Args:
            vocab_size (int): Size of vocabulary (N)
            d_model (int): Must be 2*vocab_size (to accommodate basis vectors)
            kappa (float): Constant for weighted sum path. Set kappa = 0 for Chen et al 2015 and kappa = 1 for our baseline
        """
        super().__init__()
        assert d_model >= 2 * vocab_size, "d_model must be at least 2*vocab_size for basis vectors"
        
        # Constants
        self.kappa = kappa
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.att_type = att_type
        self.trigger_tokens = trigger_tokens
        self.Q = len(self.trigger_tokens)
        self.noise = noise
        self.trainW = trainW

        # Create basis vectors (non-trainable)
        # Create identity matrix of size 2N
        basis = torch.eye(d_model)  # Shape: [2N, 2N]
        self.E = basis[:self.vocab_size]
        self.E_tilde = basis[self.vocab_size:2*vocab_size]
        self.U = self.E.T       
        
        
        self.W = nn.Parameter(torch.randn(d_model, d_model)) # if we want to train W directly
        if noise == False:
            self.V = torch.eye(d_model)
            self.EEtilde = []
            for k in self.trigger_tokens:
                # print("self.E[k] shape = ", self.E[k].shape)
                self.EEtilde.append(torch.outer(self.E[k], self.E_tilde[k]))
                
            # Trainable parameters            
            # self.V = nn.Parameter(torch.randn(d_model, d_model))
            self.lambdas = nn.Parameter(torch.randn(self.Q))
            # self.tmp = nn.Parameter(torch.randn(self.Q))
            # self.lambdas = torch.randn(self.Q) # Train with normalized gradient descent
            self.F = nn.Parameter(torch.randn(d_model, d_model))
        else:
            # self.F = nn.Parameter(torch.randn(d_model, d_model))
            self.V = torch.eye(d_model)
            self.q = self.trigger_tokens[0]
            self.tau = vocab_size - 1
            self.Lambda =  nn.Parameter(torch.randn(1))            
        
        # Initialize trainable parameters
        self._init_weights()
    
    def _init_weights(self):
        """Initialize trainable parameters"""
        nn.init.zeros_(self.W)  # if we want to train W directly
        if self.noise == False:
            nn.init.zeros_(self.lambdas) 
            nn.init.zeros_(self.F)            
        else:
            nn.init.zeros_(self.Lambda)            
    
    def forward(self, z):
        """
        Args:
            z: Input token indices, shape [batch_size, H]
        Returns:
            xi: Logits for next token prediction, shape [batch_size, vocab_size]
        """
        batch_size, H = z.shape
        
        # Get E(z_h) using fixed basis
        E_z = self.E[z]  # [batch_size, H, d_model]
        z_shifted = torch.roll(z, shifts=1, dims=1)
        z_shifted[:, 0] = 0  # Padding
        E_tilde_z_prev = self.E_tilde[z_shifted]
        
        
        # Compute x_h = E(z_h) + E_tilde(z_{h-1}) 
        # BUT x_0 (first token) should be just E(z_0)
        x = E_z + E_tilde_z_prev
        x[:, 0, :] = E_z[:, 0, :]  # Override first position
        
        # Focus only on the last position (x_H)
        x_H = x[:, -1, :]  # [batch_size, d_model]
        #print("x_H = ", x_H, "; shape = ", x_H.shape)
        x_1_to_H = x  # All positions [batch_size, H, d_model]
        
        #print("self.W = ", self.W)
        
        # Compute attention scores: x_H^T W x_h
        W = None
        if self.trainW == True:
            W = self.W # if we want to train W directly
        else:
            if self.noise == False:
                W = torch.zeros(self.d_model, self.d_model)
                for i in range(self.Q):
                    # print("self.EEtilde.shape = ", self.EEtilde[i].shape)
                    W += self.lambdas[i] * self.EEtilde[i]
            else:
                W = self.Lambda * torch.outer(self.E[self.q], self.E_tilde[self.q] - self.E[self.tau])

        W_x = torch.matmul(x_1_to_H, W.T)  # [batch_size, H, d_model]
        scores = torch.matmul(x_H.unsqueeze(1), W_x.transpose(1, 2))  # [batch_size, 1, H]
        # assert torch.all(scores >= 0), "Tensor contains negative values!"
        # print(scores)
        
        # Weighted sum (over sequence length)
        attn_weights = None
        if self.att_type is None:
            assert 'Attention Type must be specificed'
        if self.att_type == 'linear':
            attn_weights = F.linear(scores, torch.eye(H, H))  # σ(x) = x for linear attention 
        elif self.att_type == 'softmax':
            attn_weights = F.softmax(scores, dim=-1)  # for softmax
        elif self.att_type == 'relu':
            attn_weights = F.relu(scores + 1e-8) # for ReLU attention   
        else:
            assert 'Attention Type must be either linear, softmax or relu'
        
        weighted_sum = torch.matmul(attn_weights, x_1_to_H).squeeze(1)  # [batch_size, d_model]
        
        # Compute outputs (for last position only)        
        phi = torch.matmul(self.V, weighted_sum.unsqueeze(2))  # [batch_size, d_model, 1]
        #print(phi.shape, self.U.shape)
        xi_A = torch.matmul(self.U.T, phi)  # [batch_size, vocab_size, 1]
        
        xi_F = None
        if self.noise == False:        
            combined = x_H + self.kappa * weighted_sum
            
            Fc = torch.matmul(self.F, combined.unsqueeze(2))
            xi_F = torch.matmul(self.U.T, Fc)  # [batch_size, vocab_size, 1]
            
            # xi = xi_A + xi_F # if we want to train F
            xi = xi_A
        else:
            combined = x_H + self.kappa * weighted_sum
            self.F = torch.outer(self.E[self.tau], self.gamma * self.E[self.q] + self.E_tilde[self.q])
            Fc = torch.matmul(self.F, combined.unsqueeze(2))
            xi_F = torch.matmul(self.U.T, Fc)  # [batch_size, vocab_size, 1]
        
            xi = xi_A + xi_F  # [batch_size, vocab_size, 1]
        return xi, xi_A, xi_F
       
def verifyOriginTransformer():
    # Test case: Single sentence with 3 tokens
    z_test = torch.tensor([[1, 2, 3, 0, 4, 5]])  # (batch_size=1, H=6)
    model = OriginTransformer(vocab_size=10, d_model=20)

    # Check shapes
    E_z = model.E[z_test]
    print(E_z)
    print(E_z.shape)  # Should be torch.Size([1, 3, 20])

    # Verify first token handling
    x = E_z + model.E_tilde[torch.roll(z_test, 1, 1)]
    x[:, 0, :] = E_z[:, 0, :]
    assert torch.allclose(x[:, 0, :], E_z[:, 0, :])
    print(x)
    
    logits = model(z_test)
    print(logits.shape)
    
def verifyReparamTransformer(noise = False):
    if noise == False:
        # Test case: Noiseless, Single sentence with 3 tokens
        z_test = torch.tensor([[1, 2, 3, 0, 4, 5]])  # (batch_size=1, H=6)
        model = ReparamTransformer(vocab_size=10, d_model=20, trigger_tokens = [1, 2])

        # Check shapes
        E_z = model.E[z_test]
        print(E_z)
        print(E_z.shape)  # Should be torch.Size([1, 3, 20])

        # Verify first token handling
        x = E_z + model.E_tilde[torch.roll(z_test, 1, 1)]
        x[:, 0, :] = E_z[:, 0, :]
        assert torch.allclose(x[:, 0, :], E_z[:, 0, :])
        print(x)
        
        logits = model(z_test)
        print(logits.shape)
    else:
        # Test case: Noisy, Single sentence with 3 tokens
        alpha = 0.5
        gamma = math.log(alpha / (1.0 - alpha))
        z_test = torch.tensor([[1, 2, 3, 0, 4, 5]])  # (batch_size=1, H=6)           
        model = ReparamTransformer(vocab_size=10, d_model=20, noise = True, trigger_tokens = [2], gamma = gamma)

        # Check shapes
        E_z = model.E[z_test]
        print(E_z)
        print(E_z.shape)  # Should be torch.Size([1, 3, 20])

        # Verify first token handling
        x = E_z + model.E_tilde[torch.roll(z_test, 1, 1)]
        x[:, 0, :] = E_z[:, 0, :]
        assert torch.allclose(x[:, 0, :], E_z[:, 0, :])
        print(x)
        
        logits = model(z_test)
        print(logits.shape)
    
    
if __name__ == "__main__":
    # verifyOriginTransformer()
    
    verifyReparamTransformer()
    
    # Hyperparameters
    # vocab_size = 20  # Size of vocabulary (N)
    # d_model = 2*vocab_size      # Dimension of embeddings (d)
    # batch_size = 1
    # H = 20

    # # Create model
    # model = CustomTransformer(vocab_size, d_model)

    # # Example input
    # z = torch.randint(0, vocab_size, (batch_size, H))

    # # Forward pass
    # logits = model(z)
    # print(logits.shape)  # Should be (batch_size, vocab_size)