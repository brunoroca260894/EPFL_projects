import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, noise_dim=2, output_dim=2, hidden_dim=100):
        super().__init__()
        # TODO                
        self.s = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )    
        
        #self.linear1 = nn.Linear(noise_dim, hidden_dim)
        #self.linear2 = nn.Linear(hidden_dim, output_dim)
        #self.rele_fun = nn.ReLU()  

    def forward(self, z):
        """
        Evaluate on a sample. The variable z contains one sample per row
        """         
        # x = self.rele_fun( self.linear1( z ) )    
        # x = self.linear2(x)
        return self.s(z)

class DualVariable(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, c=1e-2):
        super().__init__()                            
        self.c=c                 
        self.s = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            )
        
        self.u1 = None
        self.u2 = None
        
        #self.linear1 = nn.Linear(input_dim, hidden_dim)
        #self.linear2 = nn.Linear(hidden_dim, 1)
        #self.rele_fun = nn.ReLU()          
        
        # vectors to compute spectral normalization                                
        with torch.no_grad():    
            counter = 0
            for layer in self.s.modules():                                             
                if isinstance(layer, nn.Linear):                
                    if counter == 0:
                        self.u1 = torch.empty(layer.weight.data.shape[0]).normal_()                                        
                        counter +=1 
                    elif counter ==1:
                        self.u2 = torch.empty(layer.weight.data.shape[0]).normal_()                
                            
    def forward(self, x):
        """
        Evaluate on a sample. The variable x contains one sample per row
        """        
        return self.s(x)

    def enforce_lipschitz(self):
        """Enforce the 1-Lipschitz condition of the function by doing weight clipping or spectral normalization"""
        self.spectral_normalisation() # <= you have to implement this one        
        #self.weight_clipping() <= this one is for another year/only for you as a bonus if you want to compare
                
    def spectral_normalisation(self):
        """
        Perform spectral normalisation, forcing the singular value of the weights to be upper bounded by 1.
        """        
        with torch.no_grad():                
            counter = 0            
            for layer in self.s.modules():                                                
                if isinstance(layer, nn.Linear):                
                    if counter == 0:                        
                        u = self.u1
                        v = layer.weight.data.t() @ u        
                        v = v/v.norm()            
                        u = layer.weight.data @ v
                        u = u/u.norm()                                    
                        layer.weight.data = layer.weight.data/ ( u @ layer.weight.data @ v )
                        self.u1 = u                        
                        counter +=1                                                
                    elif counter ==1:                        
                        u = self.u2
                        v = layer.weight.data.t() @ u        
                        v = v/v.norm()            
                        u = layer.weight.data @ v
                        u = u/u.norm()                                    
                        layer.weight.data = layer.weight.data/ (u @ layer.weight.data @ v)                    
                        self.u2 = u                                                
                   
    def weight_clipping(self):
        """
        Clip the parameters to $-c,c$. You can access a modules parameters via self.parameters().
        Remember to access the parameters  in-place and outside of the autograd with Tensor.data.
        """
        with torch.no_grad():
            for p in self.parameters():
                # TODO
                p.clamp_(-self.c, self.c)                