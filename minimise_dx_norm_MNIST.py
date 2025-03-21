from data import load_MNIST, remove_inaccurate_imgs
from nag_atk import nag_atk
from utils import set_seed
from model import Net

import torch

set_seed(42)

# Uncomment if error related to libiomp5md.dll initialisation occurs
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net()
model.load_state_dict(torch.load('./lenet_mnist_model.pth', map_location=device))
model.eval()

# only keep images of the number 7
imgs, labels = load_MNIST(filter_label=7) 
# remove wrongly classified images
imgs, labels = remove_inaccurate_imgs(imgs, labels, model) 
imgs, labels = imgs[:1], labels[:1] 


solver_options = ['Print Solution = NO',
                  
                  # Parameters for ssqp
                  'SSQP Iteration Limit = 50000',
                  'SSQP Minor Iteration Limit = 100',
                  'SSQP Major Iteration Limit = 100', 
                  'SSQP Monitor Frequency = 1',
                  'SSQP Hessian = LIMITED-MEMORY', 
                  'SSQP Hessian Updates = 5',
                  'SSQP Major Feasibility Tol = 0.01',
                  'SSQP Major Optimality Tol = 0.01',
                  'SSQP Minor Feasibility Tol = 0.01',
                  
                  # Parameters for ipopt
                  'Outer Iteration Limit = 100', 
                  'Stop Tolerance 1 = 0.1',
                  ]


attacker = nag_atk(imgs, labels, model,
                   atk_target=3, # perturb imgs to number 3
                   mode=1, # 1 for targeted, 3 for untargeted
                   optimiser='ipopt',
                   p=2,
                   in_domain_constraint=True,
                   solver_options=solver_options,
                   device=device,
                   print_level=1,
                   )

x, u, rinfo, stats = attacker.run_atk()

attacker.evaluate()




