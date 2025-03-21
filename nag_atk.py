from naginterfaces.base.utils import NagValueError
from naginterfaces.library import opt

import matplotlib.pyplot as plt
import numpy as np
import time
import copy

import torch.nn as nn
import torch

import gc

class data():
    pass

class nag_atk():
    def __init__(self, imgs, labels, model,
                 atk_target=None, 
                 mode=1, 
                 p=2, 
                 optimiser='ssqp',
                 lossf=nn.CrossEntropyLoss,
                 con_tol=0.1,
                 in_domain_constraint=True,
                 record_all_x=False,
                 solver_options=None,
                 print_level=2,
                 device=None,
                 ):
        '''
        imgs: torch.tensor 
            has shape [item, channel, vpix, hpix]
            
        labels: list of int
            ground truth / model classication results of imgs

        model: torch.nn.Module
            The model to be attacked
        
        atk_target: int 
            targeted class for the attack. 
            This entry is ignored if mode is set to 2
        
        mode: int
            1: pnorm as obj, targeted con
            2: pnorm as obj, untargeted con
        
        p: int 
            p-norm for computing the perturbation size
        
        optimiser: str
            'ssqp' or 'ipopt'. Please refer to NAG documentation for more details

        lossf: torch.nn.Module
            loss function used to compute the gradient of the constraint function
        
        con_tol: float
            tolerance for the constraint function.
            increase this value if atk failure rate is high
            0.01 or 0.1 can be good choices for ssqp
            0.1 or 1. can be good choices for ipopt
            
        in_domain_constraint: Boolean 
            if True, applies a simple bound that constrain the perturbed imgs 
            to be within [min(imgs), max(imgs)]
            
        solver_options: list of str
            Available option str for each solver is specified in NAG documentation
            
        print_level: int
            For print_level == 2, NAG will print iteration information
            For print_level == 1, only objective and constraint violation values will be printed
            For print_level == 0, nothing will be printed
            Any 'Print Level' specified in solver_option will be overwritten
        '''
        super().__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.imgs = imgs.to(self.device)
        self.imgs_flat = imgs.reshape(imgs.shape[0], -1)
        self.labels = labels
        self.model = model.to(self.device)
        self.optimiser = optimiser
        
        self.atk_batch_size = self.imgs.shape[0]
        self.atk_target = atk_target
        self.con_tol = con_tol
        self.mode = mode
        self.p = p
        self.lossf = lossf()
        self.in_domain_constraint = in_domain_constraint
        self.print_level = print_level
        self.record_all_x = record_all_x
        
        self.scaler = torch.cuda.amp.GradScaler()
        
        # nvar is the size of the image
        self.nvar = np.prod(self.imgs.shape[-3:]) 
        # indices of nonzero grad set to every pixel
        self.idxfd = np.arange(1, self.nvar + 1, dtype=int)
        
        # Initialize handle
        self.handle = opt.handle_init(self.nvar)
        
        # Declear problem type as nonlinear
        opt.handle_set_nlnobj(self.handle, idxfd=self.idxfd) 
        
        # set solver options
        if solver_options is not None:
            solver_options += [f'Print Level = {self.print_level}']
            for option in solver_options:
                opt.handle_opt_set(self.handle, option)
        
        self.objfun = None
        self.objgrd = None
        self.confun = None
        self.congrd = None
        
        # initialise monitor variables
        self.objective_values = []
        self.xs = []
        self.t0 = 0 # a placeholder which will be updated right before the attack starts
        self.ts = []
        self.ccalls = [] # num of constraint function calls
        
        # perturbation result
        self.pert = np.zeros(self.imgs.shape[1:])
        
        # data passed to the solver
        self.data = data()
        self.data.last_iter = [None] * 4
        self.data.best_x = None
        self.data.best_val = np.inf
        self.data.ccall = 0 # number of congrd calls
        self.data.imgs = self.imgs
    
    
    def monitor(self, x, u, rinfo, stats, data=None):
        if self.print_level == 1:
            print(f'{stats[0]} Objective value at iteration: {rinfo[0]:.4e}')
            print(f'{stats[0]} Constraint violation at iteration: {rinfo[1]:.4e}')
        
        data.last_iter = copy.deepcopy([x, u, rinfo, stats])
        
        # keep a record of x with the best obj that also satisfies constraints
        if rinfo[1] <= self.con_tol:
            if rinfo[0] < data.best_val:
              data.best_val = rinfo[0]
              data.best_x = copy.deepcopy(x)
              if self.print_level == 1:
                  print('### Best perturbation updated ###')
 
        self.objective_values.append(rinfo[0])
        if self.record_all_x:
            self.xs.append(copy.deepcopy(x.reshape(self.imgs.shape[-3:])))
            self.ts.append(time.time()-self.t0)
            self.ccalls.append(data.ccall)
    
    # calculates the p norm of x
    # inform, data are required by the solver and unused
    def pnorm(self, x, inform, data=None):
        return np.linalg.norm(x, self.p)**self.p, 0
    
    # gradient of p norm
    # inform, data are required by the solver and unused
    def pnorm_grd(self, x, fdx, inform, data=None):
        if self.p == 1:
            fdx[:] = np.sign(x)
        elif self.p == 2:
            fdx[:] = 2*x
        elif self.p%2 == 0: # for even p >= 3
            fdx[:] = self.p * x**(self.p-1)
        else: # for odd p >= 3 
            fdx[:] = self.p * x**(self.p-1) * np.sign(x)
        return 0
    
    # targeted constraint function
    # ncnln, inform are required by the solver and unused
    def confun1(self, x, ncnln, inform, data=None):
        x = x.reshape(self.imgs.shape) # unflatten x
        pert = torch.tensor(x, device=self.device, dtype=torch.float)
        
        pert_img = data.imgs + pert
        
        with torch.no_grad():  
            pred = self.model(pert_img)
            
            predt = torch.cat((pred[:,:self.atk_target], pred[:,self.atk_target+1:]), axis=1)
            conv = pred[:, self.atk_target] - torch.max(predt, 1).values
            
        return conv.detach().cpu().numpy().astype('float64'), 0
    
    # targeted constraint grad
    # inform is required by the solver and unused
    def congrd1(self, x, gdx, inform, data=None):
        data.ccall += 1
        x = x.reshape(-1, *self.imgs.shape[1:])
        
        if self.atk_batch_size == 1:
            pert = torch.tensor(x, device=self.device, requires_grad=True, dtype=torch.float)
            pert_img = data.imgs + pert

            self.model.zero_grad()
            pred = self.model(pert_img)
            
            # removes the target class from the predictions
            predt = torch.cat((pred[:, :self.atk_target], pred[:, self.atk_target+1:]), axis=1)
            # computes the difference between the target class and the class with max probability 
            conv = pred[:, self.atk_target] - torch.max(predt, 1).values
            
            conv.backward()
            grad = pert.grad
            
            # calculate gradient of the difference wrt perturbation
            gdx[:] = grad.cpu().numpy().astype('float64').flatten()
            
        self.model.zero_grad(set_to_none=True)

        return 0
    
    # constraint function for untargetted attack
    # ncnln, inform are required by the solver and unused
    def confun2(self, x, ncnln, inform, data=None):
        
        # unflatten x
        x = x.reshape(-1, *self.imgs.shape[1:])
        
        pert = torch.tensor(x, device=self.device, dtype=torch.float)
        
        with torch.no_grad():
            # get perturbed img
            pert_img = data.imgs + pert
            # get logits of the prediction
            pred = self.model(pert_img)
            # remove the true class
            predt = torch.cat((pred[:, :self.labels], pred[:, self.labels+1:]), axis=1)
            # compute the difference between the true class and most likely non-true class
            conv = torch.max(predt, 1).values - pred[:, self.labels]
        if self.print_level >= 2:
            print(f'attaking towards class {torch.argmax(predt, 1)}')
                
        return conv.detach().cpu().numpy().astype('float64')[0], 0 # WHY CONV HAS ONE MORE DIM THAN confun1

    # constraint grad for untargeted attack
    # inform is required by the solver and unused
    def congrd2(self, x, gdx, inform, data=None):
        data.ccall += 1
        
        # unflatten x
        x = x.reshape(-1, *self.imgs.shape[1:])
        
        if self.atk_batch_size == 1:
            # get perturbed img
            pert = torch.tensor(x, device=self.device, requires_grad=True, dtype=torch.float)
            # get logits of the prediction
            pert_img = data.imgs + pert
            
            self.model.zero_grad()
            # get logits of the prediction
            pred = self.model(pert_img)
            
            # removes the {input_label} class from the predictions
            predt = torch.cat((pred[:, :self.labels], pred[:, self.labels+1:]), axis=1)
            
            # computes the difference between the {input_label} class and the class with max probability 
            conv = torch.max(predt, 1).values - pred[:, self.labels] 
            
            # do backprop
            gdx[:] = torch.autograd.grad(outputs=conv, inputs=pert)[0].cpu().numpy().astype('float64').flatten()

        return 0
    
    def run_atk(self, x_start=None):
        '''
        Executes the actual attack
        Returns x, u, rinfo, stats, with same meaning as in the NAG documentation
        '''
        if self.mode == 1: # p-norm objfun, targeted confun
            self.objfun = self.pnorm
            self.objgrd = self.pnorm_grd
            self.confun = self.confun1
            self.congrd = self.congrd1
        elif self.mode == 2: # p-norm objfun, untargeted confun
            self.objfun = self.pnorm
            self.objgrd = self.pnorm_grd
            self.confun = self.confun2
            self.congrd = self.congrd2
        else:
            raise Exception('Not a recognised mode')
        
        
        # define bounds for nonlinear constraints 
        bl_nln = np.repeat(self.con_tol, self.atk_batch_size)
        bu_nln = np.repeat(np.inf, self.atk_batch_size)

        # define simple bounds on x
        # a simple bouund on x, s.t. the perturbed pixels have values within input domain
        if self.in_domain_constraint:
            bl_simp = (torch.min(self.imgs_flat)-torch.min(self.imgs_flat, dim=0)[0]).cpu().numpy().astype('float64')
            bu_simp = (torch.max(self.imgs_flat)-torch.max(self.imgs_flat, dim=0)[0]).cpu().numpy().astype('float64')
        else:
            bl_simp = np.repeat(-np.inf, self.nvar)
            bu_simp = np.repeat(np.inf, self.nvar)
        
        # apply defined bounds on design variable
        if (self.optimiser == 'ssqp') or (self.optimiser == 'ipopt'):
            opt.handle_set_nlnconstr(self.handle, 
                                     bl=bl_nln, bu=bu_nln,
                                     irowgd=np.repeat(np.arange(1, self.atk_batch_size+1), self.nvar), 
                                     icolgd=np.tile(np.arange(1, self.nvar+1), self.atk_batch_size),
                                     # irowgd=[1]*nvar, icolgd=list(range(1, nvar + 1)),
                                     )
            opt.handle_set_simplebounds(self.handle, bl=bl_simp, bu=bu_simp)
        
        # initialise design variable if not set
        if x_start is None:
            # x_start = np.random.normal(scale=0.01, size=self.nvar)
            x_start = np.zeros(self.nvar)
        
        # initalise timer
        self.t0 = time.time()
        
        try:
            if self.optimiser == 'ssqp':
                x, u, rinfo, stats = opt.handle_solve_ssqp(self.handle, x_start, 
                                             objfun=self.objfun, objgrd=self.objgrd,
                                             confun=self.confun, congrd=self.congrd,
                                             data=self.data, monit=self.monitor
                                             )
            elif self.optimiser == 'ipopt':
                x, u, rinfo, stats = opt.handle_solve_ipopt(self.handle, x_start, 
                                             objfun=self.objfun, objgrd=self.objgrd,
                                             confun=self.confun, congrd=self.congrd,
                                             data=self.data, monit=self.monitor
                                             )
            else:
                raise Exception('Not an implemented optimiser')
            
        except NagValueError:
            x, u, rinfo, stats = self.data.last_iter
            print('### solver infeasibility exit ###')
            
        if self.data.best_x is not None:
            x = self.data.best_x
        else:
            print('### no solution satisfying all constraints was found ###')
        
        # unflatten x
        x = x.reshape(*self.imgs.shape[1:])
        
        self.pert = x
        
        # Free memory
        opt.handle_free(self.handle)
        gc.collect()
        torch.cuda.empty_cache()
        
        return x, u, rinfo, stats
    
    def evaluate(self):
        with torch.no_grad():
            pred = self.model(self.imgs)
            pred_atk = self.model(self.imgs+torch.tensor(self.pert,dtype=torch.float, device=self.device))
            
        print(f'original: {torch.argmax(pred, axis=1).item()}', 
              f'attacked: {torch.argmax(pred_atk, axis=1).item()}', 
              f'perturbation {self.p}norm**{self.p} {np.linalg.norm(self.pert.flatten(), self.p)**self.p:.06f}',
              sep='\n'
              )
        
        if self.imgs.shape[1] == 3: # if image has RGB colour
            to_plot_x = self.imgs.detach().cpu().numpy()[0].transpose((1,2,0))
            pert = self.pert.transpose((1,2,0))
            to_plot_combined = to_plot_x + pert
        elif self.imgs.shape[1] == 1: # if image is grayscale
            to_plot_x = self.imgs.detach().cpu().numpy()[0][0]
            pert = self.pert[0]
            to_plot_combined = to_plot_x + pert
        else:
            print('Plotting not implemented for this image shape')
            return
        
        to_plot_pert = (pert - pert.min()) / (pert.max() - pert.min())
        
        plt.figure()
        plt.title('original')
        plt.imshow(to_plot_x)
        plt.colorbar()
        plt.show()
        plt.figure()
        plt.title('attacked')
        plt.imshow(to_plot_combined)
        plt.colorbar()
        plt.show()
        plt.figure()
        plt.title('perturbation\nrescaled for clarity')
        plt.imshow(to_plot_pert) 
        plt.colorbar()
        plt.show()

        return pred_atk
    
    def check_success(self):
        '''
        Check whether the atk is successful 
        If the attacked prediction is in agreement with self.atk_class return True, 
            return False otherwise
        If self.atk_class is None, return True as long as the attacked classification
            disagrees with the original unattacked classification, return False otherwise
        '''
        with torch.no_grad():
            pred = self.model(self.imgs)
            pred_atk = self.model(self.imgs+torch.tensor(self.pert,dtype=torch.float, device=self.device))
        
        if self.atk_target is None:
            if torch.argmax(pred) != torch.argmax(pred_atk, dim=1):
                return True
            else:
                return False
        elif self.atk_target == torch.argmax(pred_atk, dim=1):
            return True
        else:
            return False


if __name__ == '__main__':
    pass

