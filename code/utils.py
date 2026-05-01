import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

########################################################
########################################################
##################### helpers ##########################
########################################################
########################################################

class low_rank_rnn(nn.Module):
    def __init__(self, N=100, scale=1, rank=1, phi='erf', dt=1.0, tau=1.0):
        
        super().__init__()
        self.N, self.dt, self.tau = N, dt, tau

        if phi == 'erf':
            self.alpha = np.sqrt(np.pi) / 2.0
            self.phi = erf_alpha(self.alpha)
        else: # linear 
            self.alpha = 1.0
            self.phi = lambda x: x
        
        self.m = nn.Parameter(scale * torch.randn(N, 1), requires_grad=True)
        self.u = nn.Parameter(scale * torch.randn(N, rank), requires_grad=True)
        self.v = nn.Parameter(scale * torch.randn(N, rank), requires_grad=True)
        self.z = nn.Parameter(scale * torch.randn(N, 1), requires_grad=True)

    def forward(self, x_t, h):
        W = (1.0 / self.N) * (self.u @ self.v.T) 
        phi_h = self.phi(h) 
        h_update = W @ phi_h + self.m @ x_t
        h = h + (self.dt / self.tau) * (-h + h_update)
        y = (1.0 / self.N) * (self.z.T @ phi_h) 
                
        return y, h

def erf_alpha(alpha):
    return lambda x: torch.erf(alpha * x)


def normalize(x):
    return x / torch.norm(x, p=2)

def initialize_vectors(N=100, scale=1, n_in=1, n_out=1, rank=1):
    m = scale * torch.randn((N, n_in))
    u = scale * torch.randn((N, rank))
    v = scale * torch.randn((N, rank))
    z = scale * torch.randn((N, n_out))

    return m,u,v,z


def initialize_vectors_balanced(N, scale=1):
    eps = 1e-3   # similarity threshold
    
    while True:
        m,u,v,z = initialize_vectors(N, scale)
    
        m = normalize(m) * np.sqrt(N)
        u = normalize(u) * np.sqrt(N)
        v = normalize(v) * np.sqrt(N)
        z = normalize(z) * np.sqrt(N)
    
        # compute scalar quantities
        a1 = (z.T @ u / N).item()
        b1 = (v.T @ m / N).item()
    
        a2 = (m.T @ u / N).item()
        b2 = (z.T @ v / N).item()
    
        # conditions
        similar_1 = abs(a1 - b1) < eps and a1 * b1 > 0
        similar_2 = abs(a2 - b2) < eps and a2 * b2 > 0
    
        if similar_1 and similar_2:
            print("Breaking condition met!")
            print('==================')
            print(a1, b1)
            print('==================')
            print(a2, b2)
            break
            
    return m, u, v, z

def adjust_u_norm_and_mu_preserve_others(
    m, u, v, z,
    target_norm_u,
    target_mu,
    eps=1e-12
):
    """
    Modify u so that:
        ||u|| -> target_norm_u
        m·u   -> target_mu

    While preserving exactly:
        z·m
        v·u
        z·u
        v·m

    m and z are NOT modified.
    """

    # Ensure column vectors
    m = np.asarray(m).reshape(-1, 1)
    u = np.asarray(u).reshape(-1, 1)
    v = np.asarray(v).reshape(-1, 1)
    z = np.asarray(z).reshape(-1, 1)

    # Save overlaps to preserve
    vu0 = float(v.T @ u)
    zu0 = float(z.T @ u)
    vm0 = float(v.T @ m)
    zm0 = float(z.T @ m)

    # --------------------------------------------------
    # STEP 1: Modify u inside orthogonal complement of span{v,z}
    # --------------------------------------------------

    B = np.hstack([v, z])
    Q, _ = np.linalg.qr(B)  # orthonormal basis for span{v,z}

    u_par = Q @ (Q.T @ u)
    par_sq = float(u_par.T @ u_par)

    # required perpendicular norm
    r2 = target_norm_u**2 - par_sq
    if r2 < -1e-10:
        raise ValueError("Target ||u|| too small while preserving (v·u, z·u).")
    r = np.sqrt(max(0.0, r2))

    # project m into free subspace
    m_S = m - Q @ (Q.T @ m)
    ms = float(np.sqrt(max(0.0, float(m_S.T @ m_S))))

    base = float(m.T @ u_par)
    need = target_mu - base

    if ms < eps:
        if abs(need) > 1e-8:
            raise ValueError("Cannot change m·u (m lies in span{v,z}).")
        u_new = u_par
    else:
        if r < eps:
            if abs(need) > 1e-8:
                raise ValueError("No orthogonal freedom to change m·u.")
            u_new = u_par
        else:
            c = need / (r * ms)
            if c < -1.0 - 1e-10 or c > 1.0 + 1e-10:
                lo = base - r*ms
                hi = base + r*ms
                raise ValueError(
                    f"Target m·u not achievable. Range = [{lo}, {hi}]"
                )
            c = np.clip(c, -1.0, 1.0)
            s = np.sqrt(max(0.0, 1 - c*c))

            e1 = m_S / ms

            # construct second orthonormal direction in S
            rnd = np.random.randn(*m.shape)
            e2 = rnd - Q @ (Q.T @ rnd)
            e2 -= e1 * float(e1.T @ e2)
            n2 = float(np.sqrt(max(0.0, float(e2.T @ e2))))

            if n2 < eps:
                if s > 1e-8:
                    raise ValueError("Free subspace 1D; cannot reach interior values.")
                u_perp = r * np.sign(c) * e1
            else:
                e2 /= n2
                u_perp = r * (c * e1 + s * e2)

            u_new = u_par + u_perp

    # --------------------------------------------------
    # STEP 2: Re-adjust v to preserve vu and vm exactly
    # --------------------------------------------------

    # Solve:
    #   v = a*u_new + b*m + v_perp
    # so that:
    #   v·u_new = vu0
    #   v·m     = vm0

    uu = float(u_new.T @ u_new)
    um = float(u_new.T @ m)
    mm = float(m.T @ m)

    A = np.array([[uu, um],
                  [um, mm]])
    bvec = np.array([vu0, vm0])

    try:
        a, b = np.linalg.solve(A, bvec)
    except np.linalg.LinAlgError:
        a, b = np.linalg.lstsq(A, bvec, rcond=None)[0]

    v_parallel = a * u_new + b * m

    # keep original orthogonal component of v
    Bv = np.hstack([u_new, m])
    Qv, _ = np.linalg.qr(Bv)
    v_perp = v - Qv @ (Qv.T @ v)

    v_new = v_parallel + v_perp

    # --------------------------------------------------
    # Final safety checks
    # --------------------------------------------------

    assert np.allclose(float(z.T @ m), zm0, atol=1e-7)
    assert np.allclose(float(v_new.T @ u_new), vu0, atol=1e-7)
    assert np.allclose(float(z.T @ u_new), zu0, atol=1e-7)
    assert np.allclose(float(v_new.T @ m), vm0, atol=1e-7)

    return m, u_new, v_new, z


def grad_vec(scalar, wrt):
    g, = torch.autograd.grad(
        scalar, wrt,
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    if g is None:
        return torch.zeros_like(wrt)
    return g

def compute_D(params, N, num_epochs):

    D = torch.zeros((num_epochs, 4, 4, N), dtype=torch.float32)
    
    for ep in range(num_epochs):
        m,z,u,v = params[ep]
        
        m = m.detach().view(N, 1).requires_grad_(True)
        u = u.detach().view(N, 1).requires_grad_(True)
        v = v.detach().view(N, 1).requires_grad_(True)
        z = z.detach().view(N, 1).requires_grad_(True)
    
        sig_zm = (z.T @ m).squeeze()
        sig_vu = (v.T @ u).squeeze()
        sig_zu = (z.T @ u).squeeze()
        sig_vm = (v.T @ m).squeeze()
    
        D[ep, 0, 0, :] = grad_vec(sig_zm, z).squeeze()   # m
        D[ep, 0, 1, :] = grad_vec(sig_zm, u).squeeze()   # 0
        D[ep, 0, 2, :] = grad_vec(sig_zm, v).squeeze()   # 0
    
        D[ep, 1, 0, :] = grad_vec(sig_zu, z).squeeze()   # u
        D[ep, 1, 1, :] = grad_vec(sig_zu, u).squeeze()   # z
        D[ep, 1, 2, :] = grad_vec(sig_zu, v).squeeze()   # 0
        
        D[ep, 2, 0, :] = grad_vec(sig_vm, z).squeeze()   # 0
        D[ep, 2, 1, :] = grad_vec(sig_vm, u).squeeze()   # 0
        D[ep, 2, 2, :] = grad_vec(sig_vm, v).squeeze()   # m
        
        D[ep, 3, 0, :] = grad_vec(sig_vu, z).squeeze()   # 0
        D[ep, 3, 1, :] = grad_vec(sig_vu, u).squeeze()   # v
        D[ep, 3, 2, :] = grad_vec(sig_vu, v).squeeze()   # u
    
        D[ep, 0, 3, :] = grad_vec(sig_zm, m).squeeze()   # z
        D[ep, 1, 3, :] = grad_vec(sig_zu, m).squeeze()   # 0 
        D[ep, 2, 3, :] = grad_vec(sig_vm, m).squeeze()   # v
        D[ep, 3, 3, :] = grad_vec(sig_vu, m).squeeze()   # 0
            
    return D.detach().cpu().numpy()

def compute_K(m, u, v, z):
    # A = [z v], B = [m u]
    A = torch.cat([z, v], dim=1)  # (N, 2)
    B = torch.cat([m, u], dim=1)  # (N, 2)
    K = A @ A.T - B @ B.T         # (N, N)
    return K


########################################################
########################################################
################### Rank-1 Linear ######################
########################################################
########################################################
class effective_rnn_from_scaler(nn.Module):
    def __init__(self, zm, zu, vm, vu, dt):
        super(effective_rnn_from_scaler, self).__init__()   
        
        self.dt = dt
        self._W_eff = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=False)
        self._z_eff = torch.tensor([[0.0], [0.0]], requires_grad=False)
        
        self.zm = nn.Parameter(torch.tensor(zm, dtype=torch.float32))
        self.zu = nn.Parameter(torch.tensor(zu, dtype=torch.float32))
        self.vm = nn.Parameter(torch.tensor(vm, dtype=torch.float32))
        self.vu = nn.Parameter(torch.tensor(vu, dtype=torch.float32))
        
    def forward(self, x_t, k):
        m_eff = torch.tensor([[1.], [0.]])
        W_eff = self._W_eff.clone()        
        z_eff = self._z_eff.clone()

        W_eff[1,0] = self.vm
        W_eff[1,1] = self.vu
        z_eff[0,0] = self.zm
        z_eff[1,0] = self.zu
        
        k_update = W_eff@k + m_eff@x_t  
        k = k + self.dt * (-k + k_update)
        y = z_eff.T@k
        return y, k

def vectors_to_overlaps_rank_1(N, params, num_epochs):
    all_overlaps_rnn = []
    for epoch in range(num_epochs):
        m,u,v,z = params[epoch]
        m,u,v,z = m.detach().numpy().flatten(), u.detach().numpy().flatten(), v.detach().numpy().flatten(), z.detach().numpy().flatten()
    
        sig_zm = z@m/N
        sig_zu = z@u/N
        sig_vm = v@m/N
        sig_vu = v@u/N
        
        sig_mu = m@u/N
        sig_zv = z@v/N
        
        mm  = m@m/N
        uu  = u@u/N
        vv  = v@v/N
        zz  = z@z/N
        
        all_overlaps_rnn.append([sig_zm, sig_zu, sig_vm, sig_vu, sig_mu, sig_zv, mm, uu, vv, zz])
        
    return all_overlaps_rnn

def train_full_model( target_exp, N, num_epochs, time_window, noise=0, 
                     m0=None, u0=None, v0=None, z0=None, dt=0.01, lr=0.01, with_plots=False, return_all=True, non_lin=False):    

    m = nn.Parameter(m0.clone().detach(), requires_grad=True)
    
    u = nn.Parameter(u0.clone().detach(), requires_grad=True)
    v = nn.Parameter(v0.clone().detach(), requires_grad=True)
    z = nn.Parameter(z0.clone().detach(), requires_grad=True)
        
    trainable_params = []
    for p in [m, u, v, z]:
        if p.requires_grad: trainable_params.append(p)
        
    optimizer = optim.SGD(trainable_params, lr)

    losses = [] 
    all_params = [] 
    all_grads = [] 
    
    def rollout_window(m_vec, z_vec, W_mat):
        
        x = torch.zeros((time_window, 1))
        x[0, 0] = 1.0 / dt
        h = torch.zeros((N, 1))
        
        f_vals = []
        for step in range(time_window):
            if non_lin:
                alpha = np.sqrt(np.pi) / 2.0
                dh = dt * (-h + (W_mat @ torch.erf(alpha * h)) + m_vec*x[step] )
            else:
                dh = dt * ( -h + (W_mat @ h) + m_vec*x[step] ) 
            h = h + dh
            window_t = (1/N) * torch.sum(z_vec.T @ h )
            f_vals.append(window_t)
        return torch.stack(f_vals)

    for t_e in target_exp:
        
        t_axis = torch.linspace(0.0, float(time_window), time_window)
        y = torch.exp(-t_e * t_axis * dt).detach()
        
        for step in range(num_epochs):
            if return_all:
                all_params.append(deepcopy([m, u, v, z]))
                            
            optimizer.zero_grad()
            W = (1/N)*u@v.T
            f = rollout_window(m, z, W)
            loss = torch.sum((f - ( y + noise*torch.randn( len(t_axis) ) ) ) ** 2) * dt
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        if not return_all:
            all_params.append(deepcopy([m, u, v, z]))
            
            
    if with_plots:
        with torch.no_grad():
            fig, ax = plt.subplots(1,3,figsize=(15, 4))
    
            W = (1/N) * (u@v.T)
            f_final = rollout_window(m, z, W)
            ax[0].plot(y.cpu().numpy(), label="target")
            ax[0].plot(f_final.cpu().numpy(), "--", label="RNN")
            ax[0].legend()
            ax[0].set_title("RNN vs target")
            
            ax[1].plot(losses)
            ax[1].set_title(r'$\text{Loss Curve}$')
            ax[1].set_xlabel(r'$\text{Epoch}$')
            ax[1].set_ylabel(r'$\text{Loss}$')
            ax[1].set_xscale('log')
            
            eig,_ = np.linalg.eig(W.detach().numpy())
            ax[2].scatter(np.real(eig),np.imag(eig),color='b')
            ax[2].set_xlim(-1,1)
            ax[2].set_ylim(-1,1)
            ax[2].axhline(y=0,ls='--',color='k')
            ax[2].axvline(x=0,ls='--',color='k')
            theta = np.linspace(0, 2*np.pi, 400)
            g=0
            ax[2].plot(g * np.cos(theta), g * np.sin(theta), 'r--', label=f'|z| = g = {g}')
            ax[2].axis('equal')
            ax[2].scatter(1-t_e, 0, marker='x', color='r')
    
            plt.tight_layout()
            sns.despine()
        plt.show()
        
    return losses, all_params, all_grads

def run_10d_ode(params, N, dt, target_exp, num_epochs, time_window, lr, init_scalers=None, run_4d=False):

    if params is not None:
        m,u,v,z = params
        m,u,v,z = m.detach().numpy().flatten(), u.detach().numpy().flatten(), v.detach().numpy().flatten(), z.detach().numpy().flatten()
        
        sig_zm = z@m/N
        sig_zu = z@u/N
        sig_vm = v@m/N
        sig_vu = v@u/N
        
        sig_mu = m@u/N
        sig_zv = z@v/N
        
        mm  = m@m/N
        uu  = u@u/N
        vv  = v@v/N
        zz  = z@z/N
        
    else : 
        sig_zm, sig_zu, sig_vm, sig_vu, sig_mu, sig_zv, mm, uu, vv, zz = init_scalers
        
    all_loss_eff = [] 
    all_params_eff = [] 
    all_grads_eff = []
    
    for epoch in range(num_epochs):
        
        all_params_eff.append([sig_zm, sig_zu, sig_vm, sig_vu, sig_mu, sig_zv, mm, uu, vv, zz])
    
        eff_rnn = effective_rnn_from_scaler(sig_zm, sig_zu, sig_vm, sig_vu, dt)
        optimizer  = torch.optim.SGD(eff_rnn.parameters(), lr=lr) 
        
        h = torch.zeros((2,1))               
        x = torch.zeros((time_window,1))
        x[0,0] = 1.0/dt                      
        
        t_axis = torch.linspace(0.0, float(time_window), time_window)
        y_target = torch.exp(-target_exp * t_axis * dt).detach()
        
        y_pred = []                
        for step in range(time_window):
            x_t = x[step,:].unsqueeze(0)
            y_hat, h = eff_rnn(x_t, h)
            y_pred.append(y_hat)  
        
        _loss = torch.sum((torch.stack(y_pred).squeeze() - y_target) ** 2) * dt
        all_loss_eff.append(_loss.item())
            
        optimizer.zero_grad()
        _loss.backward()
    
        d_zm, d_zu = eff_rnn.zm.grad.item(),eff_rnn.zu.grad.item() 
        d_vm, d_vu = eff_rnn.vm.grad.item(),eff_rnn.vu.grad.item()
        all_grads_eff.append([d_zm,d_zu,d_vm,d_vu])
        
        scale = 1.0    

        if run_4d:
            
            sig_zm -= scale * lr * d_zm
            sig_vm -= scale * lr * d_vm
            sig_zu -= scale * lr * d_zu
            sig_vu -= scale * lr * d_vu
            
        else:
            sig_zm -= scale * lr * ( (mm + zz) * d_zm + sig_mu * d_zu + sig_zv * d_vm ) 
            sig_vm -= scale * lr * ( (mm + vv) * d_vm + sig_mu * d_vu + sig_zv * d_zm )
            sig_zu -= scale * lr * ( (uu + zz) * d_zu + sig_mu * d_zm + sig_zv * d_vu )
            sig_vu -= scale * lr * ( (uu + vv) * d_vu + sig_mu * d_vm + sig_zv * d_zu)
        
            sig_mu -= scale * lr * (sig_zm * d_zu + sig_vm * d_vu + sig_zu * d_zm + sig_vu * d_vm)
            sig_zv -= scale * lr * (sig_vm * d_zm + sig_vu * d_zu + sig_zm * d_vm + sig_zu * d_vu)
            
            mm -= scale * lr * 2 * (sig_zm * d_zm + sig_vm * d_vm)        
            uu -= scale * lr * 2 * (sig_zu * d_zu + sig_vu * d_vu)
            vv -= scale * lr * 2 * (sig_vm * d_vm + sig_vu * d_vu)
            zz -= scale * lr * 2 * (sig_zm * d_zm + sig_zu * d_zu)
        #############################################

    return all_loss_eff, all_params_eff, all_grads_eff

def train_full_model_adam( target_exp, N, num_epochs, time_window, noise=0, m0=None, u0=None, v0=None, z0=None, dt=0.01, lr=0.01):    

    m = nn.Parameter(m0.clone().detach(), requires_grad=True)
    u = nn.Parameter(u0.clone().detach(), requires_grad=True)
    v = nn.Parameter(v0.clone().detach(), requires_grad=True)
    z = nn.Parameter(z0.clone().detach(), requires_grad=True)
        
    trainable_params = []
    for p in [m, u, v, z]:
        if p.requires_grad: trainable_params.append(p)
        
    optimizer = optim.Adam(trainable_params, lr)

    losses = [] 
    all_params = [] 
    all_grads = [] 
    
    def rollout_window(m_vec, z_vec, W_mat):
        
        x = torch.zeros((time_window, 1))
        x[0, 0] = 1.0 / dt
        h = torch.zeros((N, 1))
        
        f_vals = []
        for step in range(time_window):
       
            dh = dt * ( -h + (W_mat @ h) + m_vec*x[step] ) 
            h = h + dh
            window_t = (1/N) * torch.sum(z_vec.T @ h )
            f_vals.append(window_t)
        return torch.stack(f_vals)

    for t_e in target_exp:
        
        t_axis = torch.linspace(0.0, float(time_window), time_window)
        y = torch.exp(-t_e * t_axis * dt).detach()
        
        for step in range(num_epochs):
            
            all_params.append(deepcopy([m, u, v, z]))
                            
            optimizer.zero_grad()
            W = (1/N)*u@v.T
            f = rollout_window(m, z, W)
            loss = torch.sum((f - ( y + noise*torch.randn( len(t_axis) ) ) ) ** 2) * dt
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    
    return losses, all_params, all_grads

def exponential_filter(x_seq, c_star, dt, a_star=1):
    alpha = np.exp(-c_star * dt)
    y = torch.zeros_like(x_seq)
    y[0,:] = dt * x_seq[0,:] 
    for t in range(1, len(x_seq)):
        y[t,:] = alpha * y[t-1,:] + dt * x_seq[t,:]
    return y*a_star

########################################################
########################################################
################# Rank-1 Nonlinear #####################
########################################################
########################################################
class effective_rnn_from_scaler_erf(nn.Module):
    """
    2D effective rank-1 RNN with erf non-linearity
    """
    def __init__(self, zm, zu, vm, vu, mu, mm, uu, dt, eps=1e-8):
        super().__init__()
        self.dt = dt
        self.eps = eps

        # overlaps (scalars, trainable)
        self.zm = nn.Parameter(torch.tensor(zm, dtype=torch.float32))
        self.zu = nn.Parameter(torch.tensor(zu, dtype=torch.float32))
        self.vm = nn.Parameter(torch.tensor(vm, dtype=torch.float32))
        self.vu = nn.Parameter(torch.tensor(vu, dtype=torch.float32))
        self.mu = nn.Parameter(torch.tensor(mu, dtype=torch.float32))
        self.mm = nn.Parameter(torch.tensor(mm, dtype=torch.float32))
        self.uu = nn.Parameter(torch.tensor(uu, dtype=torch.float32))

        # basis input vector m = [1,0]^T and a placeholder z buffer
        self.register_buffer("m", torch.tensor([[1.0], [0.0]]))   # (2,1)
        self.register_buffer("zbuf", torch.zeros((2, 1)))         # (2,1)

    def erf_gain(self, Delta):
        """
        phi(x) = erf( (sqrt(pi)/2) x )
        G(Delta) = <phi'(sqrt(Delta) z)>
        Delta: (B,) or scalar
        """
        return 1.0 / torch.sqrt(1.0 + (torch.pi / 2.0) * Delta + self.eps)

    def forward(self, x_t, k):
        """
        x_t: (1,B) or (B,1) or scalar
        k:   (2,B)
        returns:
          y: (1,B)
          k: (2,B)
        """
        # ---- normalize x_t to (1,B)
        if x_t.ndim == 0:
            x_t = x_t.view(1, 1)
        elif x_t.ndim == 1:
            # could be (B,) -> (1,B)
            x_t = x_t.view(1, -1)
        elif x_t.ndim == 2:
            # (B,1) -> (1,B), (1,B) keep
            if x_t.shape[0] != 1 and x_t.shape[1] == 1:
                x_t = x_t.T
            # else assume already (1,B)
        else:
            raise ValueError(f"x_t must be scalar/1D/2D, got shape {tuple(x_t.shape)}")

        # ---- k must be (2,B)
        if k.ndim != 2 or k.shape[0] != 2:
            raise ValueError(f"k must have shape (2,B), got {tuple(k.shape)}")

        k_m = k[0, :]   # (B,)
        k_u = k[1, :]   # (B,)

        Delta = (self.mm * k_m**2 + self.uu * k_u**2 + 2.0 * self.mu * k_m * k_u)  # (B,)

        G = self.erf_gain(Delta)                           # (B,)
        s = (self.vm * k_m + self.vu * k_u) * G            # (B,)

        # k_update = [0; s] + m*x_t  -> (2,B)
        k_update = torch.stack([torch.zeros_like(s), s], dim=0) + self.m * x_t
        k = k + self.dt * (-k + k_update)

        z = self.zbuf.clone()
        z[0, 0] = self.zm
        z[1, 0] = self.zu
        y = (z.T @ k) * G  
        
        return y, k

def flip_flop(t_max, dt, batch_size, 
              stim_range=(4.0, 10.0), 
              input_amp=1.0,          
              target_amp=0.5, 
              stimulus_duration=0.5,
              use_torch=False):
    
    delay_duration = 2.0
    stim_min, stim_max = stim_range
    
    stim_d = int(stimulus_duration / dt)
    ded  = int(delay_duration / dt)
    T      = int(t_max / dt)

    if use_torch:
        x = torch.zeros((batch_size, T, 1), dtype=torch.float32)
        y = torch.zeros((batch_size, T, 1), dtype=torch.float32)
        m = torch.zeros((batch_size, T, 1), dtype=torch.float32)
    else:
        x = np.zeros((batch_size, T, 1), dtype=np.float32)
        y = np.zeros((batch_size, T, 1), dtype=np.float32)
        m = np.zeros((batch_size, T, 1), dtype=np.float32)

    for b in range(batch_size):
        curr_target = 0.0
        last_delay_end = 0
        had_pulse = False

        while True:
            if not had_pulse:
                wait = int(0.5 / dt)
            else:
                if use_torch:
                    wait_val = (torch.rand(1) * (stim_max - stim_min) + stim_min).item()
                else:
                    wait_val = np.random.uniform(stim_min, stim_max)
                wait = int(wait_val / dt)

            pulse_start = last_delay_end + wait
            if pulse_start >= T: break

            if had_pulse and last_delay_end < pulse_start:
                m[b, last_delay_end:pulse_start, 0] = 1.0

            pulse_end = min(pulse_start + stim_d, T)
            delay_end = min(pulse_end + ded, T)

            if use_torch:
                sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
            else:
                sign = np.random.choice([1.0, -1.0])

            x[b, pulse_start:pulse_end, 0] = float(sign)
            curr_target = 1.0 if sign > 0 else -1.0
            y[b, pulse_start:, 0] = curr_target
            m[b, max(0, pulse_start-1):delay_end, 0] = 0.0

            had_pulse = True
            last_delay_end = delay_end

        if had_pulse and last_delay_end < T:
            m[b, last_delay_end:T, 0] = 1.0

    x *= input_amp
    y *= target_amp
    return x, y, m


def decision_making(t_max, dt, batch_size, 
                    coherences=[-16, -8, -4, -2, 2, 4, 8, 16], 
                    noise_std=0.03,                                     
                    scale=3.2, # 3.2
                    target_amp=1.0, 
                    continuous_target=True,
                    fixed_coh_max=16.0, # Add this to lock the normalization
                    use_torch=False):
    
    fix = int(1.0 / dt)
    stim = int(9.0 / dt)
    delay = int(1.0 / dt)
    response = int(9.0 / dt)
    T = int(t_max / dt)

    stim_end = fix + stim
    resp_beg = stim_end + delay
    resp_end = min(resp_beg + response, T)

    if use_torch:
        x = noise_std * torch.randn((batch_size, T, 1), dtype=torch.float32) 
        y = torch.zeros((batch_size, T, 1), dtype=torch.float32)
        m = torch.zeros((batch_size, T, 1), dtype=torch.float32)
        
        coh_idx = torch.randint(len(coherences), (batch_size,))
        coh = torch.tensor(coherences, dtype=torch.float32)[coh_idx]
        
        if continuous_target:
            target = (coh / fixed_coh_max) * target_amp
        else:
            target = torch.where(coh > 0, torch.tensor(1.0), torch.tensor(-1.0)) * target_amp
            
    else:
        x = (noise_std * np.random.normal(size=(batch_size, T, 1))).astype(np.float32) 
        y = np.zeros((batch_size, T, 1), dtype=np.float32)
        m = np.zeros((batch_size, T, 1), dtype=np.float32)
        coh = np.random.choice(coherences, size=batch_size).astype(np.float32)
        
        if continuous_target:
            target = (coh / fixed_coh_max) * target_amp
        else:
            target = np.where(coh > 0, 1.0, -1.0).astype(np.float32) * target_amp

    x[:, fix:stim_end, 0] += coh[:, None] * scale / 100.0
    y[:, resp_beg:resp_end, 0] = target[:, None]
    m[:, resp_beg:resp_end, 0] = 1.0

    return x, y, m, coh


def masked_mse(pred, target, mask, factor=1, eps=1e-8):
    # pred/target/mask: (B,T,1)    
    num = ((pred - target) ** 2) * mask * factor
    den = mask.sum().clamp_min(eps)
    loss = num.sum() / den
    return loss


########################################################
########################################################
################### Rank-2 Linear ######################
########################################################
########################################################
def vectors_to_overlaps_rank_2(N, params, num_epochs):
    
    all_overlaps_rnn = []
    for epoch in range(num_epochs):
        m,u,v,z = params[epoch]
        m,z = m.detach().numpy().flatten(), z.detach().numpy().flatten()
        u1, u2 = u[:,0].detach().numpy().flatten(), u[:,1].detach().numpy().flatten()
        v1, v2 = v[:,0].detach().numpy().flatten(), v[:,1].detach().numpy().flatten()
    
        sig_zm = z@m/N
        sig_zu1 = z@u1/N
        sig_zu2 = z@u2/N
    
        sig_v1m = v1@m/N
        sig_v1u1 = v1@u1/N
        sig_v1u2 = v1@u2/N
    
        sig_v2m  = v2@m/N
        sig_v2u1 = v2@u1/N
        sig_v2u2 = v2@u2/N
        
        sig_mu1 = m@u1/N
        sig_mu2 = m@u2/N
        
        sig_zv1 = z@v1/N
        sig_zv2 = z@v2/N
    
        sig_u1u2 = u1@u2/N
        sig_v1v2 = v1@v2/N
    
        mm   = m@m/N
        uu1  = u1@u1/N
        uu2  = u2@u2/N
        vv1  = v1@v1/N
        vv2  = v2@v2/N
        zz   = z@z/N
        
        all_overlaps_rnn.append([
                         sig_zm, sig_zu1, sig_zu2,
                         sig_v1m, sig_v1u1, sig_v1u2,
                         sig_v2m, sig_v2u1, sig_v2u2,
                         sig_mu1, sig_mu2, sig_zv1, sig_zv2, 
                         sig_u1u2,sig_v1v2,
                         mm, uu1, uu2, vv1, vv2, zz
                        ])
        
    return all_overlaps_rnn


class effective_rnn_rank2_from_scalars(nn.Module):

    def __init__(self, zm, zu1, zu2, v1m, v1u1, v1u2, v2m, v2u1, v2u2, dt):
        super().__init__()
        self.dt = dt

        self.W = torch.tensor([[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]], requires_grad=False)
        self._z = torch.tensor([[0.0],
                                [0.0],
                                [0.0]], requires_grad=False)

        self.zm  = nn.Parameter(torch.tensor(zm,  dtype=torch.float32))
        self.zu1 = nn.Parameter(torch.tensor(zu1, dtype=torch.float32))
        self.zu2 = nn.Parameter(torch.tensor(zu2, dtype=torch.float32))

        self.v1m  = nn.Parameter(torch.tensor(v1m,  dtype=torch.float32))
        self.v1u1 = nn.Parameter(torch.tensor(v1u1, dtype=torch.float32))
        self.v1u2 = nn.Parameter(torch.tensor(v1u2, dtype=torch.float32))

        self.v2m  = nn.Parameter(torch.tensor(v2m,  dtype=torch.float32))
        self.v2u1 = nn.Parameter(torch.tensor(v2u1, dtype=torch.float32))
        self.v2u2 = nn.Parameter(torch.tensor(v2u2, dtype=torch.float32))

    def forward(self, x_t, k):
        W = self.W.clone()
        m_eff = torch.tensor([[1.0], [0.0], [0.0]])
        z = self._z.clone()

        W[1, 0] = self.v1m
        W[1, 1] = self.v1u1
        W[1, 2] = self.v1u2
        W[2, 0] = self.v2m
        W[2, 1] = self.v2u1
        W[2, 2] = self.v2u2

        z[0, 0] = self.zm
        z[1, 0] = self.zu1
        z[2, 0] = self.zu2

        k_update = W @ k + m_eff * x_t
        k = k + self.dt * (-k + k_update)
        y = z.T @ k
        return y, k


def train_model_rank_2( c_star, omega_star, N, num_epochs, time_window, g=0, m0=None, u0=None, v0=None, z0=None, dt=0.05, lr=0.01, plot_target=False):    

    m = nn.Parameter(m0.clone().detach(), requires_grad=True)
    u = nn.Parameter(u0.clone().detach(), requires_grad=True)
    v = nn.Parameter(v0.clone().detach(), requires_grad=True)
    z = nn.Parameter(z0.clone().detach(), requires_grad=True)
        
    trainable_params = []
    for p in [m, u, v, z]:
        if p.requires_grad: trainable_params.append(p)
        
    optimizer = optim.SGD(trainable_params, lr)
    
    losses = [] 
    all_params = [] 
    
    def rollout_window(m_vec, z_vec, W_mat):

        x = torch.zeros((time_window, 1))
        x[0, 0] = 1.0 / dt
        h = torch.zeros((N, 1))
        
        f_vals = []
        for step in range(time_window):
            dh = dt * (-h + (W_mat @ h) + m_vec*x[step])
            h = h + dh
            window_t = (1/N) * torch.sum(z_vec.T @ h )
            f_vals.append(window_t)
        return torch.stack(f_vals)

    t_axis = torch.linspace(0.0, float(time_window), time_window) * dt
    y = np.exp(-c_star * t_axis) * np.cos(omega_star * t_axis)

        
    for step in range(num_epochs):
        all_params.append(deepcopy([m, u, v, z]))
        optimizer.zero_grad()
        W = (1/N)*u@v.T 
        f = rollout_window(m, z, W)
        loss = torch.sum((f - y) ** 2) * dt
        
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    with torch.no_grad():
        fig, ax = plt.subplots(1,3,figsize=(15, 4))
        
        W = (1/N)*u@v.T
        f_final = rollout_window(m, z, W)
        ax[0].plot(y.cpu().numpy(), label="target")
        ax[0].plot(f_final.cpu().numpy(), "--", label="RNN")
        ax[0].legend()
        ax[0].set_title("RNN vs target")
        
        ax[1].plot(losses)
        ax[1].set_title(r'$\text{Loss Curve}$')
        ax[1].set_xlabel(r'$\text{Epoch}$')
        ax[1].set_ylabel(r'$\text{Loss}$')
        ax[1].set_xscale('log')
        
        eig,_ = np.linalg.eig(W.detach().numpy())
        ax[2].scatter(np.real(eig),np.imag(eig),color='b')
        ax[2].set_xlim(-1,1)
        ax[2].set_ylim(-1,1)
        ax[2].axhline(y=0,ls='--',color='k')
        ax[2].axvline(x=0,ls='--',color='k')
        ax[2].axis('equal')
        ax[2].scatter(1-c_star, omega_star, marker='x', color='r')
        ax[2].scatter(1-c_star, -omega_star, marker='x', color='r')

        plt.tight_layout()
        sns.despine()
        plt.show()
        
    return losses, all_params

def run_21d_ode(params, N, dt, c_star, omega_star, num_epochs, time_window, lr, init_scalers=None):

    if params is not None:
        m,u,v,z = params
        m,z = m.detach().numpy().flatten(), z.detach().numpy().flatten()
        u1, u2 = u[:,0].detach().numpy().flatten(), u[:,1].detach().numpy().flatten()
        v1, v2 = v[:,0].detach().numpy().flatten(), v[:,1].detach().numpy().flatten()
        
        sig_zm = z@m/N
        sig_zu1 = z@u1/N
        sig_zu2 = z@u2/N
    
        sig_v1m = v1@m/N
        sig_v1u1 = v1@u1/N
        sig_v1u2 = v1@u2/N
    
        sig_v2m  = v2@m/N
        sig_v2u1 = v2@u1/N
        sig_v2u2 = v2@u2/N
        
        sig_mu1 = m@u1/N
        sig_mu2 = m@u2/N
        
        sig_zv1 = z@v1/N
        sig_zv2 = z@v2/N
    
        sig_u1u2 = u1@u2/N
        sig_v1v2 = v1@v2/N
    
        mm   = m@m/N
        uu1  = u1@u1/N
        uu2  = u2@u2/N
        vv1  = v1@v1/N
        vv2  = v2@v2/N
        zz   = z@z/N
        
    else : 
        (sig_zm, sig_zu1, sig_zu2, sig_v1m, sig_v1u1, sig_v1u2,
         sig_v2m, sig_v2u1, sig_v2u2, sig_mu1, sig_mu2,
         sig_zv1, sig_zv2, sig_u1u2, sig_v1v2,
         mm, uu1, uu2, vv1, vv2, zz) = init_scalers
        
    all_loss_eff = [] 
    all_params_eff = [] 
    
    for epoch in range(num_epochs):
    
    
        all_params_eff.append([ sig_zm, sig_zu1, sig_zu2,
                     sig_v1m, sig_v1u1, sig_v1u2,
                     sig_v2m, sig_v2u1, sig_v2u2,
                     sig_mu1, sig_mu2, sig_zv1,
                    sig_zv2, sig_u1u2, sig_v1v2,
                    mm, uu1, uu2, vv1, vv2, zz])
    
        eff_rnn = effective_rnn_rank2_from_scalars(sig_zm, sig_zu1, sig_zu2, sig_v1m, sig_v1u1, sig_v1u2, sig_v2m, sig_v2u1, sig_v2u2, dt)
        optimizer = torch.optim.SGD(eff_rnn.parameters(), lr=lr)
        
        h = torch.zeros((3,1))               
        x = torch.zeros((time_window,1))
        x[0,0] = 1.0/dt                     
        
        t_axis = torch.linspace(0.0, float(time_window), time_window) * dt
        y_target = np.exp(-c_star * t_axis) * np.cos(omega_star * t_axis)
        
        y_pred = []  
        for step in range(time_window):
            x_t = x[step, :].unsqueeze(0)
            y_hat, h = eff_rnn(x_t, h)
            y_pred.append(y_hat)
    
        _loss = torch.sum((torch.stack(y_pred).squeeze() - y_target) ** 2) * dt
        all_loss_eff.append(_loss.item())
            
        optimizer.zero_grad()
        _loss.backward()
        
        # Gradients
        d_zm, d_zu1, d_zu2 = eff_rnn.zm.grad.item(), eff_rnn.zu1.grad.item(), eff_rnn.zu2.grad.item()
        d_v1m, d_v1u1, d_v1u2 = eff_rnn.v1m.grad.item(), eff_rnn.v1u1.grad.item(), eff_rnn.v1u2.grad.item()
        d_v2m, d_v2u1, d_v2u2 = eff_rnn.v2m.grad.item(), eff_rnn.v2u1.grad.item(), eff_rnn.v2u2.grad.item()

        scale = 1.0
        ################################  
        sig_zm -= scale * lr * ( (mm + zz) * d_zm + (sig_mu1 * d_zu1 + sig_mu2 * d_zu2) + (sig_zv1 * d_v1m + sig_zv2 * d_v2m) )
        sig_v1m -= scale * lr * ( sig_zv1 * d_zm + (vv1 * d_v1m + sig_v1v2 * d_v2m) + mm * d_v1m + (sig_mu1 * d_v1u1 + sig_mu2 * d_v1u2) )
        sig_v2m -= scale * lr * ( sig_zv2 * d_zm + (sig_v1v2 * d_v1m + vv2 * d_v2m) + mm * d_v2m + (sig_mu1 * d_v2u1 + sig_mu2 * d_v2u2) )
        
    
        sig_zu1 -= scale * lr * ( sig_mu1 * d_zm + (uu1 + zz) * d_zu1 + sig_zv1 * d_v1u1 + sig_zv2 * d_v2u1 + sig_u1u2 * d_zu2 )
        sig_zu2 -= scale * lr * ( sig_mu2 * d_zm + (uu2 + zz) * d_zu2 + sig_zv1 * d_v1u2 + sig_zv2 * d_v2u2 + sig_u1u2 * d_zu1 )
    
        sig_v1u1 -= scale * lr * ( sig_mu1 * d_v1m + (uu1 + vv1) * d_v1u1 + sig_u1u2 * d_v1u2 + sig_zv1 * d_zu1 + sig_v1v2 * d_v2u1 )
        sig_v1u2 -= scale * lr * ( sig_mu2 * d_v1m + (uu2 + vv1) * d_v1u2 + sig_u1u2 * d_v1u1 + sig_zv1 * d_zu2 + sig_v1v2 * d_v2u2 )
        sig_v2u1 -= scale * lr * ( sig_mu1 * d_v2m + (uu1 + vv2) * d_v2u1 + sig_u1u2 * d_v2u2 + sig_zv2 * d_zu1 + sig_v1v2 * d_v1u1 )
        sig_v2u2 -= scale * lr * ( sig_mu2 * d_v2m + (uu2 + vv2) * d_v2u2 + sig_u1u2 * d_v2u1 + sig_zv2 * d_zu2 + sig_v1v2 * d_v1u2 )
        
    
        sig_mu1 -= scale * lr * ( sig_zu1 * d_zm + sig_zm * d_zu1 + (sig_v1m * d_v1u1 + sig_v2m * d_v2u1) + (sig_v1u1 * d_v1m + sig_v2u1 * d_v2m) )
        sig_mu2 -= scale * lr * ( sig_zu2 * d_zm + sig_zm * d_zu2 + (sig_v1m * d_v1u2 + sig_v2m * d_v2u2) + (sig_v1u2 * d_v1m + sig_v2u2 * d_v2m) )
        
        sig_zv1 -= scale * lr * ( sig_v1m * d_zm + sig_v1u1 * d_zu1 + sig_v1u2 * d_zu2 + sig_zm * d_v1m + sig_zu1 * d_v1u1 + sig_zu2 * d_v1u2 )
        sig_zv2 -= scale * lr * ( sig_v2m * d_zm + sig_v2u1 * d_zu1 + sig_v2u2 * d_zu2 + sig_zm * d_v2m + sig_zu1 * d_v2u1 + sig_zu2 * d_v2u2 )
        
    
        sig_u1u2 -= scale * lr * ( sig_zu2 * d_zu1 + sig_zu1 * d_zu2 + sig_v1u2 * d_v1u1 + sig_v1u1 * d_v1u2 + sig_v2u2 * d_v2u1 + sig_v2u1 * d_v2u2 )
        sig_v1v2 -= scale * lr * ( sig_v2m * d_v1m + sig_v2u1 * d_v1u1 + sig_v2u2 * d_v1u2 + sig_v1m * d_v2m + sig_v1u1 * d_v2u1 + sig_v1u2 * d_v2u2 )
        
        # Norms
        mm  -= scale * lr * ( 2.0 * ( sig_zm * d_zm + sig_v1m * d_v1m + sig_v2m * d_v2m ) )
        uu1 -= scale * lr * ( 2.0 * ( sig_zu1 * d_zu1 + sig_v1u1 * d_v1u1 + sig_v2u1 * d_v2u1 ) )
        uu2 -= scale * lr * ( 2.0 * ( sig_zu2 * d_zu2 + sig_v1u2 * d_v1u2 + sig_v2u2 * d_v2u2 ) )
        vv1 -= scale * lr * ( 2.0 * ( sig_v1m * d_v1m + sig_v1u1 * d_v1u1 + sig_v1u2 * d_v1u2 ) )
        vv2 -= scale * lr * ( 2.0 * ( sig_v2m * d_v2m + sig_v2u1 * d_v2u1 + sig_v2u2 * d_v2u2 ) )
        zz  -= scale * lr * ( 2.0 * ( sig_zm * d_zm + sig_zu1 * d_zu1 + sig_zu2 * d_zu2 ) )
        #############################################

    return all_loss_eff, all_params_eff
