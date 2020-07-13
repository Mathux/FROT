import torch
import numpy as np
import ot
from .sinkhorn_stab import sinkhorn_stabilized
from scipy.optimize import linprog
from scipy.special import logsumexp        
from sklearn.base import BaseEstimator 

DEFAULT_PLATFORM = {"dtype": torch.float64, "device": "cpu"}


class Frot(BaseEstimator):
    """Feature Robust Optimal Transport

    The optimization objective for Frot is:
        min_{PI} max_{alpha} sum_{i=1}^n sum_{j=1}^m PI_{ij} sum_{l=1}^L alpha_l d(x_i^{S_l}, y_j^{S_l})^p

    It will compute the optimal transport ``PI`` and the inportance of the groups ``alpha``.

    Parameters
    ----------
    pnorm : int, default=2
        Use the p-norm distance for d.
    pFRWD : int, default=2
        Compute the p-FRWD distance.
    eta : float, default=1.0
        Regularization parameter for alpha.
    niter : int, default=10
        Number of iterations in Frank-Wolfe. 
        Only applicable when method is set to "sinkhorn" or "emd"
    eps : float, default=0.05
        Regularization parameter for sinkhorn.
        Only applicable when method is set to "sinkhorn"
    method : {"sinkhorn", "emd", "lp"}
        Choice of the method to compute FROT.
        sinkhorn and emd will use Frank-Wolfe algorithme
        lp will use a external linear program solver

    Attributes
    ----------
    PI_ : ndarray of shape (n_targets, m_targets)
        Optimal transport plan
    alpha_ : ndarray of shape (n_groups,)
        Group importance

    Examples
    --------
    >>> import numpy as np
    >>> from src.models.frot import Frot
    >>> model = Frot(method="emd")
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> Y = np.array([[0, 1], [0, 1], [2, -2]])
    >>> model.fit(X, Y, [[0], [1]])
    Frot(eps=0.05, eta=1.0, method='emd', niter=10, p=2)
    >>> print(model.PI_)
    [[0.04848485 0.         0.28484848]
    [0.28484848 0.04848485 0.        ]
    [0.         0.28484848 0.04848485]]
    >>> print(model.alpha_)
    [0.53696945 0.46303055]
    """
    def __init__(self, pnorm=2, pFRWD=1, eta=1.0, niter=10, eps=0.05, method="sinkhorn"):
        self.method_choices = ["sinkhorn", "emd", "lp"]
        self.check_and_update_method(method)      
        
        self.pnorm = pnorm
        self.pFRWD = pFRWD
        self.eta = eta
        self.niter = niter
        self.eps = eps
        self.method = method

    def check_and_update_method(self, method):
        if method not in self.method_choices:
            raise NotImplementedError("Try one method in this list: {}".
                                      format(self.method_choices))
        self.modelname = "Frot" + method.capitalize()
        if method == "sinkhorn":
            self._inner_fit = self.fit_frank_wolfe
            self._inner_solver_frank = self.__use_sinkhorn
        elif method == "emd":
            self._inner_fit = self.fit_frank_wolfe
            self._inner_solver_frank = self.__use_emd
        elif method == "lp":
            self._inner_fit = self.fit_lp
            
        self.method = method
        
    def __use_sinkhorn(self, M):
        return sinkhorn_stabilized(self.a_, self.b_, M,
                                   self.eps, self.platform_)

    def __use_emd(self, M):
        return ot.emd(self.a_, self.b_, M)

    def fit(self, X, Y, groups, a=None, b=None, platform=DEFAULT_PLATFORM):
        # Check in case user change the method
        self.check_and_update_method(self.method)
        
        # Dimensions
        self.n_, self.m_, self.L_ = X.shape[0], Y.shape[0], len(groups)
        
        self.platform_ = platform
        
        # Optimal tranport constraints
        self.a_ = torch.ones(self.n_, **platform)/self.n_ if a is None else a
        self.b_ = torch.ones(self.m_, **platform)/self.m_ if b is None else b
        
        if not isinstance(X, torch.Tensor) or not isinstance(Y, torch.Tensor):
            X = torch.tensor(X, **platform)
            Y = torch.tensor(Y, **platform)
        
        # Compute distances between each vectors for each groups
        self.C_ = torch.stack([torch.cdist(X[:, grp], Y[:, grp], p=self.pnorm)
                               for grp in groups])**self.pFRWD
        
        # Compute PI and alpha
        self.PI_, self.alpha_ = self._inner_fit()
        
        # Compute the FRWD distance
        dist = np.einsum("ij,l,lij", self.PI_, self.alpha_, self.C_)
        dist = dist**(1/self.pFRWD)
        self.dist_ = self.FRWD_ = dist
        
        # Store usefull informations
        self.groups_ = np.array(groups)
        self.sorted_group_importance = self.groups_[np.argsort(self.alpha_)[::-1]]
        
        return self
    
    def fit_frank_wolfe(self):
        # Initialization of PI => uniform distribution
        PI = torch.ones((self.n_, self.m_), **self.platform_)/(self.n_*self.m_)
        
        for t in range(self.niter):
            phi = torch.einsum("ij,kij->k", PI, self.C_)
            
            # This is the function we want to minimize
            # G_loss = self.eta * torch.logsumexp(phi/self.eta, 0)
            # print(G_loss)

            alpha = torch.exp(phi/self.eta - torch.logsumexp(phi/self.eta, 0))
            # It is the same as alpha = torch.exp(phi/self.eta); alpha /= alpha.sum()
            # But this is more stable and prevent overflow            

            # Solve the optimal transport subproblem
            M = torch.einsum("kij,k->ij", self.C_, alpha)
            PI_t = self._inner_solver_frank(M)

            # Update PI
            gamma = 2 / (2 + t)
            PI = (1 - gamma) * PI + gamma * PI_t
            
        return np.array(PI.cpu()), np.array(alpha.cpu())

    def fit_lp(self):
        n, m, L = self.n_, self.m_, self.L_
        
        # Compute equality constraints
        Q = one_hot(np.repeat(np.arange(n), m), num_classes=n)
        R = one_hot(np.tile(np.arange(m), n), num_classes=m)
        A_eq = np.c_[np.c_[Q, R].T, np.zeros((m+n), dtype=int)][:-1]
        b_eq = np.r_[self.a_, self.b_][:, None][:-1]
        # Last equality contraints has been removed
        # because it is redundant
        
        # Compute inequality constraints
        vecC = np.reshape(self.C_, (L, n*m))
        A_ub = np.c_[vecC, -np.ones((L, 1))]
        b_ub = np.zeros((L, 1))
        
        # Cost
        c = np.zeros(n*m+1)
        c[-1] = 1
        
        # Compute the best solution
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
        PI = res.x[:-1].reshape((n, m))
        
        # Compute alpha
        phi = np.dot(A_ub[:, :-1], res.x[:-1]) # same as np.einsum("ij,kij->k", PI, self.C_)
        alpha = np.exp(phi/self.eta - logsumexp(phi/self.eta))
        # The best eta is a hot vector, this one is instead a softmax version
        # It is the same as alpha = np.exp(phi/self.eta); alpha /= alpha.sum()
        # But this is more stable and prevent overflow
        
        return PI, alpha


def one_hot(data, num_classes=None):
    if num_classes is None:
        num_classes = data.max() + 1
    output = np.zeros((data.size, num_classes), dtype=int)
    output[np.arange(data.size), data] = 1
    return output
