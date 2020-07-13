import numpy as np
import torch


class ToyLoader:
    def __init__(self, device="cpu", seed=42, n1=50, n2=25, d_noise=8):
        import ot
        self.n1 = n1
        self.n2 = n2
                
        np.random.seed(seed)
        
        mu_s = np.array([-5, 0])
        cov_s = np.array([[5, 4], [1, 1]])
        
        mu_t = np.array([5, 0])
        cov_t = cov_s

        # import ipdb; ipdb.set_trace()
        # cov_t = np.array([[5, 4], [-0.1, 0.1]])

        # xs = np.random.multivariate_normal(mu_s, cov_s, n1)
        # xt = np.random.multivariate_normal(mu_t, cov_t, n2)

        xs = ot.datasets.make_2D_samples_gauss(n1, mu_s, cov_s)
        xt = ot.datasets.make_2D_samples_gauss(n2, mu_t, cov_t)
        
        # import ipdb; ipdb.set_trace()
                
        X = np.hstack((xs, np.random.randn(n1, d_noise)))
        Y = np.hstack((xt, np.random.randn(n2, d_noise)))
        
        self.X = torch.tensor(X, device=device)
        self.Y = torch.tensor(Y, device=device)

        self.d = self.X.shape[1]
        self.groups = [[0, 1]] + [[k] for k in range(2, 2+d_noise)]
        self.groups = [[0, 1]] + [[k for k in range(2, 2+d_noise)]]
        # self.groups = [[k] for k in range(0, 2+d_noise)]

        self.platform = {"dtype": self.X.dtype, "device": device}

        # self.show()
        
    def show(self):
        import matplotlib.pyplot as plt

        # plt.ylim(0, 10)
        # plt.xlim(0, 16)
        dim1 = 0
        dim2 = 2
        plt.scatter(self.X[..., dim1], self.X[..., dim2])
        plt.scatter(self.Y[..., dim1], self.Y[..., dim2])
        plt.show()
        
    def visualize(self, PI, titlename="", loss=None, show=True, savename=None):                    
        import matplotlib.pyplot as plt
        import ot.plot
        plt.rcParams.update({'font.size': 16})
        
        # plt.ylim(0, 10)
        # plt.xlim(0, 16)
        ot.plot.plot2D_samples_mat(self.X[..., :2], self.Y[..., :2], PI, color=[.5, .5, 1])
        plt.plot(self.X[..., :2][:, 0], self.X[..., :2][:, 1], '+b', label='Source samples')
        plt.plot(self.Y[..., :2][:, 0], self.Y[..., :2][:, 1], 'xr', label='Target samples')
        plt.legend(loc=0)
        # plt.title(titlename)
        
        if savename is not None:
            plt.savefig(savename)
            
        if loss:
            plt.figure(3)
            plt.plot(loss)
            
        if show:
            plt.show()
        plt.close()
