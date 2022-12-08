import numpy as np
from dataclasses import dataclass


@dataclass
class Parameter:
    wgt:int        
    phi1:int
    phi2:int
    sig2:int


class Model:

    def __init__(self) -> None:
        pass

    @staticmethod
    def construct_feedback_matrix(S, M, parameter:Parameter) -> None:
        n_units, n_dims = M.shape
        S = np.zeros(M.shape)
        T = np.ones(M.shape) / np.sqrt(2)
        W = np.array(
            [1, 0],
            [0, parameter.wgt]
        )
        DV = np.zeros((n_dims, 1))
        s = 0
        for i in range(n_units):
            for j in range(n_units):
                for k in range(n_dims):
                    DV[k, 0] = M[i, k] - M(j, k)
                DV = T @ DV
                s = (DV.T @ W @ DV)[0, 0]
                S[i, j] = parameter.phi2 * np.exp(-1*parameter.phi1*s*s)
        return -S

    def predict(self, M:np.ndarray, parameter:Parameter) -> None:
        assert self.stopping_time > 0, "Stopping Time has to be non-negative."

        n_units, n_dims = M.shape
        # initialize choice matrix
        choice_probability = np.zeros(n_units)
        
        # initialize S TODO what is S
        S = self.construct_feedback_matrix(M, parameter)
