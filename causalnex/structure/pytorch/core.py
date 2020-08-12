# Copyright 2019-2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code is modified from this git repo: https://github.com/xunzheng/notears

@inproceedings{zheng2020learning,
    author = {Zheng, Xun and Dan, Chen and Aragam, Bryon and Ravikumar, Pradeep and Xing, Eric P.},
    booktitle = {International Conference on Artificial Intelligence and Statistics},
    title = {{Learning sparse nonparametric DAGs}},
    year = {2020}
}
"""
import logging
from typing import List, Tuple

import numpy as np
import scipy.optimize as sopt
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator


class NotearsMLP(nn.Module, BaseEstimator):
    """
    Class for NOTEARS MLP (Multi-layer Perceptron) model. The model weights consist of fc1 and fc2 weights respectively.
    fc1 weight is the weight of the first fully connected layer which determines the causal structure.
    fc2 weights are the weight of hidden layers after the first fully connected layer
    """

    def __init__(
        self,
        n_features: int,
        use_bias: bool = False,
        bounds: List[Tuple[int, int]] = None,
        lasso_beta: float = 0.0,
        ridge_beta: float = 0.0,
    ):
        """
        Constructor for NOTEARS MLP class.

        Args:
            n_features: number of input features
            use_bias: True to add the intercept to the model
            and the numbers determine the number of nodes used for the layer in order.
            bounds: bound constraint for each parameter.
            lasso_beta: Constant that multiplies the lasso term (l1 regularisation).
            It only applies to adjacency weights (fc1 weights)
            ridge_beta: Constant that multiplies the ridge term (l2 regularisation).
            It applies to both adjacency (fc1) and fc2 weights.
        """
        super().__init__()
        self.device = torch.device("cpu")
        self.lasso_beta = lasso_beta
        self.ridge_beta = ridge_beta
        self.dims = [n_features, 1]

        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(
            self.dims[0], self.dims[0] * self.dims[1], bias=True
        ).float()
        nn.init.zeros_(self.fc1_pos.weight)
        nn.init.zeros_(self.fc1_pos.bias)

        self.fc1_neg = nn.Linear(
            self.dims[0], self.dims[0] * self.dims[1], bias=True
        ).float()
        nn.init.zeros_(self.fc1_neg.weight)
        nn.init.zeros_(self.fc1_neg.bias)

        self.use_bias = use_bias
        self.bounds = bounds

    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    @property
    def fc1_bias(self) -> torch.Tensor:
        """
        fc1 bias is the bias of the first fully connected layer which determines the causal structure.
        Returns:
            fc1 bias
        """
        return self.fc1_pos.bias - self.fc1_neg.bias

    @property
    def fc1_weight(self) -> torch.Tensor:
        """
        fc1 weight is the weight of the first fully connected layer which determines the causal structure.
        Returns:
            fc1 weight
        """
        return self.fc1_pos.weight - self.fc1_neg.weight

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        """
        Feed forward calculation for the model.

        Args:
            x: input torch tensor

        Returns:
            output tensor from the model
        """
        x = self.fc1_pos(x) - self.fc1_neg(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    @property
    def bias(self) -> np.ndarray:
        """
        Get the vector of feature biases

        Returns:
            bias vector
        """
        return self.fc1_bias.cpu().detach().numpy()

    def get_adj(self, w_threshold: float = None) -> np.ndarray:
        """
        Get the adjacency matrix from NOTEARS MLP

        Args:
            w_threshold: fixed threshold for absolute edge weights.
        Returns:
            adjacency matrix
        """
        adj = self.fc1_weight  # [j * m1, i]

        adj = adj.cpu().detach().numpy().T
        if w_threshold is not None:
            adj[np.abs(adj) < w_threshold] = 0
        return adj

    def fit(
        self,
        x: np.ndarray,
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
    ):
        """
        Fit NOTEARS MLP model using the input data x
        Args:
            x: 2d numpy array input data, axis=0 is data rows, axis=1 is data columns. Data must be row oriented.
            max_iter: max number of dual ascent steps during optimisation.
            h_tol: exit if h(w) < h_tol (as opposed to strict definition of 0).
            rho_max: to be updated
        """
        rho, alpha, h = 1.0, 0.0, np.inf
        X_torch = torch.from_numpy(x).float().to(self.device)

        for n_iter in range(max_iter):
            rho, alpha, h = self._dual_ascent_step(X_torch, rho, alpha, h, rho_max)
            if h <= h_tol or rho >= rho_max:
                break
            if n_iter == max_iter - 1 and h > h_tol:
                self._logger.warning(
                    "Failed to converge. Consider increasing max_iter."
                )

    # pylint: disable=too-many-locals
    def _dual_ascent_step(
        self, X: torch.Tensor, rho: float, alpha: float, h: float, rho_max: float
    ) -> Tuple[float, float, float]:
        """
        Perform one step of dual ascent in augmented Lagrangian.

        Args:
            X: input tensor data.
            rho: max number of dual ascent steps during optimisation.
            alpha: exit if h(w) < h_tol (as opposed to strict definition of 0).
            h: DAGness of the adjacency matrix
            rho_max: to be updated

        Returns:
            rho, alpha and h
        """

        def _get_flat_grad(params: List[torch.Tensor]) -> np.ndarray:
            """
            Get flatten gradient vector from the parameters of the model

            Args:
                params: parameters of the model

            Returns:
                flatten gradient vector in numpy form
            """
            views = [
                p.data.new(p.data.numel()).zero_()
                if p.grad is None
                else p.grad.data.to_dense().view(-1)
                if p.grad.data.is_sparse
                else p.grad.data.view(-1)
                for p in params
            ]
            return torch.cat(views, 0).cpu().detach().numpy()

        def _get_flat_params(params: List[torch.Tensor]) -> np.ndarray:
            """
            Get parameters in flatten vector from the parameters of the model

            Args:
                params: parameters of the model

            Returns:
                flatten parameters vector in numpy form
            """
            views = [
                p.data.to_dense().view(-1) if p.data.is_sparse else p.data.view(-1)
                for p in params
            ]
            return torch.cat(views, 0).cpu().detach().numpy()

        def _update_params_from_flat(
            params: List[torch.Tensor], flat_params: np.ndarray
        ):
            """
            Update parameters of the model from the parameters in the form of flatten vector

            Args:
                params: parameters of the model
                flat_params: parameters in the form of flatten vector
            """
            offset = 0
            flat_params_torch = torch.from_numpy(flat_params).to(
                torch.get_default_dtype()
            )
            for p in params:
                n_params = p.numel()
                # view_as to avoid deprecated pointwise semantics
                p.data = flat_params_torch[offset : offset + n_params].view_as(p.data)
                offset += n_params

        def _func(flat_params: np.ndarray) -> Tuple[float, np.ndarray]:
            """
            Objective function that the NOTEARS algorithm tries to minimise.

            Args:
                flat_params: parameters to be optimised to minimise the objective function

            Returns:
                Loss and gradient
            """
            _update_params_from_flat(params, flat_params)

            optimizer.zero_grad()

            X_hat = self(X)
            h_val = self._h_func()

            loss = (0.5 / X.shape[0]) * torch.sum((X_hat - X) ** 2)
            lagrange_penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            # NOTE: both the l2 and l1 regularization are not applied to the bias parameters
            l2_reg = 0.5 * self.ridge_beta * self._l2_reg()
            l1_reg = self.lasso_beta * torch.sum(params[0] + params[1])

            primal_obj = loss + lagrange_penalty + l2_reg + l1_reg
            primal_obj.backward()
            loss = primal_obj.item()

            flat_grad = _get_flat_grad(params)

            return loss, flat_grad.astype("float64")

        optimizer = torch.optim.Optimizer(self.parameters(), dict())

        # pack the bias parameters into the END of the parameter set
        params = [
            param for name, param in self.named_parameters() if "bias" not in name
        ]
        params.extend(
            [param for name, param in self.named_parameters() if "bias" in name]
        )

        flat_params = _get_flat_params(params)

        # duplicate the bounds for pos and neg fc1
        bounds = self.bounds * 2
        # get the number of additional bounds needed for intercepts
        n_intercept_bounds = len(flat_params) - len(bounds)
        # if use_bias=False, force intercepts to be zero
        rh_bias_bound = None if self.use_bias else 0
        # add the bias bounds to the END for the bound set
        bounds += [(0, rh_bias_bound)] * n_intercept_bounds

        while rho < rho_max:
            # Magic
            sol = sopt.minimize(
                _func, flat_params, method="L-BFGS-B", jac=True, bounds=bounds,
            )

            _update_params_from_flat(params, sol.x)
            with torch.no_grad():
                h_new = self._h_func().item()
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        alpha += rho * h_new
        return rho, alpha, h_new

    def _h_func(self) -> torch.Tensor:
        """
        Constraint function of the NOTEARS algorithm.
        Constrain 2-norm-squared of fc1 weights of the model along m1 dim to be a DAG

        Returns:
            DAGness of the adjacency matrix
        """
        d = self.dims[0]
        d_torch = torch.tensor(d).to(self.device)  # pylint: disable=not-callable
        fc1_weight = self.fc1_weight.view(d, -1, d)  # [j, m1, i]
        square_weight_mat = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        # h = trace_expm(a) - d  # (Zheng et al. 2018)
        characteristic_poly_mat = (
            torch.eye(d).to(self.device) + square_weight_mat / d_torch
        )  # (Yu et al. 2019)
        polynomial_mat = torch.matrix_power(characteristic_poly_mat, d - 1)
        h = (polynomial_mat.t() * characteristic_poly_mat).sum() - d
        return h

    def _l2_reg(self) -> torch.Tensor:
        """
        Take 2-norm-squared of all parameters of the model

        Returns:
            l2 regularisation term
        """
        reg = 0.0
        reg += torch.sum(self.fc1_weight ** 2)
        return reg
