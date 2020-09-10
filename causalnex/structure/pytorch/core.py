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
from typing import Iterable, List, Tuple, Union

import numpy as np
import scipy.optimize as sopt
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator

from .nonlinear import LocallyConnected


class NotearsMLP(nn.Module, BaseEstimator):
    """
    Class for NOTEARS MLP (Multi-layer Perceptron) model.
    The model weights consist of dag_layer and loc_lin_layer weights respectively.
    dag_layer weight is the weight of the first fully connected layer which determines the causal structure.
    loc_lin_layer weights are the weight of hidden layers after the first fully connected layer
    """

    def __init__(
        self,
        n_features: int,
        use_bias: bool = False,
        hidden_layer_units: Iterable[int] = (0,),
        bounds: List[Tuple[int, int]] = None,
        lasso_beta: float = 0.0,
        ridge_beta: float = 0.0,
        nonlinear_clamp: float = 1e-2,
    ):
        """
        Constructor for NOTEARS MLP class.

        Args:
            n_features: number of input features
            use_bias: True to add the intercept to the model
            hidden_layer_units: An iterable where its length determine the number of layers used,
            and the numbers determine the number of nodes used for the layer in order.
            bounds: bound constraint for each parameter.
            lasso_beta: Constant that multiplies the lasso term (l1 regularisation).
            It only applies to dag_layer weight.
            ridge_beta: Constant that multiplies the ridge term (l2 regularisation).
            It applies to both dag_layer and loc_lin_layer weights.
            nonlinear_clamp: Value used to soft clamp the nonlinear layer normalisation.
            Prevents the weights from being scaled above 1/nonlinear_clamp.
        """
        super().__init__()
        self.device = torch.device("cpu")
        self.lasso_beta = lasso_beta
        self.ridge_beta = ridge_beta
        self.nonlinear_clamp = nonlinear_clamp

        # cast to list for later concat.
        self.dims = (
            [n_features] + list(hidden_layer_units) + [1]
            if hidden_layer_units[0]
            else [n_features, 1]
        )

        # dag_layer: initial linear layer
        self.dag_layer = nn.Linear(
            self.dims[0], self.dims[0] * self.dims[1], bias=use_bias
        ).float()
        nn.init.zeros_(self.dag_layer.weight)
        if use_bias:
            nn.init.zeros_(self.dag_layer.bias)

        # loc_lin_layer: local linear layers
        layers = [
            LocallyConnected(
                self.dims[0], input_features, output_features, bias=use_bias
            ).float()
            for input_features, output_features in zip(self.dims[1:-1], self.dims[2:])
        ]
        self._loc_lin_layer_weights = nn.ModuleList(layers)
        for layer in layers:
            layer.reset_parameters()

        # set the bounds as an attribute on the weights object
        self.dag_layer.weight.bounds = bounds
        # type the adjacency matrix
        self.adj = None
        self.adj_mean_effect = None

    @property
    def _logger(self):
        return logging.getLogger(self.__class__.__name__)

    @property
    def dag_layer_bias(self) -> Union[torch.Tensor, None]:
        """
        dag_layer bias is the bias of the first fully connected layer which determines the causal structure.
        Returns:
            dag_layer bias if use_bias is True, otherwise None
        """
        return self.dag_layer.bias

    @property
    def dag_layer_weight(self) -> torch.Tensor:
        """
        dag_layer weight is the weight of the first fully connected layer which determines the causal structure.
        Returns:
            dag_layer weight
        """
        return self.dag_layer.weight

    @property
    def loc_lin_layer_weights(self) -> torch.Tensor:
        """
        loc_lin_layer weights are the weight of hidden layers after the first fully connected layer.
        Returns:
            loc_lin_layer weights
        """
        return self._loc_lin_layer_weights

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        """
        Feed forward calculation for the model.

        Args:
            x: input torch tensor

        Returns:
            output tensor from the model
        """
        x = self.dag_layer(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for layer in self.loc_lin_layer_weights:
            x = torch.sigmoid(x)  # [n, d, m1]
            # soft clamp the denominator to prevent divide by zero and prevent very large weight increases
            x = (x - x.mean(dim=0).detach()) / torch.sqrt(
                (self.nonlinear_clamp + x.var(dim=0).detach())
            )

            x = layer(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    @property
    def bias(self) -> Union[np.ndarray, None]:
        """
        Get the vector of feature biases

        Returns:
            bias vector if use_bias is True, otherwise None
        """
        bias = self.dag_layer_bias
        return bias if bias is None else bias.cpu().detach().numpy()

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

        # calculate the adjacency matrix after the fitting is finished
        self.adj = (
            self._calculate_adj(X_torch, mean_effect=False).cpu().detach().numpy()
        )
        self.adj_mean_effect = (
            self._calculate_adj(X_torch, mean_effect=True).cpu().detach().numpy()
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

        def _get_flat_bounds(
            params: List[torch.Tensor],
        ) -> List[Tuple[Union[None, float]]]:
            """
            Get bound constraint for each parameter in flatten vector form from the parameters of the model

            Args:
                params: parameters of the model

            Returns:
                flatten vector of bound constraints for each parameter in numpy form
            """
            bounds = []
            for p in params:
                try:
                    b = p.bounds
                except AttributeError:
                    b = [(None, None)] * p.numel()
                bounds += b
            return bounds

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

            n_features = X.shape[1]

            X_hat = self(X)
            h_val = self._h_func()

            loss = (0.5 / X.shape[0]) * torch.sum((X_hat - X) ** 2)
            lagrange_penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            # NOTE: both the l2 and l1 regularization are NOT applied to the bias parameters
            l2_reg = 0.5 * self.ridge_beta * self._l2_reg(n_features)
            l1_reg = self.lasso_beta * self._l1_reg(n_features)

            primal_obj = loss + lagrange_penalty + l2_reg + l1_reg
            primal_obj.backward()
            loss = primal_obj.item()

            flat_grad = _get_flat_grad(params)
            return loss, flat_grad.astype("float64")

        optimizer = torch.optim.Optimizer(self.parameters(), dict())
        params = optimizer.param_groups[0]["params"]

        flat_params = _get_flat_params(params)
        bounds = _get_flat_bounds(params)

        while rho < rho_max:
            # Magic
            sol = sopt.minimize(
                _func,
                flat_params,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
            )

            _update_params_from_flat(params, sol.x)
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
        Constrain 2-norm-squared of dag_layer weights of the model along m1 dim to be a DAG

        Returns:
            DAGness of the adjacency matrix
        """
        d = self.dims[0]
        d_torch = torch.tensor(d).to(self.device)  # pylint: disable=not-callable

        # only consider the dag_layer for h(W) for compute efficiency
        dag_layer_weight = self.dag_layer_weight.view(d, -1, d)  # [j, m1, i]
        square_weight_mat = torch.sum(
            dag_layer_weight * dag_layer_weight, dim=1
        ).t()  # [i, j]

        # h = trace_expm(a) - d  # (Zheng et al. 2018)
        characteristic_poly_mat = (
            torch.eye(d).to(self.device) + square_weight_mat / d_torch
        )  # (Yu et al. 2019)
        polynomial_mat = torch.matrix_power(characteristic_poly_mat, d - 1)
        h = (polynomial_mat.t() * characteristic_poly_mat).sum() - d
        return h

    def _l1_reg(self, n_features: int) -> torch.Tensor:
        """
        Take average l1 of all weight parameters of the model.
        NOTE: regularisation needs to be scaled up by the number of features
        because the loss scales with feature number.

        Returns:
            l1 regularisation term.
        """
        return torch.mean(torch.abs(self.dag_layer_weight)) * n_features

    def _l2_reg(self, n_features: int) -> torch.Tensor:
        """
        Take average 2-norm-squared of all weight parameters of the model.
        NOTE: regularisation needs to be scaled up by the number of features
        because the loss scales with feature number.

        Returns:
            l2 regularisation term.
        """
        reg = 0.0
        reg += torch.sum(self.dag_layer_weight ** 2)
        for layer in self.loc_lin_layer_weights:
            reg += torch.sum(layer.weight ** 2)

        # calculate the total number of elements used in the above sums
        n_elements = self.dag_layer_weight.numel()
        for layer in self.loc_lin_layer_weights:
            n_elements = n_elements + layer.weight.numel()
        return reg / n_elements * n_features

    def _calculate_adj(self, X: torch.Tensor, mean_effect: bool) -> torch.Tensor:
        """
        Calculate the adjacency matrix.

        For the linear case, this is just dag_layer_weight.
        For the nonlinear case, approximate the relationship using the gradient of X_hat wrt X.
        """

        # for the linear case, save compute by just returning the dag_layer weights
        if len(self.dims) <= 2:
            adj = (
                self.dag_layer_weight.T
                if mean_effect
                else torch.abs(self.dag_layer_weight.T)
            )
            return adj

        _, n_features = X.shape
        # get the data X and reconstruction X_hat
        X = X.clone().requires_grad_()
        X_hat = self(X).sum(dim=0)  # shape = (n_features,)

        adj = []
        # iterate over sums of reconstructed features
        for j in range(n_features):

            # calculate the gradient of X_hat wrt X
            ddx = torch.autograd.grad(X_hat[j], X, create_graph=True)[0]

            if mean_effect:
                # get the average effect
                adj.append(ddx.mean(axis=0).unsqueeze(0))
            else:
                # otherwise, use the average L1 of the gradient as the W
                adj.append(torch.abs(ddx).mean(dim=0).unsqueeze(0))
        adj = torch.cat(adj, dim=0)

        # transpose to get the adjacency matrix
        return adj.T
