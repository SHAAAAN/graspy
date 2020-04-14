# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy as sp
from scipy import stats

from ..embed import AdjacencySpectralEmbed, select_dimension
from ..utils import import_graph, is_symmetric
from ..align import SeedlessProcrustes
from .base import BaseInference


class LatentDistributionTest(BaseInference):
    """
    Two-sample hypothesis test for the problem of determining whether two
    random dot product graphs have the same distributions of latent positions.
    
    This test can operate on two graphs where there is no known matching
    between the vertices of the two graphs, or even when the number of vertices
    is different. Currently, testing is only supported for undirected graphs.

    Read more in the :ref:`tutorials <inference_tutorials>`

    Parameters
    ----------
    n_components : int or None, optional (default=None)
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.

    n_bootstraps : int (default=200)
        Number of bootstrap iterations.

    bandwidth : float, optional (default=0.5)
        Bandwidth to use for gaussian kernel. If None,
        the median heuristic will be used.

    pass_graph : boolean, optional (default=True)
        Changes the expected input form full graphs to embeddings if False.

    alignment : None, or string, {'sign_flips' (default), 'seedless_procrustes'}
        There is an inherent non-identifiability when using estimating the
        latent positions of an RDPG graph 

        - None
            does not perform any alignment.
        - 'seedless_procrustes'
            uses the EM seedless procrustes procedure to align two graphs,
            see Agterberg et al. (2020) TODO cite when possible
            note that it initializes with a Q obtained from the sign flips trick.
            computionally expensive for large number of the dimensions;
            equivalent to 'sign_flips' in the n_components=1 case.
        - 'sign_flips'
            for each dimension, if two embeddings have medians with varying signs,
            flips all signs along this dimension for one of hte graphs.
   
    size_correction : None (default), or string, {'sampling', 'expected'}
        The test degrades in validity when the sizes of two graphs diverge from
        each other, unless the way kernel matrix is computed is modified.
        See Alyakin & Agterberg & Helm & Priebe. TODO cite when possible.

        - None
            does not perform any modifications
        - 'sampling'
            in case when two graphs are not of equal sizes, estimates the
            plug-in estimator for the variance and uses it to inject
            appropriately scaled noise into the embedding of the larger graph.
            overlaps exactly with no modifications for graphs of the same size
        - 'expected'
            in case when two graphs are not of equal sizes, estimates the
            plug-in estimator for the variance and uses it to computes expected
            gaussian kernel between two points if noise was to be injected.
            overlaps exactly with no modifications for graphs of the same size

    embedding : string, { 'ase' (default), 'omnibus'}
        String describing the embedding method to use:


    Attributes
    ----------
    sample_T_statistic_ : float
        The observed difference between the embedded latent positions of the two 
        input graphs.

    p_ : float
        The overall p value from the test.
    
    null_distribution_ : ndarray, shape (n_bootstraps, )
        The distribution of T statistics generated under the null.

    References
    ----------
    .. [1] Tang, M., Athreya, A., Sussman, D. L., Lyzinski, V., & Priebe, C. E.
        (2017). "A nonparametric two-sample hypothesis testing problem for
        random graphs." Bernoulli, 23(3), 1599-1630.
    """

    def __init__(
        self,
        n_components=None,
        n_bootstraps=200,
        bandwidth=None,
        pass_graph=True,
        alignment="sign_flips",
        size_correction=None,
        n_samples=1,
    ):
        if n_components is not None:
            if not isinstance(n_components, int):
                msg = "n_components must an int, not {}.".format(type(n_components))
                raise TypeError(msg)

        if not isinstance(n_bootstraps, int):
            msg = "n_bootstraps must an int, not {}".format(type(n_bootstraps))
            raise TypeError(msg)
        elif n_bootstraps < 1:
            msg = "{} is invalid number of bootstraps, must be greater than 1"
            raise ValueError(msg.format(n_bootstraps))

        if bandwidth is not None and not isinstance(bandwidth, float):
            msg = "bandwidth must an int, not {}".format(type(bandwidth))
            raise TypeError(msg)

        if not isinstance(pass_graph, bool):
            msg = "pass_graph must be a bool, not {}".format(type(pass_graph))
            raise TypeError(msg)

        if alignment is None:
            pass
        elif not isinstance(alignment, str):
            msg = "alignment must be None or a str, not {}".format(type(alignment))
            raise TypeError(msg)
        else:
            alignments_supported = ["sign_flips", "seedless_procrustes"]
            if alignment not in alignments_supported:
                msg = "supported alignments are {}".fomat(alignment)
                raise NotImplementedError(msg)

        if size_correction is None:
            pass
        elif not isinstance(size_correction, str):
            msg = "size_correction must be None or a str, not {}".format(
                type(size_correction)
            )
            raise TypeError(msg)
        else:
            size_corrections_supported = ["sampling", "expected"]
            if size_correction not in size_corrections_supported:
                msg = "supported size corrections are {}".fomat(
                    size_corrections_supported
                )
                raise NotImplementedError(msg)

        if not isinstance(n_samples, int):
            msg = "n_samples must an int, not {}".format(type(n_samples))
            raise TypeError(msg)
        elif n_samples < 0:
            msg = "{} is invalid number of samples, must be greater than 1"
            raise ValueError(msg.format(n_samples))

        super().__init__(embedding="ase", n_components=n_components)

        self.n_bootstraps = n_bootstraps
        self.bandwidth = bandwidth
        # moved this to here out of the methods
        # TODO implement adaptively selected bandwidth
        if self.bandwidth is None:
            self.bandwidth = 0.5
        self.pass_graph = pass_graph

        self.alignment = alignment

        if size_correction == "sampling":
            self.sampling = True
            self.expected = False
        elif size_correction == "expected":
            self.sampling = False
            self.expected = True
        else:
            self.sampling = False
            self.expected = False

        self.samples = n_samples

    def _fit_plug_in_variance_estimator(self, X):
        """
        Takes in ASE of a graph and returns a function that estimates
        the variance-covariance matrix at a given point using the
        plug-in estimator from the RDPG Central Limit Theorem.
        (Athreya et al., RDPG survey, Equation 10)

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            adjacency spectral embedding of a graph

        Returns
        -------
        plug_in_variance_estimtor: functions
            a function that estimates variance (see below)
        """
        n = len(X)
        delta = 1 / (n) * (X.T @ X)
        delta_inverse = np.linalg.inv(delta)

        def plug_in_variance_estimator(x):
            """
            Takes in a point of a matrix of points in R^d and returns an
            estimated covariance matrix for each of the points

            Parameters:
            -----------
            x: np.ndarray, shape (n, d)
                points to estimate variance at
                if 1-dimensional - reshaped to (1, d)

            Returns:
            -------
            covariances: np.ndarray, shape (n, d, d)
                n estimated variance-covariance matrices of the points provided
            """
            if x.ndim < 2:
                x = x.reshape(1, -1)
            middle_term_scalar = x @ X.T - (x @ X.T) ** 2
            middle_term_matrix = np.einsum("bi,bo->bio", X, X)  # can be precomputed
            middle_term = np.tensordot(middle_term_scalar, middle_term_matrix, axes=1)
            # preceeding three lines are a vectorized version of this
            # middle_term = 0
            # for i in range(n):
            #     middle_term += np.multiply.outer((x @ X[i] - (x @ X[i]) ** 2),
            #                                      np.outer(X[i], X[i]))
            covariances = delta_inverse @ (middle_term / n) @ delta_inverse
            return covariances

        return plug_in_variance_estimator

    # def _estimate_correction_variances(self, X_hat, Y_hat, pooled=True):
    #     N, d_X = X_hat.shape  # dont really need to do this (n_components)
    #     M, d_Y = Y_hat.shape
    #     if N == M:
    #         X_sigmas = np.zeros((N, d_X, d_X))
    #         Y_sigmas = np.zeros((M, d_Y, d_Y))
    #     elif N > M:
    #         if pooled:
    #             two_samples = np.concatenate([X_hat, Y_hat], axis=0)
    #             get_sigma = self._fit_plug_in_variance_estimator(two_samples)
    #         else:
    #             get_sigma = self._fit_plug_in_variance_estimator(X_hat)
    #         X_sigmas = get_sigma(X_hat) * (N - M) / (N * M)
    #         Y_sigmas = np.zeros((M, d_Y, d_Y))
    #     else:
    #         if pooled:
    #             two_samples = np.concatenate([X_hat, Y_hat], axis=0)
    #             get_sigma = self._fit_plug_in_variance_estimator(two_samples)
    #         else:
    #             get_sigma = self._fit_plug_in_variance_estimator(X_hat)
    #         X_sigmas = np.zeros((N, d_X, d_X))
    #         Y_sigmas = get_sigma(Y_hat) * (M - N) / (N * M)
    #     return X_sigmas, Y_sigmas

    def _estimate_correction_variances(self, X_hat, Y_hat, pooled=True):
        N, d_X = X_hat.shape  # dont really need to do this (n_components)
        M, d_Y = Y_hat.shape
        if N == M:
            X_sigmas = np.zeros((N, d_X, d_X))
            Y_sigmas = np.zeros((M, d_Y, d_Y))
        elif N > M:
            X_sigmas = np.ones((N, d_Y, d_Y)) * (N - M) / (N * M)
            Y_sigmas = np.zeros((M, d_Y, d_Y))
        else:
            X_sigmas = np.zeros((N, d_X, d_X))
            Y_sigmas = np.ones((M, d_Y, d_Y)) * (M - N) / (N * M)
        return X_sigmas, Y_sigmas

    def _sample_modified_ase(self, X, Y):
        n = len(X)
        m = len(Y)
        if n == m:
            return X, Y
        elif n > m:
            X_sigmas, _ = self._estimate_correction_variances(X, Y)
            X_sampled = np.zeros(X.shape)
            for i in range(n):
                X_sampled[i, :] = X[i, :] + stats.multivariate_normal.rvs(
                    cov=X_sigmas[i]
                )
            return X_sampled, Y
        else:
            _, Y_sigmas = self._estimate_correction_variances(X, Y)
            Y_sampled = np.zeros(Y.shape)
            for i in range(m):
                Y_sampled[i, :] = Y[i, :] + stats.multivariate_normal.rvs(
                    cov=Y_sigmas[i]
                )
            return X, Y_sampled

    def _rbfk_matrix_regular(self, X, Y, X_sigmas=None, Y_sigmas=None):
        # a regular rbfk kernel
        # happens to be a specific case of the expected rbfk matrix
        # when the sigmas are zero matrices, they should overlap
        # this is faster and also less prone to potential bugs
        diffs = np.expand_dims(X, 1) - np.expand_dims(Y, 0)
        kernel_matrix = np.exp(-0.5 * np.sum(diffs ** 2, axis=2) / self.bandwidth ** 2)
        return kernel_matrix

    def _rbfk_matrix_expected(self, X, Y, X_sigmas, Y_sigmas):
        # use the appropriately broadcasted formula:
        # if    Z ~ N(mu, Sigma), c constant
        # then  E[exp(-c Z^T Z)]  =  exp(- c mu^T (I + 2 c Sigma)^{-1} mu)
        #                              / det (I + 2c Sigma)^{1/2}
        n, d = X.shape
        m, _ = Y.shape

        c = 1 / (2 * self.bandwidth ** 2)
        mu = np.expand_dims(X, 1) - np.expand_dims(Y, 0)
        sigma = np.expand_dims(X_sigmas, 1) + np.expand_dims(Y_sigmas, 0)

        inverted_matrix = np.linalg.inv(np.eye(d) + 2 * c * sigma)
        numer = np.exp(
            -c * np.expand_dims(mu, -2) @ inverted_matrix @ np.expand_dims(mu, -1)
        )
        denom = np.linalg.det(np.eye(d) + 2 * c * sigma) ** (1 / 2)
        kernel_matrix = numer.reshape(n, m) / denom

        return kernel_matrix

    def _kernel_matrix(self, X, Y, X_sigmas, Y_sigmas):
        # compute the within and between the samplesa kernel matrices
        X_rbfk = self._rbfk_matrix(X, X, X_sigmas, X_sigmas)
        np.fill_diagonal(X_rbfk, 1)
        Y_rbfk = self._rbfk_matrix(Y, Y, Y_sigmas, Y_sigmas)
        np.fill_diagonal(Y_rbfk, 1)
        XY_rbfk = self._rbfk_matrix(X, Y, X_sigmas, Y_sigmas)
        # assemble one large kernel matrix
        full_kernel_matrix = np.block([[X_rbfk, XY_rbfk], [XY_rbfk.T, Y_rbfk]])
        return full_kernel_matrix

    def _statistic(self, kernel_matrix, N, M):
        # extract within and between samples matrices kernel matrices
        X_rbfk = kernel_matrix[:N, :N]
        Y_rbfk = kernel_matrix[N:, N:]
        XY_rbfk = kernel_matrix[N:, :N]
        # compute the test statistic
        X_stat = np.sum(X_rbfk - np.eye(N)) / (N * (N - 1))
        Y_stat = np.sum(Y_rbfk - np.eye(M)) / (M * (M - 1))
        XY_stat = np.sum(XY_rbfk) / (N * M)
        total_stat = X_stat - 2 * XY_stat + Y_stat
        return total_stat

    def _embed(self, A1, A2):
        ase = AdjacencySpectralEmbed(n_components=self.n_components)
        X1_hat = ase.fit_transform(A1)
        X2_hat = ase.fit_transform(A2)
        if isinstance(X1_hat, tuple) and isinstance(X2_hat, tuple):
            X1_hat = np.concatenate(X1_hat, axis=-1)
            X2_hat = np.concatenate(X2_hat, axis=-1)
        elif isinstance(X1_hat, tuple) ^ isinstance(X2_hat, tuple):
            raise ValueError("Input graphs do not have same directedness")
        return X1_hat, X2_hat

    def _bootstrap(self, kernel_matrix, N, M, bootstraps):
        statistics = np.zeros(bootstraps)
        for i in range(bootstraps):
            permutation = np.random.choice(
                np.arange(0, N + M), size=int(N + M), replace=False
            )
            shuffled_kernel_matrix = kernel_matrix[permutation, :][:, permutation]
            statistics[i] = self._statistic(shuffled_kernel_matrix, N, M)
        return statistics

    def fit(self, A, B):
        """
        Fits the test to the two input graphs

        Parameters
        ----------
        A, B : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            The two graphs to run a hypothesis test on or two embeddings if
            pass_graph was set to false

        Returns
        -------
        p_ : float
            The p value corresponding to the specified hypothesis test
        """
        if self.pass_graph:
            A = import_graph(A)
            B = import_graph(B)
            if self.n_components is None:
                # get the last elbow from ZG for each and take the maximum
                num_dims1 = select_dimension(A)[0][-1]
                num_dims2 = select_dimension(B)[0][-1]
                self.n_components = max(num_dims1, num_dims2)
            X_hat, Y_hat = self._embed(A, B)
        else:
            X_hat, Y_hat = A, B

        # perform sign slips or seedless procrustes
        if self.alignment == "sign_flips":
            aligner = SeedlessProcrustes()
            Q = aligner._sign_flips(X_hat, Y_hat)
            Y_hat = Y_hat @ Q
        elif self.alignment == "seedless_procrustes":
            aligner = SeedlessProcrustes()
            Q = aligner.fit_predict(X_hat, Y_hat)
            Y_hat = Y_hat @ Q

        if self.sampling:
            if self.samples == 0:
                self._rbfk_matrix = self._rbfk_matrix_regular
                kernel_matrix = self._kernel_matrix(X_hat, Y_hat, 0 , 0)
            else:
                self._rbfk_matrix = self._rbfk_matrix_regular
                kernel_matrices = []
                for i in range(self.samples):
                    X_hat_temp, Y_hat_temp = self._sample_modified_ase(X_hat, Y_hat)
                    kernel_matrices.append(self._kernel_matrix(X_hat_temp, Y_hat_temp, 0 , 0))
                kernel_matrix = np.mean(kernel_matrices, axis=0)
        elif self.expected:
            X_sigmas, Y_sigmas = self._estimate_correction_variances(X_hat, Y_hat)
            self._rbfk_matrix = self._rbfk_matrix_expected
            kernel_matrix = self._kernel_matrix(X_hat, Y_hat, X_sigmas, Y_sigmas)
        else:
            self._rbfk_matrix = self._rbfk_matrix_regular
            kernel_matrix = self._kernel_matrix(X_hat, Y_hat, 0 , 0)

        self.kernel_matrix_ = kernel_matrix
        U = self._statistic(kernel_matrix, len(X_hat), len(Y_hat))
        self.null_distribution_ = self._bootstrap(
            kernel_matrix, len(X_hat), len(Y_hat), self.n_bootstraps
        )

        # compute the value
        self.sample_T_statistic_ = U
        self.p_ = (np.sum(self.null_distribution_ >= U) + 1) / (self.n_bootstraps + 1)
        return self.p_
