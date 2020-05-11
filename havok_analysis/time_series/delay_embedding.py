'''

      Project: HAVOK analysis
         File: delay_embedding.py
 File Created: 03.03.2020
       Author: Jan Zetzmann (yangntze+github_havok@gmail.com)
-----
Last Modified: 11.05.2020 21:55:55
  Modified By: Jan Zetzmann (yangntze+github_havok@gmail.com)
-----
    Copyright: 2020 havokanalysisteam

'''


import numpy as np
from scipy.linalg import hankel, svd, diagsvd, norm


class DelayEmbedding():
    """
    INPUTS / necessary input variables:
        time_series
            np.array $\in \mathbb{R}^{m \times n}$
        embedding_dimension
            integer $\in \mathbb{N}$
            Full space of the non linear dynamical system, that also contains 
            the attractor. This is going to be the height of the Hankel matrix
    METHODS:
        hankel_matrix(self)                                         (property)
            Returns the Hankel matrix of given time_series
            Calculated each call - becomes critical only above 1e6 time points
        hankel_matrix_shape(self)                                   (property)
            Returns tuple containing the shape of the Hankel matrix
        singular_values(self)                                       (property)
            Returns singular values as an array
        singular_value_matrix(self)                                 (property)
            Returns singular values as an diagonal matrix
        get_reduced_svd(self, cutoff=0)
            Returns the dimension reduced SVD u,s,v
        get_reconstructed_hankel(self, cutoff=0)
            Returns the reconstructed Hankel matrix from possibly reduced SVDs
        reconstruct_hankel(u, s, vh)                                (static)
            Reconstructs Hankel matrices from SVD returns
        determine_embedding_dimension(self)
            not yet implemented
        determine_cutoff_dimension(self, method='gavish_donoho_2014', **kwargs)
            Finds the number of relevant dimensions and sets num_rel_dim

        _cut(u_mat, s_vec, v_mat, cutoff)                           (static)
            Takes the SVD result and cuts it
            (not VH!)
        _cut_usvh(u_mat, s_vec, vh_mat, cutoff)                     (static)
            Takes the SVD result and cuts it

    Relevant dimension determination techniques (method-options for determine_cutoff_dimension)
        _gavish_donoho_2014(self)
        _close_hankel_reconstruction(self, **kwargs)
        _hankel_diff_in_norm(self, measure_fnc_key, cutoff=0, **kwargs)
        _sparse_svd_like(self)
            
    NOTES:
        cutoff
            formerly num_rel_dim / number of relevant dimensions
            integer $\in \mathbb{N}$
            Number of relevant dimensions that should suffice to model dynamics
    """
    def __init__(
        self, 
        time_series, 
        embedding_dimension
        ):
        self.time_series            = time_series
        self.embedding_dimension    = embedding_dimension
        self.num_rel_dim            = 0
        self._u, self._s, self._vh  = svd(
            self.hankel_matrix,
            full_matrices   = False,
            compute_uv      = True
            )

    @property
    def hankel_matrix_shape(self):
        return (
            self.embedding_dimension,
            (len(self.time_series) - self.embedding_dimension + 1)
            )

    @property
    def hankel_matrix(self):
        return hankel(
            self.time_series[:self.embedding_dimension], 
            self.time_series[self.embedding_dimension-1:]
            )

    @property
    def singular_values(self):
        if self.num_rel_dim==0:
            return self._s
        else:
            return self._s[:self.num_rel_dim]

    @property
    def singular_value_matrix(self):
        if self.num_rel_dim==0:
            return diagsvd(self._s, len(self._s), len(self._s))
        else:
            return diagsvd(self._s[:self.num_rel_dim], self.num_rel_dim, self.num_rel_dim)

    def get_reduced_svd(self, cutoff=0):
        """
            returns the dimension reduced SVD u,s,v
            INPUTS:
                cutoff                                                  (optional)
                        integer maximal dimension to include $\in \mathbb{N}$
                -1 full matrices
                    0 use num_rel_dim (which -if not set- defaults to 0)
                    n arbitrary number of dimensions to include
        """
        if      cutoff == -1:
            # flag for full matrices
            return self._u, self._s, np.conjugate(np.transpose(self._vh))
        elif    cutoff == 0:
            # default case flag
            if self.num_rel_dim == 0:
                # num_rel_dim not set -> give me all
                return self._u, self._s, np.conjugate(np.transpose(self._vh))
            else:
                # return previously set number of relevant dimensions
                return self._cut(self._u, self._s,
                                 np.conjugate(np.transpose(self._vh)),
                                 self.num_rel_dim)
        elif    cutoff > 0:
            # user defined dimension
            return self._cut(self._u, self._s,
                             np.conjugate(np.transpose(self._vh)), cutoff)
        else:
            raise ValueError('Provide an integer cutoff in [-1, inf]!')

    def get_reconstructed_hankel(self, cutoff=0):
        """
            returns the reconstructed Hankel matrix from possibly reduced SVDs
            INPUTS:
                cutoff                                                  (optional)
                    integer maximal dimension to include $\in \mathbb{N}$
                    -1 full matrices
                    0 use num_rel_dim (which -if not set- defaults to 0)
                    n arbitrary number of dimensions to include
        """
        if      cutoff == -1:
            # flag for full matrices
            return DelayEmbedding.reconstruct_hankel(self._u, self._s, self._vh)
        elif    cutoff == 0:
            # default case flag
            if self.num_rel_dim == 0:
                # num_rel_dim not set -> give me all
                return DelayEmbedding.reconstruct_hankel(self._u, self._s, self._vh)
            else:
                # return previously set number of relevant dimensions
                return DelayEmbedding.reconstruct_hankel(*self._cut_usvh(self._u, self._s, self._vh, self.num_rel_dim))
        elif    cutoff > 0:
            # user defined dimension
            return DelayEmbedding.reconstruct_hankel(*self._cut_usvh(self._u, self._s, self._vh, cutoff))
        else:
            raise ValueError('Provide an integer cutoff in [-1, inf)!')

    @staticmethod
    def reconstruct_hankel(u, s, vh):
        """
        Reconstructs Hankel matrices from SVD returns
        INPUT
            u,s,vh as yielded from SVD
        """
        return np.dot(u * s, vh)

    @staticmethod
    def _cut(u_mat, s_vec, v_mat, cutoff):
        """
        Takes the SVD result U, s, V
        and cuts it to return U_cut, s_cut, V_cut
        Make sure not to input VH but 
                np.conjugate(np.transpose(vh_mat))
        see _cut_usvh() for that
        """
        if cutoff > len(s_vec):
            print('Warning: Specified cutoff is greater than number of singular values! Will proceed anyways ...')
        elif cutoff < -len(s_vec):
            raise ValueError('Specified cutoff is smaller than to negative number of singular values! This would yield an empty resulting array.')
        return  u_mat[:, :cutoff], \
                s_vec[   :cutoff], \
                v_mat[:, :cutoff]

    @staticmethod
    def _cut_usvh(u_mat, s_vec, vh_mat, cutoff):
        """
        Takes the SVD result U, s, VH
        and cuts it to return U_cut, s_cut, VH_cut
        Make sure not to input V but VH
        see _cut() for cutting V
        """
        if cutoff > len(s_vec):
            print('Warning: Specified cutoff is greater than number of singular values! Will proceed anyways ...')
        elif cutoff < -len(s_vec):
            raise ValueError('Specified cutoff is smaller than to negative number of singular values! This would yield an empty resulting array.')
        return  u_mat[:, :cutoff], \
                s_vec[:cutoff], \
                vh_mat[:cutoff,:]

    def determine_embedding_dimension(self):
        print('determine_embedding_dimension:\nHello my dear user, I do not do anything right now. If you need me, feel free to implement me')

    def determine_cutoff_dimension(self, method='gavish_donoho_2014', **kwargs):
        """
        Finds the number of relevant dimensions
        Returns and sets the num_rel_dim variable
        INPUTS
            method
                string                                              (optional)
                'gavish_donoho_2014'                                (default)
                    Uses the threshold calculation from the paper The Optimal Hard Threshold for Singular Values is $4/\sqrt {3}$ by Gavish Donoho 2014
                'sparse_svd_like'
                    Mimics the sparse svd implementation scipy/sparse/linalg/eigen/arpack/arpack.py#L1724-L1910
                'close_hankel_reconstruction'
                    Compares the reconstruction of the Hankel matrix
                    Needs/allows for the following optional kwargs:
                        threshold
                        measure                                     (optional)    
                            allclose
                            frobenius
                            norm
                # todo?
                # 'time_series_reconstruction_error'
                #     Compares the reconstruction of the Hankel matrix
                #     Needs/allows for the following kwargs:
                #         measure
                #             allclose
                #             frobenius
                #             norm
        OUTPUT
            num_rel_dim
                integer
                The found number of relevant dimensions
        """
        # https://jaxenter.com/implement-switch-case-statement-python-138315.html
        # switcher = {
        #         'gavish_donoho_2014':             self._paper_cutoff,
        #         'sparse_svd_like':                self._sparse_cutoff,
        #         'close_hankel_reconstruction':    self._hankel_reconstruction_cutoff,
        #     }
        # # Get the function from switcher dictionary
        # func_pointer = switcher.get(method, lambda: print("Invalid method"))

        # for key in kwargs.keys():
        #     if key == 'method':
        #         kwargs[key] = ''.join([ '_', kwargs.pop(kwargs[key], 'gavish_donoho_2014') ])
        
        cls_method_to_use = ''.join([ '_', method ])

        func_pointer = getattr(self, cls_method_to_use, lambda: print("Invalid method", cls_method_to_use, kwargs))
        # methods need be named like strings, prefixed with _

        # print(func_pointer)
        # print(kwargs)

        # Execute the function and return the result
        self.num_rel_dim = func_pointer(**kwargs)
        return self.num_rel_dim

    def get_reconstructed_hankel_errors(self, max_dim=100, **kwargs):
        measure_fnc_key = kwargs.pop('measure_fnc_key', 'frobenius')
        errors = np.zeros(max_dim)
        dimensions = range(1,max_dim+1)
        for k in dimensions:
            errors[k-1] = self._hankel_diff_in_norm(measure_fnc_key, cutoff=k, **kwargs)
        return errors, np.array(dimensions)

    # BEGIN relevant dimension determination techniques (method-options for determine_cutoff_dimension)

    def _gavish_donoho_2014(self):
        print('_gavish_donoho_2014')
        beta = self.hankel_matrix_shape[0] / self.hankel_matrix_shape[1]
        assert beta > 0 or beta < 1
        w = (8 * beta) / (beta + 1 + np.sqrt(beta**2 + 14 * beta + 1))
        lambda_star = np.sqrt(2 * (beta + 1) + w)
        threshold = lambda_star * np.median(self._s)
        r = np.sum(self._s > threshold)
        # print('estimated embedding dimension: {:d}'.format(r))
        return r

    def _close_hankel_reconstruction(self, **kwargs):

        # if not 'measure' in kwargs:
        #     print('No measure found, using Frobenius.')
        #     measure_fnc_key = 'frobenius'
        # else:
        measure_fnc_key = kwargs.pop('measure', 'frobenius') # kwargs['measure'] and remove
        print('_close_hankel_reconstruction', measure_fnc_key)

        res = 0
        if measure_fnc_key=='allclose':
            for k in range(2,100):
                if np.allclose(self.hankel_matrix, self.get_reconstructed_hankel(cutoff=k), **kwargs):
                    res = k
                    break
        else:
            if not 'threshold' in kwargs:
                raise Exception('No threshold found!')
            else:
                threshold = kwargs.pop('threshold', 1e-4) # kwargs['threshold'] and remove

            # print(measure_fnc_key)
            # print('kargs: ', kwargs)

            for k in range(2,100):
                if (
                    self._hankel_diff_in_norm(measure_fnc_key, cutoff=k, **kwargs)
                    <= threshold
                    ):
                    res = k
                    break
        return res
    
    def _hankel_diff_in_norm(self, measure_fnc_key, cutoff=0, **kwargs):
        # https://jaxenter.com/implement-switch-case-statement-python-138315.html
        switcher = {
                'frobenius'     : lambda x,y: norm((x-y), ord='fro'),
                'norm'          : lambda x,y: norm((x-y), **kwargs)
        }
        # }[measure_fnc_key](self.hankel_matrix, self.get_reconstructed_hankel)

        # Get the function from switcher dictionary
        func_pointer = switcher.get(measure_fnc_key, lambda: print("Invalid method"))
        # calculate the error and return
        return func_pointer(self.hankel_matrix, self.get_reconstructed_hankel(cutoff))

    def _sparse_svd_like(self):
        """
        Use the 'sophisticated' way of cutting small values as in the sparse svd
        implementation scipy/sparse/linalg/eigen/arpack/arpack.py#L1724-L1910 in
        line 1875
        """
        print('_sparse_svd_like')
        # Use the sophisticated detection of small eigenvalues from pinvh.
        t = self._s.dtype.char.lower()
        factor = {'f': 1E3, 'd': 1E6}
        cond = factor[t] * np.finfo(t).eps
        cutoff_precision = cond * np.max(self._s)
        return np.where(self._s > cutoff_precision)[0][0]

    # END relevant dimension determination techniques
