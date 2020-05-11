'''

      Project: HAVOK analysis
         File: sparse_regression.py
 File Created: 02.03.2020
       Author: Paul Wilhelm (paul.wilhelm@stud.uni-goettingen.de)
-----
Last Modified: 11.05.2020 21:53:50
  Modified By: Jan Zetzmann (yangntze+github_havok@gmail.com)
-----
    Copyright: 2020 havokanalysisteam

'''


import numpy as np
from sklearn.linear_model import MultiTaskLasso, MultiTaskLassoCV, \
MultiTaskElasticNetCV, MultiTaskElasticNet
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from functools import partial


class SparseRegression:
    def __init__(self, v, delta_v, f, q, lin_args=(),
                 force_args=(), split='shuffle', split_kargs={}):
        """
        v.shape = (n_steps, n_variables),
        delta_v.shape = (n_steps, n_variables)

        q (in [1, n_variables]) number of first variables
        to fit the linear model to,
        remaining n_variables-q are used as forcing

        f: (n_steps, n_variables) -> (n_steps, n_features)
        f will be called with f(..., *lin_args) when fitting the linear model
        and with f(..., *force_args) when fitting the force
        """
        if v.shape == delta_v.shape and type(q) == int and q > 0 \
                and q <= v.shape[1]:
            self.v, self.delta_v = self._check_reduce(v, delta_v)
            self.params = [*self.v.shape, q]  # [n_steps, n_vars, q]
            # derivatives used for the model
            self.delta_v = self.delta_v[:, :q]
            # calculate features based on first q variables for linear model
            self.features_lin_model = f(self.v[:, :q], *lin_args)
            # calculate features based on remaining variables for forcing term
            self.features_forcing = f(self.v[:, q:], *force_args)
            # two different types of splitting
            split_dict = {'shuffle': self._shuffle_split,
                          'lorenz': self._lobes_split}
            # split the timesteps into two parts:
            # first is used for fitting linear model, second for forcing
            self.mask_l_m, self.mask_f = split_dict[split](**split_kargs)
            # self.mask_l_m, self.mask_f = self._split_lobes(self.v[:, 0])
            self.feature_generation = f
            self.feature_generation_args = {'linear': lin_args,
                                            'forcing': force_args}
        else:
            raise Exception('Error: invalid init parameter')

    def _shuffle_split(self, fraction=0.5):
        """
        creates two masks to split n_steps elements into two disjunct sets
        where the first has length=fraction*n
        """
        assert fraction > 0 and fraction < 1
        n_steps = self.params[0]
        n_1 = int(n_steps*fraction)
        shuffled_ind = np.random.permutation(n_steps)

        ind_1 = shuffled_ind[:n_1]
        mask_1 = np.zeros(n_steps, dtype=np.bool)
        mask_1[ind_1] = True

        ind_2 = shuffled_ind[n_1:]
        mask_2 = np.zeros(n_steps, dtype=np.bool)
        mask_2[ind_2] = True

        # each element is part of either one or the other mask
        assert np.all(mask_1 ^ mask_2)
        return mask_1, mask_2

    def _lobes_split(self, window_pos=200, window_neg=400):
        """
        use regions in which trajectories are on the lobes to fit the
        linear model and the remaining steps for modeling the force
        """
        v_1 = self.v[:, 0]
        n_steps = self.params[0]
        # find lobe switches
        m_pos = v_1 > 0
        m_neg = v_1 < 0
        mask_switch = (m_pos[:-1] & m_neg[1:]) | (m_neg[:-1] & m_pos[1:])
        switch_ind = np.nonzero(mask_switch)[0]
        print('no. of lobe switches detected in v_1: {:d}'
              .format(len(switch_ind)))
        force_ind_list = []
        for switch in switch_ind:
            if switch + 1 - window_neg < 0:
                l_neg = switch
            else:
                l_neg = window_neg
            if switch + 1 + window_pos > n_steps:
                l_pos = n_steps - switch
            else:
                l_pos = window_pos
            force_ind_list.append(np.arange(switch-l_neg, switch+l_pos))
        force_ind = np.concatenate(force_ind_list)
        assert np.all(force_ind >= 0) and np.all(force_ind < n_steps)

        mask_lobes = np.ones(n_steps, dtype=np.bool)
        mask_lobes[force_ind] = False
        mask_switch = np.zeros(n_steps, dtype=np.bool)
        mask_switch[force_ind] = True
        assert np.all(mask_lobes ^ mask_switch)
        return mask_lobes, mask_switch

    def _check_reduce(self, v, delta_v):
        """
        check both matrices for columns containg nan and excludes them
        """
        invalid_v = np.any(np.isnan(v), axis=1)
        if np.any(invalid_v):
            print('Warning: v matrix contains NaNs')
        invalid_delta_v = np.any(np.isnan(delta_v), axis=1)
        if np.any(invalid_delta_v):
            print('Warning: delta_v matrix contains NaNs')
        valid_steps = (~invalid_v) & (~invalid_delta_v)
        valid_fraction = np.sum(valid_steps) / len(valid_steps)
        if not np.isclose(valid_fraction, 1):
            print('Warning: only {:.1%} of time steps are valid'
                  .format(valid_fraction))
        if valid_fraction < 0.95:
            raise Exception('Error: less than 95% of time steps are valid')
        return v[valid_steps], delta_v[valid_steps]

    def fit_lin_model(self, alpha=None):
        """
        fit sparse linear regression on first q variables
        alpha is penalization parameter, None triggers cross validation
        """
        if alpha is None:  # do cross validation
            self.lin_model = \
                MultiTaskLassoCV(eps=1e-3, n_alphas=50, cv=10, n_jobs=-1,
                                 fit_intercept=False, normalize=False,
                                 max_iter=3500)
        else:
            self.lin_model = \
                MultiTaskLasso(alpha=alpha, fit_intercept=False,
                               normalize=False)
        self.lin_model.fit(self.features_lin_model[self.mask_l_m],
                           self.delta_v[self.mask_l_m])

    def pred_lin_model(self):
        """
        calculate prediction of the linear model on the data set not used for
        training it
        """
        pred_d_v = self.lin_model.predict(self.features_lin_model[self.mask_f])
        d_v = self.delta_v[self.mask_f]
        # calculate correlation for each variable
        n_variables = d_v.shape[1]
        print('corr. of prediction and true delta_v:')
        for i in range(n_variables):
            r, p = pearsonr(pred_d_v[:, i], d_v[:, i])
            print('{:d}th variable: r={:.2f} (p={:.2f})'.format(i+1, r, p))
        self.eps = d_v - pred_d_v  # d_v - Af(v)

    def fit_force_params(self, alpha=None):
        """
        fit sparse linear regression on remaining n_variables-q variables
        alpha is penalization parameter, None triggers cross validation
        """
        if alpha is None:  # do cross validation
            self.force_model = \
                MultiTaskLassoCV(eps=1e-3, n_alphas=50, cv=10, n_jobs=-1,
                                 fit_intercept=False, normalize=False)
        else:
            self.force_model = \
                MultiTaskLasso(alpha=alpha, fit_intercept=False,
                               normalize=False)
        self.force_model.fit(self.features_forcing[self.mask_f], self.eps)

    def fit(self, alpha_lin=None, alpha_force=None):
        self.fit_lin_model(alpha=alpha_lin)
        self.pred_lin_model()
        self.fit_force_params(alpha=alpha_force)

    def plot_coefs(self, f_descr=None):
        """
        plot coef matrix of linear and force model
        f_descr(n_vars, offset, *args) -> n_features
        """
        n_f_lin_model = self.features_lin_model.shape[1]
        n_f_forcing = self.features_forcing.shape[1]
        q = self.params[-1]
        if f_descr is not None:
            # get names of the features
            f_lin_model_str = f_descr(q, 0,
                                      *self.feature_generation_args['linear'])
            f_forcing_str = f_descr(self.v.shape[1] - q, q,
                                    *self.feature_generation_args['forcing'])
            assert len(f_lin_model_str) == n_f_lin_model
            assert len(f_forcing_str) == n_f_forcing
        else:
            f_lin_model_str = \
                [str(i) for i in range(n_f_lin_model)]
            f_forcing_str = \
                [str(i) for i in range(n_f_forcing)]

        n_f = n_f_lin_model + n_f_forcing
        fractions = (n_f_lin_model/n_f, n_f_forcing/n_f)
        fig, axes = plt.subplots(ncols=2, sharey=True,
                                 gridspec_kw={'width_ratios': fractions})
        plt.subplots_adjust(wspace=0.2)
        a = self.lin_model.coef_
        b = self.force_model.coef_
        assert a.shape[0] == b.shape[0]
        n_vars = a.shape[0]
        max_abs_coef = max(abs(a.min()), abs(b.min()), a.max(), b.max())

        titles = ['A', 'B']
        matrices = [a, b]
        ticklabels = [f_lin_model_str, f_forcing_str]
        for i, ax in enumerate(axes):
            ax.set_title(titles[i])
            im = ax.imshow(matrices[i], vmin=-max_abs_coef, vmax=max_abs_coef,
                           origin='upper', cmap='seismic')
            ax.set_xticks(np.arange(len(ticklabels[i])))
            ax.set_xticklabels(ticklabels[i], rotation=45)
            ax.set_xlabel('features')
            ax.set_yticks(np.arange(n_vars))
            ax.set_yticklabels(['$v_{:d}$'.format(i+1) for i in range(n_vars)])
        axes[0].set_ylabel('variables')
        plt.colorbar(im, ax=axes, fraction=0.05, shrink=0.75)

    def _dv(self, t, v, force):
        """
        v.shape = (q,)
        force(t)
        """
        # linear part
        lin_args = self.feature_generation_args['linear']
        features_lin = \
            self.feature_generation(v.reshape(1, -1), *lin_args).squeeze()
        lin_contr = np.dot(self.lin_model.coef_, features_lin)
        # forcing part
        force_args = self.feature_generation_args['forcing']
        features_force = \
            self.feature_generation(force(t).reshape(1, -1),
                                    *force_args).squeeze()
        force_contr = np.dot(self.force_model.coef_, features_force)
        dv = lin_contr + force_contr
        return dv

    def solve_model(self, dt, ind_v_init, force=None):
        """
        use time serie of the force variables and simulate the system from
        ind_v_init
        """
        n_steps, n_vars, q = self.params
        v_init = self.v[ind_v_init, :q]
        # resemble the timesteps at which the original data was evaluated
        n_remaining = n_steps - ind_v_init
        t_remaining = dt * (n_remaining - 1)
        t_eval = np.linspace(0, t_remaining, num=n_remaining)
        if force is None:
            def f_dummy(t): return np.zeros(n_vars - q)
            dv = partial(self._dv, force=f_dummy)
        elif force.shape == (n_remaining, n_vars - q):
            f_interp = interp1d(t_eval, force, axis=0, kind='quadratic')
            dv = partial(self._dv, force=f_interp)
        else:
            raise Exception('invalid force')

        result = solve_ivp(dv, [0, t_remaining], v_init, t_eval=t_eval,
                           method='RK45', rtol=1e-6, atol=1e-12)
        print(result.message)
        return result


class SRLorenz(SparseRegression):
    def _split(self, window_pos=200, window_neg=400):
        """
        use regions in which trajectories are on the lobes to fit the
        linear model and the remaining steps for modeling the force
        """
        v_1 = self.v[:, 0]
        n_steps = self.params[0]
        # find lobe switches
        m_pos = v_1 > 0
        m_neg = v_1 < 0
        mask_switch = (m_pos[:-1] & m_neg[1:]) | (m_neg[:-1] & m_pos[1:])
        switch_ind = np.nonzero(mask_switch)[0]
        print('no. of lobe switches detected in v_1: {:d}'
              .format(len(switch_ind)))
        force_ind_list = []
        for switch in switch_ind:
            if switch + 1 - window_neg < 0:
                l_neg = switch
            else:
                l_neg = window_neg
            if switch + 1 + window_pos > n_steps:
                l_pos = n_steps - switch
            else:
                l_pos = window_pos
            force_ind_list.append(np.arange(switch-l_neg, switch+l_pos))
        force_ind = np.concatenate(force_ind_list)
        assert np.all(force_ind >= 0) and np.all(force_ind < n_steps)

        mask_lobes = np.ones(n_steps, dtype=np.bool)
        mask_lobes[force_ind] = False
        mask_switch = np.zeros(n_steps, dtype=np.bool)
        mask_switch[force_ind] = True
        assert np.all(mask_lobes ^ mask_switch)
        return mask_lobes, mask_switch
