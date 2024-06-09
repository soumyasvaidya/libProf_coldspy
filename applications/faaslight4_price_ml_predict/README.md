- Application Name: faaslight4_price_ml_predict
- App Repo: https://github.com/mpavlovic/serverless-machine-learning/tree/master/scikit-example

Optimization: Lazy loaded scipy.stats

### Average Initialization latency of
- Original code: 1297.4333333333334
- Optimized code: 914.1660402684565

#### Average Intialization latency reduced: 29.54%

### Average End to End latency of
- Original code: 5542.114444444444
- Optimized code: 4667.022483221475

#### Average End to End latency reduced: 15.78%

### Average Memory Utilization of
- Original code: 210.55555555555554
- Optimized code: 193.6778523489933

#### Average Memory Utilization reduced: 8.01%

## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)

```diff
diff -x *.pyc -bur --co original/package/scipy/signal/_peak_finding.py optimized/package/scipy/signal/_peak_finding.py
--- original/package/scipy/signal/_peak_finding.py	2024-05-06 20:28:37
+++ optimized/package/scipy/signal/_peak_finding.py	2024-05-06 20:48:45
@@ -5,7 +5,6 @@
 import numpy as np
 
 from scipy.signal._wavelets import _cwt, _ricker
-from scipy.stats import scoreatpercentile
 
 from ._peak_finding_utils import (
     _local_maxima_1d,
@@ -1181,6 +1180,7 @@
     # Filter based on SNR
     row_one = cwt[0, :]
     noises = np.empty_like(row_one)
+    from scipy.stats import scoreatpercentile
     for ind, val in enumerate(row_one):
         window_start = max(ind - hf_window, 0)
         window_end = min(ind + hf_window + odd, num_points)
diff -x *.pyc -bur --co original/package/scipy/stats/_bws_test.py optimized/package/scipy/stats/_bws_test.py
--- original/package/scipy/stats/_bws_test.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_bws_test.py	2024-05-07 16:35:24
@@ -1,6 +1,5 @@
 import numpy as np
 from functools import partial
-from scipy import stats
 
 
 def _bws_input_validation(x, y, alternative, method):
@@ -13,6 +12,7 @@
     if np.size(x) == 0 or np.size(y) == 0:
         raise ValueError('`x` and `y` must be of nonzero size.')
 
+    from scipy import stats
     z = stats.rankdata(np.concatenate((x, y)))
     x, y = z[:len(x)], z[len(x):]
 
@@ -165,6 +165,7 @@
     difference in performance between the two groups.
     '''
 
+    from scipy import stats
     x, y, alternative, method = _bws_input_validation(x, y, alternative,
                                                       method)
     bws_statistic = partial(_bws_statistic, alternative=alternative)
diff -x *.pyc -bur --co original/package/scipy/stats/_continuous_distns.py optimized/package/scipy/stats/_continuous_distns.py
--- original/package/scipy/stats/_continuous_distns.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_continuous_distns.py	2024-05-07 16:38:45
@@ -30,7 +30,8 @@
 from ._constants import (_XMIN, _LOGXMIN, _EULER, _ZETA3, _SQRT_PI,
                          _SQRT_2_OVER_PI, _LOG_SQRT_2_OVER_PI)
 from ._censored_data import CensoredData
-import scipy.stats._boost as _boost
+# import scipy.stats._boost as _boost
+
 from scipy.optimize import root_scalar
 from scipy.stats._warnings_errors import FitError
 import scipy.stats as stats
@@ -678,6 +679,8 @@
         # beta.pdf(x, a, b) = ------------------------------------
         #                              gamma(a)*gamma(b)
         with np.errstate(over='ignore'):
+            import scipy.stats._boost as _boost
+
             return _boost._beta_pdf(x, a, b)
 
     def _logpdf(self, x, a, b):
@@ -686,20 +689,30 @@
         return lPx
 
     def _cdf(self, x, a, b):
+        import scipy.stats._boost as _boost
+
         return _boost._beta_cdf(x, a, b)
 
     def _sf(self, x, a, b):
+        import scipy.stats._boost as _boost
+
         return _boost._beta_sf(x, a, b)
 
     def _isf(self, x, a, b):
         with np.errstate(over='ignore'):  # see gh-17432
+            import scipy.stats._boost as _boost
+
             return _boost._beta_isf(x, a, b)
 
     def _ppf(self, q, a, b):
         with np.errstate(over='ignore'):  # see gh-17432
+            import scipy.stats._boost as _boost
+
             return _boost._beta_ppf(q, a, b)
 
     def _stats(self, a, b):
+        import scipy.stats._boost as _boost
+
         return (
             _boost._beta_mean(a, b),
             _boost._beta_variance(a, b),
@@ -4787,6 +4800,8 @@
         return np.exp(self._logcdf(x, mu))
 
     def _ppf(self, x, mu):
+        import scipy.stats._boost as _boost
+
         with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
             x, mu = np.broadcast_arrays(x, mu)
             ppf = _boost._invgauss_ppf(x, mu, 1)
@@ -4797,6 +4812,8 @@
         return ppf
 
     def _isf(self, x, mu):
+        import scipy.stats._boost as _boost
+
         with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
             x, mu = np.broadcast_arrays(x, mu)
             isf = _boost._invgauss_isf(x, mu, 1)
@@ -7437,36 +7454,48 @@
                           f2=lambda x, df, _: chi2._logpdf(x, df))
 
     def _pdf(self, x, df, nc):
+        import scipy.stats._boost as _boost
+
         cond = np.ones_like(x, dtype=bool) & (nc != 0)
         with np.errstate(over='ignore'):  # see gh-17432
             return _lazywhere(cond, (x, df, nc), f=_boost._ncx2_pdf,
                               f2=lambda x, df, _: chi2._pdf(x, df))
 
     def _cdf(self, x, df, nc):
+        import scipy.stats._boost as _boost
+
         cond = np.ones_like(x, dtype=bool) & (nc != 0)
         with np.errstate(over='ignore'):  # see gh-17432
             return _lazywhere(cond, (x, df, nc), f=_boost._ncx2_cdf,
                               f2=lambda x, df, _: chi2._cdf(x, df))
 
     def _ppf(self, q, df, nc):
+        import scipy.stats._boost as _boost
+
         cond = np.ones_like(q, dtype=bool) & (nc != 0)
         with np.errstate(over='ignore'):  # see gh-17432
             return _lazywhere(cond, (q, df, nc), f=_boost._ncx2_ppf,
                               f2=lambda x, df, _: chi2._ppf(x, df))
 
     def _sf(self, x, df, nc):
+        import scipy.stats._boost as _boost
+
         cond = np.ones_like(x, dtype=bool) & (nc != 0)
         with np.errstate(over='ignore'):  # see gh-17432
             return _lazywhere(cond, (x, df, nc), f=_boost._ncx2_sf,
                               f2=lambda x, df, _: chi2._sf(x, df))
 
     def _isf(self, x, df, nc):
+        import scipy.stats._boost as _boost
+
         cond = np.ones_like(x, dtype=bool) & (nc != 0)
         with np.errstate(over='ignore'):  # see gh-17432
             return _lazywhere(cond, (x, df, nc), f=_boost._ncx2_isf,
                               f2=lambda x, df, _: chi2._isf(x, df))
 
     def _stats(self, df, nc):
+        import scipy.stats._boost as _boost
+
         return (
             _boost._ncx2_mean(df, nc),
             _boost._ncx2_variance(df, nc),
@@ -7538,20 +7567,30 @@
         #             gamma(df1/2)*gamma(1+df2/2) *
         #             L^{v1/2-1}^{v2/2}(-nc*v1*x/(2*(v1*x+v2))) /
         #             (B(v1/2, v2/2) * gamma((v1+v2)/2))
+        import scipy.stats._boost as _boost
+
         return _boost._ncf_pdf(x, dfn, dfd, nc)
 
     def _cdf(self, x, dfn, dfd, nc):
+        import scipy.stats._boost as _boost
+
         return _boost._ncf_cdf(x, dfn, dfd, nc)
 
     def _ppf(self, q, dfn, dfd, nc):
         with np.errstate(over='ignore'):  # see gh-17432
+            import scipy.stats._boost as _boost
+
             return _boost._ncf_ppf(q, dfn, dfd, nc)
 
     def _sf(self, x, dfn, dfd, nc):
+        import scipy.stats._boost as _boost
+
         return _boost._ncf_sf(x, dfn, dfd, nc)
 
     def _isf(self, x, dfn, dfd, nc):
         with np.errstate(over='ignore'):  # see gh-17432
+            import scipy.stats._boost as _boost
+
             return _boost._ncf_isf(x, dfn, dfd, nc)
 
     def _munp(self, n, dfn, dfd, nc):
@@ -7562,6 +7601,8 @@
         return val
 
     def _stats(self, dfn, dfd, nc, moments='mv'):
+        import scipy.stats._boost as _boost
+
         mu = _boost._ncf_mean(dfn, dfd, nc)
         mu2 = _boost._ncf_variance(dfn, dfd, nc)
         g1 = _boost._ncf_skewness(dfn, dfd, nc) if 's' in moments else None
@@ -7753,21 +7794,31 @@
 
     def _cdf(self, x, df, nc):
         with np.errstate(over='ignore'):  # see gh-17432
+            import scipy.stats._boost as _boost
+
             return np.clip(_boost._nct_cdf(x, df, nc), 0, 1)
 
     def _ppf(self, q, df, nc):
         with np.errstate(over='ignore'):  # see gh-17432
+            import scipy.stats._boost as _boost
+
             return _boost._nct_ppf(q, df, nc)
 
     def _sf(self, x, df, nc):
+        import scipy.stats._boost as _boost
+
         with np.errstate(over='ignore'):  # see gh-17432
             return np.clip(_boost._nct_sf(x, df, nc), 0, 1)
 
     def _isf(self, x, df, nc):
         with np.errstate(over='ignore'):  # see gh-17432
+            import scipy.stats._boost as _boost
+
             return _boost._nct_isf(x, df, nc)
 
     def _stats(self, df, nc, moments='mv'):
+        import scipy.stats._boost as _boost
+
         mu = _boost._nct_mean(df, nc)
         mu2 = _boost._nct_variance(df, nc)
         g1 = _boost._nct_skewness(df, nc) if 's' in moments else None
@@ -9194,6 +9245,8 @@
         )
 
     def _cdf(self, x, a):
+        import scipy.stats._boost as _boost
+
         a = np.atleast_1d(a)
         cdf = _boost._skewnorm_cdf(x, 0, 1, a)
         # for some reason, a isn't broadcasted if some of x are invalid
@@ -9204,6 +9257,8 @@
         return np.clip(cdf, 0, 1)
 
     def _ppf(self, x, a):
+        import scipy.stats._boost as _boost
+
         return _boost._skewnorm_ppf(x, 0, 1, a)
 
     def _sf(self, x, a):
@@ -9212,6 +9267,8 @@
         return self._cdf(-x, -a)
 
     def _isf(self, x, a):
+        import scipy.stats._boost as _boost
+
         return _boost._skewnorm_isf(x, 0, 1, a)
 
     def _rvs(self, a, size=None, random_state=None):
diff -x *.pyc -bur --co original/package/scipy/stats/_covariance.py optimized/package/scipy/stats/_covariance.py
--- original/package/scipy/stats/_covariance.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_covariance.py	2024-05-07 16:39:23
@@ -2,7 +2,7 @@
 
 import numpy as np
 from scipy import linalg
-from scipy.stats import _multivariate
+# from scipy.stats import _multivariate
 
 
 __all__ = ["Covariance"]
@@ -590,6 +590,8 @@
         self._null_basis = eigenvectors * i_zero
         # This is only used for `_support_mask`, not to decide whether
         # the covariance is singular or not.
+        from scipy.stats import _multivariate
+
         self._eps = _multivariate._eigvalsh_to_eps(eigenvalues) * 10**3
         self._allow_singular = True
 
diff -x *.pyc -bur --co original/package/scipy/stats/_discrete_distns.py optimized/package/scipy/stats/_discrete_distns.py
--- original/package/scipy/stats/_discrete_distns.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_discrete_distns.py	2024-05-07 16:41:07
@@ -15,7 +15,7 @@
 
 from ._distn_infrastructure import (rv_discrete, get_distribution_names,
                                     _check_shape, _ShapeInfo)
-import scipy.stats._boost as _boost
+# import scipy.stats._boost as _boost
 from ._biasedurn import (_PyFishersNCHypergeometric,
                          _PyWalleniusNCHypergeometric,
                          _PyStochasticLib3)
@@ -73,23 +73,35 @@
 
     def _pmf(self, x, n, p):
         # binom.pmf(k) = choose(n, k) * p**k * (1-p)**(n-k)
+        import scipy.stats._boost as _boost
+
         return _boost._binom_pdf(x, n, p)
 
     def _cdf(self, x, n, p):
         k = floor(x)
+        import scipy.stats._boost as _boost
+
         return _boost._binom_cdf(k, n, p)
 
     def _sf(self, x, n, p):
         k = floor(x)
+        import scipy.stats._boost as _boost
+
         return _boost._binom_sf(k, n, p)
 
     def _isf(self, x, n, p):
+        import scipy.stats._boost as _boost
+
         return _boost._binom_isf(x, n, p)
 
     def _ppf(self, q, n, p):
+        import scipy.stats._boost as _boost
+
         return _boost._binom_ppf(q, n, p)
 
     def _stats(self, n, p, moments='mv'):
+        import scipy.stats._boost as _boost
+
         mu = _boost._binom_mean(n, p)
         var = _boost._binom_variance(n, p)
         g1, g2 = None, None
@@ -324,6 +336,8 @@
 
     def _pmf(self, x, n, p):
         # nbinom.pmf(k) = choose(k+n-1, n-1) * p**n * (1-p)**k
+        import scipy.stats._boost as _boost
+
         return _boost._nbinom_pdf(x, n, p)
 
     def _logpmf(self, x, n, p):
@@ -332,6 +346,8 @@
 
     def _cdf(self, x, n, p):
         k = floor(x)
+        import scipy.stats._boost as _boost
+
         return _boost._nbinom_cdf(k, n, p)
 
     def _logcdf(self, x, n, p):
@@ -351,17 +367,25 @@
 
     def _sf(self, x, n, p):
         k = floor(x)
+        import scipy.stats._boost as _boost
+
         return _boost._nbinom_sf(k, n, p)
 
     def _isf(self, x, n, p):
         with np.errstate(over='ignore'):  # see gh-17432
+            import scipy.stats._boost as _boost
+
             return _boost._nbinom_isf(x, n, p)
 
     def _ppf(self, q, n, p):
         with np.errstate(over='ignore'):  # see gh-17432
+            import scipy.stats._boost as _boost
+
             return _boost._nbinom_ppf(q, n, p)
 
     def _stats(self, n, p):
+        import scipy.stats._boost as _boost
+
         return (
             _boost._nbinom_mean(n, p),
             _boost._nbinom_variance(n, p),
@@ -637,12 +661,18 @@
         return result
 
     def _pmf(self, k, M, n, N):
+        import scipy.stats._boost as _boost
+
         return _boost._hypergeom_pdf(k, n, N, M)
 
     def _cdf(self, k, M, n, N):
+        import scipy.stats._boost as _boost
+
         return _boost._hypergeom_cdf(k, n, N, M)
 
     def _stats(self, M, n, N):
+        import scipy.stats._boost as _boost
+
         M, n, N = 1. * M, 1. * n, 1. * N
         m = M - n
 
@@ -665,6 +695,8 @@
         return np.sum(entr(vals), axis=0)
 
     def _sf(self, k, M, n, N):
+        import scipy.stats._boost as _boost
+
         return _boost._hypergeom_sf(k, n, N, M)
 
     def _logsf(self, k, M, n, N):
@@ -1560,6 +1592,8 @@
                 random_state.poisson(mu2, n))
 
     def _pmf(self, x, mu1, mu2):
+        import scipy.stats._boost as _boost
+
         with np.errstate(over='ignore'):  # see gh-17432
             px = np.where(x < 0,
                           _boost._ncx2_pdf(2*mu2, 2*(1-x), 2*mu1)*2,
@@ -1568,6 +1602,8 @@
         return px
 
     def _cdf(self, x, mu1, mu2):
+        import scipy.stats._boost as _boost
+
         x = floor(x)
         with np.errstate(over='ignore'):  # see gh-17432
             px = np.where(x < 0,
diff -x *.pyc -bur --co original/package/scipy/stats/_distn_infrastructure.py optimized/package/scipy/stats/_distn_infrastructure.py
--- original/package/scipy/stats/_distn_infrastructure.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_distn_infrastructure.py	2024-05-07 16:42:36
@@ -30,7 +30,7 @@
 
 # for scipy.stats.entropy. Attempts to import just that function or file
 # have cause import problems
-from scipy import stats
+# from scipy import stats
 
 from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
                    logical_and, log, sqrt, place, argmax, vectorize, asarray,
@@ -39,7 +39,7 @@
 import numpy as np
 from ._constants import _XMAX, _LOGXMAX
 from ._censored_data import CensoredData
-from scipy.stats._warnings_errors import FitError
+# from scipy.stats._warnings_errors import FitError
 
 # These are the docstring parts used for substitution in specific
 # distribution docstrings
@@ -2688,11 +2688,13 @@
 
         loc, scale, shapes = self._unpack_loc_scale(vals)
         if not (np.all(self._argcheck(*shapes)) and scale > 0):
+            from scipy.stats._warnings_errors import FitError
             raise FitError("Optimization converged to parameters that are "
                            "outside the range allowed by the distribution.")
 
         if method == 'mm':
             if not np.isfinite(obj):
+                from scipy.stats._warnings_errors import FitError
                 raise FitError("Optimization failed: either a data moment "
                                "or fitted distribution moment is "
                                "non-finite.")
@@ -3719,6 +3721,7 @@
 
     def _entropy(self, *args):
         if hasattr(self, 'pk'):
+            from scipy import stats
             return stats.entropy(self.pk)
         else:
             _a, _b = self._get_support(*args)
@@ -4015,6 +4018,8 @@
         return Y
 
     def _entropy(self):
+        from scipy import stats
+
         return stats.entropy(self.pk)
 
     def generic_moment(self, n):
diff -x *.pyc -bur --co original/package/scipy/stats/_fit.py optimized/package/scipy/stats/_fit.py
--- original/package/scipy/stats/_fit.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_fit.py	2024-05-07 16:53:01
@@ -1,7 +1,8 @@
 import warnings
 from collections import namedtuple
 import numpy as np
-from scipy import optimize, stats
+from scipy import optimize
+# from scipy import stats
 from scipy._lib._util import check_random_state
 
 
@@ -1143,6 +1144,7 @@
         rfd_dist = dist(*rfd_vals)
         return compare_fun(rfd_dist, data, axis=-1)
 
+    from scipy import stats
     res = stats.monte_carlo_test(data, rvs, statistic_fun, vectorized=True,
                                  n_resamples=n_mc_samples, axis=-1,
                                  alternative=alternative)
@@ -1169,6 +1171,8 @@
     guessed_shapes = [guessed_params.pop(x, None)
                       for x in shape_names if x in guessed_params]
 
+    from scipy import stats
+    _fit_funs = {stats.norm: _fit_norm}
     if all_fixed:
         def fit_fun(data):
             return [fixed_params[name] for name in fparam_names]
@@ -1210,7 +1214,7 @@
     return loc, scale
 
 
-_fit_funs = {stats.norm: _fit_norm}  # type: ignore[attr-defined]
+# _fit_funs = {stats.norm: _fit_norm}  # type: ignore[attr-defined]
 
 
 # Vectorized goodness of fit statistic functions. These accept a frozen
@@ -1271,6 +1275,7 @@
     # We can just as easily use the (theoretically) exact values. See e.g.
     # https://en.wikipedia.org/wiki/Order_statistic
     # "Order statistics sampled from a uniform distribution"
+    from scipy import stats
     m = stats.beta(k, n + 1 - k).median()
 
     # [7] Section 8 # 3
@@ -1297,6 +1302,7 @@
 def _gof_iv(dist, data, known_params, fit_params, guessed_params, statistic,
             n_mc_samples, random_state):
 
+    from scipy import stats
     if not isinstance(dist, stats.rv_continuous):
         message = ("`dist` must be a (non-frozen) instance of "
                    "`stats.rv_continuous`.")
diff -x *.pyc -bur --co original/package/scipy/stats/_hypotests.py optimized/package/scipy/stats/_hypotests.py
--- original/package/scipy/stats/_hypotests.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_hypotests.py	2024-05-07 16:53:43
@@ -4,7 +4,7 @@
 import numpy as np
 import warnings
 from itertools import combinations
-import scipy.stats
+# import scipy.stats
 from scipy.optimize import shgo
 from . import distributions
 from ._common import ConfidenceInterval
@@ -16,7 +16,7 @@
     _concordant_pairs as _P, _discordant_pairs as _Q
 )
 from ._axis_nan_policy import _axis_nan_policy_factory
-from scipy.stats import _stats_py
+# from scipy.stats import _stats_py
 
 __all__ = ['epps_singleton_2samp', 'cramervonmises', 'somersd',
            'barnard_exact', 'boschloo_exact', 'cramervonmises_2samp',
@@ -262,6 +262,7 @@
     # case the null hypothesis cannot be rejected ... [and] it is not necessary
     # to compute the p-value". [1] page 26 below eq. (3.6).
     if lmbd_hat2 <= 0:
+        from scipy.stats import _stats_py
         return _stats_py.SignificanceResult(0, 1)
 
     # The unbiased variance estimate [1] (3.2)
@@ -693,6 +694,7 @@
     with np.errstate(divide='ignore'):
         Z = (PA - QA)/(4*(S))**0.5
 
+    import scipy.stats
     p = scipy.stats._stats_py._get_pvalue(Z, distributions.norm, alternative)
 
     return d, p
@@ -864,6 +866,7 @@
     if x.ndim == 1:
         if x.size != y.size:
             raise ValueError("Rankings must be of equal length.")
+        import scipy.stats
         table = scipy.stats.contingency.crosstab(x, y)[1]
     elif x.ndim == 2:
         if np.any(x < 0):
@@ -1659,6 +1662,7 @@
     # get ranks of x and y in the pooled sample
     z = np.concatenate([xa, ya])
     # in case of ties, use midrank (see [1])
+    import scipy.stats
     r = scipy.stats.rankdata(z, method='average')
     rx = r[:nx]
     ry = r[nx:]
diff -x *.pyc -bur --co original/package/scipy/stats/_mannwhitneyu.py optimized/package/scipy/stats/_mannwhitneyu.py
--- original/package/scipy/stats/_mannwhitneyu.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_mannwhitneyu.py	2024-05-07 16:57:06
@@ -1,8 +1,8 @@
 import numpy as np
 from collections import namedtuple
 from scipy import special
-from scipy import stats
-from scipy.stats._stats_py import _rankdata
+# from scipy import stats
+# from scipy.stats._stats_py import _rankdata
 from ._axis_nan_policy import _axis_nan_policy_factory
 
 
@@ -222,6 +222,7 @@
     if axis != axis_int:
         raise ValueError('`axis` must be an integer.')
 
+    from scipy import stats
     if not isinstance(method, stats.PermutationMethod):
         methods = {"asymptotic", "exact", "auto"}
         method = method.lower()
@@ -479,6 +480,7 @@
     n1, n2 = x.shape[-1], y.shape[-1]
 
     # Follows [2]
+    from scipy.stats._stats_py import _rankdata
     ranks, t = _rankdata(xy, 'average', return_ties=True)  # method 2, step 1
     R1 = ranks[..., :n1].sum(axis=-1)                      # method 2, step 2
     U1 = R1 - n1*(n1+1)/2                                  # method 2, step 3
@@ -498,6 +500,7 @@
         p = _mwu_state.sf(U.astype(int), min(n1, n2), max(n1, n2))
     elif method == "asymptotic":
         z = _get_mwu_z(U, n1, n2, t, continuity=use_continuity)
+        from scipy import stats
         p = stats.norm.sf(z)
     else:  # `PermutationMethod` instance (already validated)
         def statistic(x, y, axis):
diff -x *.pyc -bur --co original/package/scipy/stats/_morestats.py optimized/package/scipy/stats/_morestats.py
--- original/package/scipy/stats/_morestats.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_morestats.py	2024-05-07 16:58:05
@@ -9,7 +9,10 @@
                    compress, pi, exp, ravel, count_nonzero, sin, cos,  # noqa: F401
                    arctan2, hypot)
 
-from scipy import optimize, special, interpolate, stats
+from scipy import optimize, special, interpolate
+# from scipy import optimize, special, interpolate, stats
+# from scipy import stats
+
 from scipy._lib._bunch import _make_tuple_bunch
 from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
 
@@ -2302,6 +2305,7 @@
         m, loc, scale = distributions.weibull_min.fit(y)
         m, loc, scale = _weibull_fit_check((m, loc, scale), y)
         fit_params = m, loc, scale
+        from scipy import stats
         logcdf = stats.weibull_min(*fit_params).logcdf(y)
         logsf = stats.weibull_min(*fit_params).logsf(y)
         c = 1 / m  # m and c are as used in [7]
@@ -2554,6 +2558,7 @@
         return A2kN_fun(samples, Z, Zstar, k, n, N)
 
     if method is not None:
+        from scipy import stats
         res = stats.permutation_test(samples, statistic, **method._asdict(),
                                      alternative='greater')
 
diff -x *.pyc -bur --co original/package/scipy/stats/_mstats_basic.py optimized/package/scipy/stats/_mstats_basic.py
--- original/package/scipy/stats/_mstats_basic.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_mstats_basic.py	2024-05-07 16:59:25
@@ -44,7 +44,7 @@
 from scipy._lib._util import _rename_parameter, _contains_nan
 from scipy._lib._bunch import _make_tuple_bunch
 import scipy.special as special
-import scipy.stats._stats_py
+# import scipy.stats._stats_py
 
 from ._stats_mstats_common import (
         _find_repeats,
@@ -568,6 +568,7 @@
     if df < 0:
         return (masked, masked)
 
+    import scipy.stats._stats_py
     return scipy.stats._stats_py.pearsonr(
                 ma.masked_array(x, mask=m).compressed(),
                 ma.masked_array(y, mask=m).compressed())
@@ -696,6 +697,7 @@
 
         t, prob = _ttest_finish(dof, t, alternative)
 
+        import scipy.stats._stats_py
         # For backwards compatibility, return scalars when comparing 2 columns
         if rs.shape == (2, 2):
             res = scipy.stats._stats_py.SignificanceResult(rs[1, 0],
@@ -723,6 +725,7 @@
                 prob[var1, var2] = result.pvalue
                 prob[var2, var1] = result.pvalue
 
+        import scipy.stats._stats_py
         res = scipy.stats._stats_py.SignificanceResult(rs, prob)
         res.correlation = rs
         return res
@@ -856,6 +859,7 @@
         n -= int(m.sum())
 
     if n < 2:
+        import scipy.stats._stats_py
         res = scipy.stats._stats_py.SignificanceResult(np.nan, np.nan)
         res.correlation = np.nan
         return res
@@ -1691,6 +1695,7 @@
         Corresponding p-value.
 
     """
+    import scipy.stats._stats_py
     alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)
     return scipy.stats._stats_py.ks_1samp(
@@ -1733,6 +1738,7 @@
     # but the circular dependencies between _mstats_basic and stats prevent that.
     alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)
+    import scipy.stats._stats_py
     return scipy.stats._stats_py.ks_2samp(data1, data2,
                                           alternative=alternative,
                                           method=method)
@@ -1759,6 +1765,7 @@
     tuple of (K-S statistic, probability)
 
     """
+    import scipy.stats._stats_py
     return scipy.stats._stats_py.kstest(data1, data2, args,
                                         alternative=alternative, method=method)
 
@@ -2979,6 +2986,7 @@
     alpha = ma.sqrt(2.0/(W2-1))
     y = ma.where(y == 0, 1, y)
     Z = delta*ma.log(y/alpha + ma.sqrt((y/alpha)**2+1))
+    import scipy.stats._stats_py
     pvalue = scipy.stats._stats_py._get_pvalue(Z, distributions.norm, alternative)
 
     return SkewtestResult(Z[()], pvalue[()])
@@ -3053,6 +3061,7 @@
     term2 = np.ma.where(denom > 0, ma.power((1-2.0/A)/denom, 1/3.0),
                         -ma.power(-(1-2.0/A)/denom, 1/3.0))
     Z = (term1 - term2) / np.sqrt(2/(9.0*A))
+    import scipy.stats._stats_py
     pvalue = scipy.stats._stats_py._get_pvalue(Z, distributions.norm, alternative)
 
     return KurtosistestResult(Z[()], pvalue[()])
diff -x *.pyc -bur --co original/package/scipy/stats/_mstats_extras.py optimized/package/scipy/stats/_mstats_extras.py
--- original/package/scipy/stats/_mstats_extras.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_mstats_extras.py	2024-05-07 17:03:43
@@ -22,7 +22,11 @@
 
 from . import _mstats_basic as mstats
 
-from scipy.stats.distributions import norm, beta, t, binom
+# from scipy.stats.distributions import norm, beta, t, binom
+# from scipy.stats.distributions import norm
+# from scipy.stats.distributions import beta
+# from scipy.stats.distributions import t
+# from scipy.stats.distributions import binom
 
 
 def hdquantiles(data, prob=list([.25,.5,.75]), axis=None, var=False,):
@@ -91,6 +95,7 @@
             return hd[0]
 
         v = np.arange(n+1) / float(n)
+        from scipy.stats.distributions import beta
         betacdf = beta.cdf
         for (i,p) in enumerate(prob):
             _w = betacdf(v, (n+1)*p, (n+1)*(1-p))
@@ -181,6 +186,7 @@
             hdsd.flat = np.nan
 
         vv = np.arange(n) / float(n-1)
+        from scipy.stats.distributions import beta
         betacdf = beta.cdf
 
         for (i,p) in enumerate(prob):
@@ -257,6 +263,8 @@
     tmean = trimmed.mean(axis)
     tstde = mstats.trimmed_stde(data,limits=limits,inclusive=inclusive,axis=axis)
     df = trimmed.count(axis) - 1
+    
+    from scipy.stats.distributions import t
     tppf = t.ppf(1-alpha/2.,df)
     return np.array((tmean - tppf*tstde, tmean+tppf*tstde))
 
@@ -281,6 +289,7 @@
         data = np.sort(data.compressed())
         n = data.size
         prob = (np.array(p) * n + 0.5).astype(int)
+        from scipy.stats.distributions import beta
         betacdf = beta.cdf
 
         mj = np.empty(len(prob), float64)
@@ -334,6 +343,7 @@
 
     """
     alpha = min(alpha, 1 - alpha)
+    from scipy.stats.distributions import norm
     z = norm.ppf(1 - alpha/2.)
     xq = mstats.mquantiles(data, prob, alphap=0, betap=0, axis=axis)
     smj = mjci(data, prob, axis=axis)
@@ -364,6 +374,7 @@
 
     """
     def _cihs_1D(data, alpha):
+        from scipy.stats.distributions import binom
         data = np.sort(data.compressed())
         n = len(data)
         alpha = min(alpha, 1-alpha)
@@ -441,6 +452,7 @@
        Statistics-Simulation and Computation 13.6 (1984): 751-773.
 
     """
+    from scipy.stats.distributions import norm
     (med_1, med_2) = (ma.median(group_1,axis=axis), ma.median(group_2,axis=axis))
     (std_1, std_2) = (mstats.stde_median(group_1, axis=axis),
                       mstats.stde_median(group_2, axis=axis))
diff -x *.pyc -bur --co original/package/scipy/stats/_multicomp.py optimized/package/scipy/stats/_multicomp.py
--- original/package/scipy/stats/_multicomp.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_multicomp.py	2024-05-07 17:05:17
@@ -6,11 +6,11 @@
 
 import numpy as np
 
-from scipy import stats
+# from scipy import stats
 from scipy.optimize import minimize_scalar
 from scipy.stats._common import ConfidenceInterval
-from scipy.stats._qmc import check_random_state
-from scipy.stats._stats_py import _var
+# from scipy.stats._qmc import check_random_state
+# from scipy.stats._stats_py import _var
 
 if TYPE_CHECKING:
     import numpy.typing as npt
@@ -362,6 +362,7 @@
     random_state: SeedType
 ) -> tuple[list[np.ndarray], np.ndarray, SeedType]:
     """Input validation for Dunnett's test."""
+    from scipy.stats._qmc import check_random_state
     rng = check_random_state(random_state)
 
     if alternative not in {'two-sided', 'less', 'greater'}:
@@ -426,6 +427,7 @@
     all_means = np.concatenate([[mean_control], mean_samples])
 
     # Variance estimate s^2 from [1] Eq. 1
+    from scipy.stats._stats_py import _var
     s2 = np.sum([_var(sample, mean=mean)*sample.size
                  for sample, mean in zip(all_samples, all_means)]) / df
     std = np.sqrt(s2)
@@ -447,6 +449,7 @@
     """
     statistic = statistic.reshape(-1, 1)
 
+    from scipy import stats
     mvt = stats.multivariate_t(shape=rho, df=df, seed=rng)
     if alternative == "two-sided":
         statistic = abs(statistic)
diff -x *.pyc -bur --co original/package/scipy/stats/_page_trend_test.py optimized/package/scipy/stats/_page_trend_test.py
--- original/package/scipy/stats/_page_trend_test.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_page_trend_test.py	2024-05-07 17:07:18
@@ -2,7 +2,7 @@
 import numpy as np
 import math
 from ._continuous_distns import norm
-import scipy.stats
+# import scipy.stats
 from dataclasses import dataclass
 
 
@@ -334,6 +334,7 @@
             raise ValueError("`data` is not properly ranked. Rank the data or "
                              "pass `ranked=False`.")
     else:
+        import scipy.stats
         ranks = scipy.stats.rankdata(data, axis=-1)
 
     # generate predicted ranks if not provided, ensure valid NumPy array
diff -x *.pyc -bur --co original/package/scipy/stats/_qmc.py optimized/package/scipy/stats/_qmc.py
--- original/package/scipy/stats/_qmc.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_qmc.py	2024-05-07 17:08:04
@@ -24,7 +24,7 @@
         DecimalNumber, GeneratorType, IntNumber, SeedType
     )
 
-import scipy.stats as stats
+# import scipy.stats as stats
 from scipy._lib._util import rng_integers, _rng_spawn
 from scipy.sparse.csgraph import minimum_spanning_tree
 from scipy.spatial import distance, Voronoi
@@ -2316,6 +2316,7 @@
         if self._inv_transform:
             # apply inverse transform
             # (values to close to 0/1 result in inf values)
+            import scipy.stats as stats
             return stats.norm.ppf(0.5 + (1 - 1e-10) * (samples - 0.5))  # type: ignore[attr-defined]  # noqa: E501
         else:
             # apply Box-Muller transform (note: indexes starting from 1)
diff -x *.pyc -bur --co original/package/scipy/stats/_qmvnt.py optimized/package/scipy/stats/_qmvnt.py
--- original/package/scipy/stats/_qmvnt.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_qmvnt.py	2024-05-07 17:08:25
@@ -36,7 +36,7 @@
 
 from scipy.fft import fft, ifft
 from scipy.special import gammaincinv, ndtr, ndtri
-from scipy.stats._qmc import primes_from_2_to
+# from scipy.stats._qmc import primes_from_2_to
 
 
 phi = ndtr
@@ -48,6 +48,7 @@
     """
     # NOTE: There are lots faster ways to do this, but this isn't terrible.
     factors = set()
+    from scipy.stats._qmc import primes_from_2_to
     for p in primes_from_2_to(int(np.sqrt(n)) + 1):
         while not (n % p):
             factors.add(p)
@@ -114,6 +115,7 @@
            Springer-Verlag, 2006, 371-385.
     """
     # Round down to the nearest prime number.
+    from scipy.stats._qmc import primes_from_2_to
     primes = primes_from_2_to(n_qmc_samples + 1)
     n_qmc_samples = primes[-1]
 
diff -x *.pyc -bur --co original/package/scipy/stats/_rvs_sampling.py optimized/package/scipy/stats/_rvs_sampling.py
--- original/package/scipy/stats/_rvs_sampling.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_rvs_sampling.py	2024-05-07 17:09:06
@@ -1,5 +1,5 @@
 import warnings
-from scipy.stats.sampling import RatioUniforms
+# from scipy.stats.sampling import RatioUniforms
 
 def rvs_ratio_uniforms(pdf, umax, vmin, vmax, size=1, c=0, random_state=None):
     """
@@ -51,6 +51,7 @@
                   "`scipy.stats.rvs_ratio_uniforms` namespace is deprecated "
                   "and will be removed in SciPy 1.15.0",
                   category=DeprecationWarning, stacklevel=2)
+    from scipy.stats.sampling import RatioUniforms
     gen = RatioUniforms(pdf, umax=umax, vmin=vmin, vmax=vmax,
                         c=c, random_state=random_state)
     return gen.rvs(size)
diff -x *.pyc -bur --co original/package/scipy/stats/_sampling.py optimized/package/scipy/stats/_sampling.py
--- original/package/scipy/stats/_sampling.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_sampling.py	2024-05-07 17:09:25
@@ -1,7 +1,7 @@
 import math
 import numbers
 import numpy as np
-from scipy import stats
+# from scipy import stats
 from scipy import special as sc
 from ._qmc import (check_random_state as check_random_state_qmc,
                    Halton, QMCEngine)
@@ -621,6 +621,7 @@
         ignore_shape_range=False,
         random_state=None,
     ):
+        from scipy import stats
 
         if isinstance(dist, stats.distributions.rv_frozen):
             distname = dist.dist.name
diff -x *.pyc -bur --co original/package/scipy/stats/_sensitivity_analysis.py optimized/package/scipy/stats/_sensitivity_analysis.py
--- original/package/scipy/stats/_sensitivity_analysis.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_sensitivity_analysis.py	2024-05-07 17:12:21
@@ -8,10 +8,10 @@
 
 import numpy as np
 
-from scipy.stats._common import ConfidenceInterval
-from scipy.stats._qmc import check_random_state
+# from scipy.stats._common import ConfidenceInterval
+# from scipy.stats._qmc import check_random_state
 from scipy.stats._resampling import BootstrapResult
-from scipy.stats import qmc, bootstrap
+# from scipy.stats import qmc, bootstrap
 
 
 if TYPE_CHECKING:
@@ -79,6 +79,7 @@
        :doi:`10.1016/j.cpc.2009.09.018`, 2010.
     """
     d = len(dists)
+    from scipy.stats import qmc
     A_B = qmc.Sobol(d=2*d, seed=random_state, bits=64).random(n).T
     A_B = A_B.reshape(2, d, -1)
     try:
@@ -211,6 +212,7 @@
 
         n = self._f_A.shape[1]
 
+        from scipy.stats import bootstrap, qmc
         res = bootstrap(
             [np.arange(n)], statistic=statistic, method="BCa",
             n_resamples=n_resamples,
@@ -219,6 +221,7 @@
         )
         self._bootstrap_result = res
 
+        from scipy.stats._common import ConfidenceInterval
         first_order = BootstrapResult(
             confidence_interval=ConfidenceInterval(
                 res.confidence_interval.low[0], res.confidence_interval.high[0]
diff -x *.pyc -bur --co original/package/scipy/stats/_stats_py.py optimized/package/scipy/stats/_stats_py.py
--- original/package/scipy/stats/_stats_py.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_stats_py.py	2024-05-07 17:13:53
@@ -62,16 +62,16 @@
                                _broadcast_concatenate)
 from ._binomtest import _binary_search_for_binom_tst as _binary_search
 from scipy._lib._bunch import _make_tuple_bunch
-from scipy import stats
+# from scipy import stats
 from scipy.optimize import root_scalar
 from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
 from scipy._lib._util import normalize_axis_index
 
 # In __all__ but deprecated for removal in SciPy 1.13.0
 from scipy._lib._util import float_factorial  # noqa: F401
-from scipy.stats._mstats_basic import (  # noqa: F401
-    PointbiserialrResult, Ttest_1sampResult,  Ttest_relResult
-)
+# from scipy.stats._mstats_basic import (  # noqa: F401
+#     PointbiserialrResult, Ttest_1sampResult,  Ttest_relResult
+# )
 
 
 # Functions/classes in other files should be added in `__init__.py`, not here
@@ -3939,6 +3939,7 @@
     # and reported before checking for 0 length inputs.
     if any(sample.shape[axis] == 0 for sample in samples):
         msg = 'at least one input has length 0'
+        from scipy import stats
         warnings.warn(stats.DegenerateDataWarning(msg), stacklevel=2)
         return True
 
@@ -3946,6 +3947,7 @@
     if all(sample.shape[axis] == 1 for sample in samples):
         msg = ('all input arrays have length 1.  f_oneway requires that at '
                'least one input has length greater than 1.')
+        from scipy import stats
         warnings.warn(stats.DegenerateDataWarning(msg), stacklevel=2)
         return True
 
@@ -4125,6 +4127,7 @@
     if all_const.any():
         msg = ("Each of the input arrays is constant; "
                "the F statistic is not defined or infinite")
+        from scipy import stats
         warnings.warn(stats.ConstantInputWarning(msg), stacklevel=2)
 
     # all_same_const is True if all the values in the groups along the axis=0
@@ -4281,6 +4284,7 @@
 
     if np.any([(sample == sample[0]).all() for sample in samples]):
         msg = "An input array is constant; the statistic is not defined."
+        from scipy import stats
         warnings.warn(stats.ConstantInputWarning(msg), stacklevel=2)
         return AlexanderGovernResult(np.nan, np.nan)
 
@@ -4734,6 +4738,7 @@
     if (x == x[0]).all() or (y == y[0]).all():
         msg = ("An input array is constant; the correlation coefficient "
                "is not defined.")
+        from scipy import stats
         warnings.warn(stats.ConstantInputWarning(msg), stacklevel=2)
         result = PearsonRResult(statistic=np.nan, pvalue=np.nan, n=n,
                                 alternative=alternative, x=x, y=y)
@@ -4800,6 +4805,7 @@
         # might result in large errors in r.
         msg = ("An input array is nearly constant; the computed "
                "correlation coefficient may be inaccurate.")
+        from scipy import stats
         warnings.warn(stats.NearConstantInputWarning(msg), stacklevel=2)
 
     r = np.dot(xm/normxm, ym/normym)
@@ -5390,6 +5396,7 @@
         if (a[:, 0][0] == a[:, 0]).all() or (a[:, 1][0] == a[:, 1]).all():
             # If an input is constant, the correlation coefficient
             # is not defined.
+            from scipy import stats
             warnings.warn(stats.ConstantInputWarning(warn_msg), stacklevel=2)
             res = SignificanceResult(np.nan, np.nan)
             res.correlation = np.nan
@@ -5398,6 +5405,7 @@
         if (a[0, :][0] == a[0, :]).all() or (a[1, :][0] == a[1, :]).all():
             # If an input is constant, the correlation coefficient
             # is not defined.
+            from scipy import stats
             warnings.warn(stats.ConstantInputWarning(warn_msg), stacklevel=2)
             res = SignificanceResult(np.nan, np.nan)
             res.correlation = np.nan
@@ -9741,6 +9749,7 @@
         p = self._p
         x = np.sort(self._x)
         n = len(x)
+        from scipy import stats
         bd = stats.binom(n, p)
 
         if confidence_level <= 0 or confidence_level >= 1:
@@ -10086,6 +10095,7 @@
     # "p = p* as given in the null hypothesis.... Y has the binomial "
     # "distribution with parameters n and p*."
     n = len(X)
+    from scipy import stats
     Y = stats.binom(n=n, p=p_star)
 
     # "H1: the p* population quantile is less than x*"
diff -x *.pyc -bur --co original/package/scipy/stats/_survival.py optimized/package/scipy/stats/_survival.py
--- original/package/scipy/stats/_survival.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_survival.py	2024-05-07 17:17:07
@@ -5,10 +5,12 @@
 import warnings
 
 import numpy as np
-from scipy import special, interpolate, stats
+# from scipy import special
+# from scipy import interpolate
+# from scipy import stats
 from scipy.stats._censored_data import CensoredData
-from scipy.stats._common import ConfidenceInterval
-from scipy.stats import norm  # type: ignore[attr-defined]
+# from scipy.stats._common import ConfidenceInterval
+# from scipy.stats import norm  # type: ignore[attr-defined]
 
 if TYPE_CHECKING:
     from typing import Literal
@@ -54,6 +56,7 @@
         x = np.insert(q, [0, len(q)], [-np.inf, np.inf])
         y = np.insert(p, [0, len(p)], [f0, f1])
         # `or` conditions handle the case of empty x, points
+        from scipy import interpolate
         self._f = interpolate.interp1d(x, y, kind='previous',
                                        assume_sorted=True)
 
@@ -180,6 +183,7 @@
                                             self._kind)
         high = EmpiricalDistributionFunction(self.quantiles, high, None, None,
                                              self._kind)
+        from scipy.stats._common import ConfidenceInterval
         return ConfidenceInterval(low, high)
 
     def _linear_ci(self, confidence_level):
@@ -191,6 +195,7 @@
             var = sf ** 2 * np.cumsum(d / (n * (n - d)))
 
         se = np.sqrt(var)
+        from scipy import special
         z = special.ndtri(1 / 2 + confidence_level / 2)
 
         z_se = z * se
@@ -206,6 +211,7 @@
             var = 1 / np.log(sf) ** 2 * np.cumsum(d / (n * (n - d)))
 
         se = np.sqrt(var)
+        from scipy import special
         z = special.ndtri(1 / 2 + confidence_level / 2)
 
         with np.errstate(divide='ignore'):
@@ -681,6 +687,8 @@
     statistic = (n_died_x - sum_exp_deaths_x)/np.sqrt(sum_var)
 
     # Equivalent to chi2(df=1).sf(statistic**2) when alternative='two-sided'
+    from scipy import stats
+    from scipy.stats import norm
     pvalue = stats._stats_py._get_pvalue(statistic, norm, alternative)
 
     return LogRankResult(statistic=statistic[()], pvalue=pvalue[()])
diff -x *.pyc -bur --co original/package/scipy/stats/_unuran/unuran_wrapper.pyi optimized/package/scipy/stats/_unuran/unuran_wrapper.pyi
--- original/package/scipy/stats/_unuran/unuran_wrapper.pyi	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_unuran/unuran_wrapper.pyi	2024-05-07 17:43:43
@@ -3,7 +3,7 @@
 from typing import (overload, Callable, NamedTuple, Protocol)
 import numpy.typing as npt
 from scipy._lib._util import SeedType
-import scipy.stats as stats
+# import scipy.stats as stats
 
 
 ArrayLike0D = bool | int | float | complex | str | bytes | np.generic
@@ -110,7 +110,7 @@
     def qrvs(self,
              size: None | int | tuple[int, ...] = ...,
              d: None | int = ...,
-             qmc_engine: None | stats.qmc.QMCEngine = ...) -> npt.ArrayLike: ...
+             qmc_engine: None = ...) -> npt.ArrayLike: ...
 
 
 class HINVDist(Protocol):
@@ -141,7 +141,7 @@
     def qrvs(self,
              size: None | int | tuple[int, ...] = ...,
              d: None | int = ...,
-             qmc_engine: None | stats.qmc.QMCEngine = ...) -> npt.ArrayLike: ...
+             qmc_engine: None = ...) -> npt.ArrayLike: ...
     def u_error(self, sample_size: int = ...) -> UError: ...
 
 
diff -x *.pyc -bur --co original/package/scipy/stats/_wilcoxon.py optimized/package/scipy/stats/_wilcoxon.py
--- original/package/scipy/stats/_wilcoxon.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/_wilcoxon.py	2024-05-07 17:17:52
@@ -1,7 +1,7 @@
 import warnings
 import numpy as np
 
-from scipy import stats
+# from scipy import stats
 from ._stats_py import _get_pvalue, _rankdata
 from . import _morestats
 from ._axis_nan_policy import _broadcast_arrays
@@ -96,6 +96,7 @@
     if alternative not in alternatives:
         raise ValueError(message)
 
+    from scipy import stats
     if not isinstance(method, stats.PermutationMethod):
         methods = {"auto", "approx", "exact"}
         message = (f"`method` must be one of {methods} or "
@@ -211,6 +212,7 @@
         if correction:
             sign = _correction_sign(z, alternative)
             z -= sign * 0.5 / se
+        from scipy import stats
         p = _get_pvalue(z, stats.norm, alternative)
     elif method == 'exact':
         dist = WilcoxonDistribution(count)
diff -x *.pyc -bur --co original/package/scipy/stats/mstats.py optimized/package/scipy/stats/mstats.py
--- original/package/scipy/stats/mstats.py	2024-05-06 20:28:37
+++ optimized/package/scipy/stats/mstats.py	2024-05-07 17:39:23
@@ -134,7 +134,7 @@
 from ._mstats_extras import *  # noqa: F403
 # Functions that support masked array input in stats but need to be kept in the
 # mstats namespace for backwards compatibility:
-from scipy.stats import gmean, hmean, zmap, zscore, chisquare
+# from scipy.stats import gmean, hmean, zmap, zscore, chisquare
 
 __all__ = _mstats_basic.__all__ + _mstats_extras.__all__
-__all__ += ['gmean', 'hmean', 'zmap', 'zscore', 'chisquare']
+# __all__ += ['gmean', 'hmean', 'zmap', 'zscore', 'chisquare']
Binary files original/package/sklearn/.DS_Store and optimized/package/sklearn/.DS_Store differ
diff -x *.pyc -bur --co original/package/sklearn/__init__.py optimized/package/sklearn/__init__.py
--- original/package/sklearn/__init__.py	2024-05-06 20:28:38
+++ optimized/package/sklearn/__init__.py	2024-05-07 15:39:04
@@ -84,8 +84,8 @@
         __check_build,  # noqa: F401
         _distributor_init,  # noqa: F401
     )
-    from .base import clone
-    from .utils._show_versions import show_versions
+    # from .base import clone
+    # from .utils._show_versions import show_versions
 
     __all__ = [
         "calibration",
@@ -126,11 +126,11 @@
         "impute",
         "compose",
         # Non-modules:
-        "clone",
+        # "clone",
         "get_config",
         "set_config",
         "config_context",
-        "show_versions",
+        # "show_versions",
     ]
 
     _BUILT_WITH_MESON = False
diff -x *.pyc -bur --co original/package/sklearn/_loss/link.py optimized/package/sklearn/_loss/link.py
--- original/package/sklearn/_loss/link.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/_loss/link.py	2024-05-06 20:04:06
@@ -8,7 +8,6 @@
 
 import numpy as np
 from scipy.special import expit, logit
-from scipy.stats import gmean
 
 from ..utils.extmath import softmax
 
@@ -259,6 +258,7 @@
 
     def link(self, y_pred, out=None):
         # geometric mean as reference category
+        from scipy.stats import gmean
         gm = gmean(y_pred, axis=1)
         return np.log(y_pred / gm[:, np.newaxis], out=out)
 
diff -x *.pyc -bur --co original/package/sklearn/covariance/_robust_covariance.py optimized/package/sklearn/covariance/_robust_covariance.py
--- original/package/sklearn/covariance/_robust_covariance.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/covariance/_robust_covariance.py	2024-05-06 20:07:56
@@ -13,7 +13,6 @@
 
 import numpy as np
 from scipy import linalg
-from scipy.stats import chi2
 
 from ..base import _fit_context
 from ..utils import check_array, check_random_state
@@ -809,8 +808,7 @@
             raise ValueError(
                 "The covariance matrix of the support data "
                 "is equal to 0, try to increase support_fraction"
-            )
-        correction = np.median(self.dist_) / chi2(data.shape[1]).isf(0.5)
+            )(data.shape[1]).isf(0.5)
         covariance_corrected = self.raw_covariance_ * correction
         self.dist_ /= correction
         return covariance_corrected
@@ -849,8 +847,7 @@
             Determinant Estimator, 1999, American Statistical Association
             and the American Society for Quality, TECHNOMETRICS
         """
-        n_samples, n_features = data.shape
-        mask = self.dist_ < chi2(n_features).isf(0.025)
+        n_samples, n_features = data.shape(n_features).isf(0.025)
         if self.assume_centered:
             location_reweighted = np.zeros(n_features)
         else:
diff -x *.pyc -bur --co original/package/sklearn/feature_selection/_univariate_selection.py optimized/package/sklearn/feature_selection/_univariate_selection.py
--- original/package/sklearn/feature_selection/_univariate_selection.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/feature_selection/_univariate_selection.py	2024-05-06 20:07:02
@@ -9,7 +9,7 @@
 from numbers import Integral, Real
 
 import numpy as np
-from scipy import special, stats
+from scipy import special
 from scipy.sparse import issparse
 
 from ..base import BaseEstimator, _fit_context
@@ -504,6 +504,7 @@
 
     with np.errstate(divide="ignore", invalid="ignore"):
         f_statistic = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
+        from scipy import stats
         p_values = stats.f.sf(f_statistic, 1, deg_of_freedom)
 
     if force_finite and not np.isfinite(f_statistic).all():
diff -x *.pyc -bur --co original/package/sklearn/impute/_iterative.py optimized/package/sklearn/impute/_iterative.py
--- original/package/sklearn/impute/_iterative.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/impute/_iterative.py	2024-05-06 20:08:50
@@ -4,7 +4,6 @@
 from time import time
 
 import numpy as np
-from scipy import stats
 
 from ..base import _fit_context, clone
 from ..exceptions import ConvergenceWarning
@@ -441,6 +440,7 @@
             a = (self._min_value[feat_idx] - mus) / sigmas
             b = (self._max_value[feat_idx] - mus) / sigmas
 
+            from scipy import stats
             truncated_normal = stats.truncnorm(a=a, b=b, loc=mus, scale=sigmas)
             imputed_values[inrange_mask] = truncated_normal.rvs(
                 random_state=self.random_state_
diff -x *.pyc -bur --co original/package/sklearn/inspection/_partial_dependence.py optimized/package/sklearn/inspection/_partial_dependence.py
--- original/package/sklearn/inspection/_partial_dependence.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/inspection/_partial_dependence.py	2024-05-06 20:09:16
@@ -9,7 +9,6 @@
 
 import numpy as np
 from scipy import sparse
-from scipy.stats.mstats import mquantiles
 
 from ..base import is_classifier, is_regressor
 from ..ensemble import RandomForestRegressor
@@ -116,6 +115,7 @@
             axis = uniques
         else:
             # create axis based on percentiles and grid resolution
+            from scipy.stats.mstats import mquantiles
             emp_percentiles = mquantiles(
                 _safe_indexing(X, feature, axis=1), prob=percentiles, axis=0
             )
diff -x *.pyc -bur --co original/package/sklearn/inspection/_plot/partial_dependence.py optimized/package/sklearn/inspection/_plot/partial_dependence.py
--- original/package/sklearn/inspection/_plot/partial_dependence.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/inspection/_plot/partial_dependence.py	2024-05-06 20:25:06
@@ -4,7 +4,6 @@
 
 import numpy as np
 from scipy import sparse
-from scipy.stats.mstats import mquantiles
 
 from ...base import is_regressor
 from ...utils import (
@@ -742,6 +741,7 @@
             for fx, cat in zip(fxs, cats):
                 if not cat and fx not in deciles:
                     X_col = _safe_indexing(X, fx, axis=1)
+                    from scipy.stats.mstats import mquantiles
                     deciles[fx] = mquantiles(X_col, prob=np.arange(0.1, 1.0, 0.1))
 
         display = cls(
diff -x *.pyc -bur --co original/package/sklearn/isotonic.py optimized/package/sklearn/isotonic.py
--- original/package/sklearn/isotonic.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/isotonic.py	2024-05-06 20:03:31
@@ -9,7 +9,6 @@
 
 import numpy as np
 from scipy import interpolate
-from scipy.stats import spearmanr
 
 from ._isotonic import _inplace_contiguous_isotonic_regression, _make_unique
 from .base import BaseEstimator, RegressorMixin, TransformerMixin, _fit_context
@@ -71,6 +70,7 @@
     """
 
     # Calculate Spearman rho estimate and set return accordingly.
+    from scipy.stats import spearmanr
     rho, _ = spearmanr(x, y)
     increasing_bool = rho >= 0
 
diff -x *.pyc -bur --co original/package/sklearn/metrics/_ranking.py optimized/package/sklearn/metrics/_ranking.py
--- original/package/sklearn/metrics/_ranking.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/metrics/_ranking.py	2024-05-06 20:25:05
@@ -25,7 +25,6 @@
 
 import numpy as np
 from scipy.sparse import csr_matrix, issparse
-from scipy.stats import rankdata
 
 from ..exceptions import UndefinedMetricWarning
 from ..preprocessing import label_binarize
@@ -1237,6 +1236,7 @@
             aux = 1.0
         else:
             scores_i = y_score[i]
+            from scipy.stats import rankdata
             rank = rankdata(scores_i, "max")[relevant]
             L = rankdata(scores_i[relevant], "max")
             aux = (L / rank).mean()
diff -x *.pyc -bur --co original/package/sklearn/model_selection/_search.py optimized/package/sklearn/model_selection/_search.py
--- original/package/sklearn/model_selection/_search.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/model_selection/_search.py	2024-05-06 20:45:16
@@ -22,7 +22,6 @@
 
 import numpy as np
 from numpy.ma import MaskedArray
-from scipy.stats import rankdata
 
 from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
 from ..exceptions import NotFittedError
@@ -1074,6 +1073,7 @@
                 else:
                     min_array_means = np.nanmin(array_means) - 1
                     array_means = np.nan_to_num(array_means, nan=min_array_means)
+                    from scipy.stats import rankdata
                     rank_result = rankdata(-array_means, method="min").astype(
                         np.int32, copy=False
                     )
diff -x *.pyc -bur --co original/package/sklearn/preprocessing/_data.py optimized/package/sklearn/preprocessing/_data.py
--- original/package/sklearn/preprocessing/_data.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/preprocessing/_data.py	2024-05-06 20:45:53
@@ -12,7 +12,8 @@
 from numbers import Integral, Real
 
 import numpy as np
-from scipy import optimize, sparse, stats
+from scipy import optimize, sparse
+
 from scipy.special import boxcox
 
 from ..base import (
diff -x *.pyc -bur --co original/package/sklearn/utils/estimator_checks.py optimized/package/sklearn/utils/estimator_checks.py
--- original/package/sklearn/utils/estimator_checks.py	2024-05-06 20:28:38
+++ optimized/package/sklearn/utils/estimator_checks.py	2024-05-06 20:45:52
@@ -15,7 +15,6 @@
 import joblib
 import numpy as np
 from scipy import sparse
-from scipy.stats import rankdata
 
 from .. import config_context
 from ..base import (
@@ -2194,6 +2193,7 @@
         for i in range(n_classes):
             y_proba = estimator.predict_proba(X)[:, i]
             y_decision = estimator.decision_function(X)
+            from scipy.stats import rankdata
             assert_array_equal(rankdata(y_proba), rankdata(y_decision[:, i]))
 
 
diff -x *.pyc -bur --co original/package/sklearn/utils/fixes.py optimized/package/sklearn/utils/fixes.py
--- original/package/sklearn/utils/fixes.py	2024-05-07 19:04:16
+++ optimized/package/sklearn/utils/fixes.py	2024-05-06 20:01:34
@@ -14,7 +14,6 @@
 import numpy as np
 import scipy
 import scipy.sparse.linalg
-import scipy.stats
 import threadpoolctl
 
 import sklearn
@@ -120,6 +119,7 @@
 
 # TODO: Remove when SciPy 1.11 is the minimum supported version
 def _mode(a, axis=0):
+    import scipy.stats
     if sp_version >= parse_version("1.9.0"):
         mode = scipy.stats.mode(a, axis=axis, keepdims=True)
         if sp_version >= parse_version("1.10.999"):

```