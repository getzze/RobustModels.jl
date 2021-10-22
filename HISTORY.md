v0.4.3 (2021-10-22)
-------------------
* Add dependencies compat versions
* Register package

v0.4.2 (2021-10-22)
-------------------
* Minimal compatibility set to julia 1.3 (because of Tulip.jl>=0.8)

v0.4.1 (2021-09-19)
-------------------
* Correctly handle multidimensional arrays with univariate robust functions.
* Correct code formatting.

v0.4.0 (2021-09-17)
-------------------
* Drop the heavy `JuMP` dependency and use `Tulip` with the unstable internal API instead.
* Add univariate robust functions: `mean`, `std`, `var`, `sem`, `mean_and_std`, `mean_and_var` and `mean_and_sem`.
* Small bug corrections.

v0.3.0 (2021-03-22)
-------------------
* BREAKING: Implement the loss functions as subclasses of `LossFunction` and estimators as subclasses of `AbstractEstimator`.
The `kind` keyword argument is not used anymore, instead use `rlm(form, data, MMEstimator{TukeyLoss}(); initial_scale=:L1)`
* Implement Robust Ridge regression by using the keyword argument `ridgeλ` (and `ridgeG` and `βprior` for more general penalty).
* Add documentation.

v0.2.0 (2020-06-04)
-------------------
* τ-Estimator
* New estimator function: optimal Yohai-Zamar estimator
* Resampling algorithm to find the global minimum of S- and τ-estimators.

v0.1.0 (2020-05-13)
-------------------
First public release:
* M-Estimator
* S-Estimator
* MM-Estimator
* M-Quantile (Expectile, etc...)
* Quantile regression with interior point
