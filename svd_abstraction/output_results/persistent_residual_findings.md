# Persistent Residual `Ae = r` Findings

Setup:
- Fixed linear residual problem
  - `r = b - A x_odometry`
  - solve `A e = r`
  - recovered state is `x = x_odometry + e`
- Base graph is the residual graph itself
- Coarse solve is exact
- Hidden state is persistent across multigrid cycles

Key metrics:
- Relative state error:
  - `||x - x*|| / ||x*||`
- Fixed residual norm:
  - `||r - A e||`
- Pre residual ratio inside one cycle:
  - `||r - A e_pre_end|| / ||r - A e_cycle_start||`

## Main comparison

| experiment | inner sweeps | final relerr | final fixed residual norm |
|---|---:|---:|---:|
| `single_sync_1000` | `1000` | `2.8996e-02` | `3.3289e-02` |
| `single_sync_5000` | `5000` | `2.3388e-02` | `2.0185e-02` |
| `fixed_k_2` | `200` | `1.0300e-03` | `1.8324e+01` |
| `fixed_k_10` | `880` | diverged | diverged |
| `fixed_k_50` | `5000` | `1.0674e-08` | `5.6556e-08` |
| `adaptive_ratio_0.5` | `6119` mean total | `1.0657e-08` | `8.1348e-08` |
| `adaptive_ratio_0.1` | `18810` mean total | `1.0648e-08` | `1.1377e-07` |

Interpretation:
- Persistent-state on fixed `Ae=r` **does work**.
- `K=50` is the first clearly strong stable regime.
- Adaptive `ratio≈0.5` also works very well, but is slightly more expensive than fixed `K=50`.
- Adaptive `ratio≈0.1` is too strict and eventually drives `K` to the cap.

## Threshold behavior (`K = 1,2,5,10,20,50`)

Cycle-2 probes are the most informative.

| K | cycle2 pre ratio | cycle2 lam delta | cycle2 e step | long-run outcome |
|---|---:|---:|---:|---|
| `1` | `4.2239e-01` | `1.0000e+00` | `2.5643e+01` | diverges badly |
| `2` | `9.8134e-01` | `3.3333e-01` | `3.1477e+01` | unstable/drifty |
| `5` | `2.2716e+00` | `1.1111e-01` | `3.6848e+01` | unstable |
| `10` | `1.9663e+00` | `2.2040e-01` | `3.7644e+01` | diverges |
| `20` | `6.7738e-01` | `1.4840e-02` | `1.2840e+01` | still unstable |
| `50` | `7.8839e-02` | `2.1822e-06` | `6.3859e-01` | stable to `1e-8` |

Interpretation:
- For persistent residual multigrid, the useful threshold is not “current geometry looks good.”
- The useful threshold is: after pre-smoothing, the hidden state has actually left the bad transient regime.
- In this problem, that first happens around `K ≈ 50`.

## Practical conclusion

For the fixed residual problem `Ae = r`, the best current baseline is:

- persistent-state residual multigrid
- exact coarse solve
- `pre50 + post0`

This is the correct linear prototype to carry into the fixed-chart inner solve for future `SE(2)` work.
