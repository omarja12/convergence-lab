# convergence-lab

The Black–Scholes PDE solved four ways in C++20 — lattices, finite differences,
finite elements and Monte Carlo — with every scheme measured against the closed
form, and **every convergence rate verified rather than asserted**.

The name is the point. Getting close to the analytic price is easy and proves
little. What this repository tests is whether each scheme converges at the rate
its theory promises — and it documents one well-known case where the textbook
rate is simply not achieved.

```
$ ./build/convergence_tests
...
18/18 checks passed
```

---

## Why convergence rates, not prices

Any of these methods will land near the analytic price. That is a weak test. A
scheme with the drift term mis-signed often lands close too, if you only ever
run it at one grid size.

What distinguishes a correct implementation is the **rate**. Halve the step and
the error must fall by the factor the theory predicts. Every method here is
refined across four resolutions and the observed order is measured as a
least-squares slope of `log2(error)` against `log2(resolution)`:

| Method | Observed order | Expected |
|---|---|---|
| Binomial (Cox–Ross–Rubinstein) | 1.000 | 1 |
| Implicit Euler, in time | 0.996 | 1 |
| Crank–Nicolson *without* start-up | **1.202** | 2 ✗ |
| Crank–Nicolson *with* Rannacher | **1.898** | 2 ✓ |
| Finite differences, in space | 2.003 | 2 |
| P1 finite elements, in mesh size | 2.000 | 2 |
| Monte Carlo standard error | 0.502 | 0.5 |

Those two Crank–Nicolson rows are the reason this repository exists.

---

## The Crank–Nicolson trap

Crank–Nicolson is second-order accurate **for smooth data**. A vanilla payoff is
not smooth: it has a kink at the strike. That kink excites high-frequency modes,
and Crank–Nicolson's amplification factor tends to −1 as frequency rises — so
those modes are not damped, they merely alternate sign. The result is
oscillation near the strike and an observed order closer to 1 than 2.

Textbook order: 2. Measured order here without a fix: **1.202**.

The fix is Rannacher start-up: run the first couple of steps fully implicit, as
half-steps, before handing over to Crank–Nicolson. Implicit Euler damps those
modes completely, so once they are gone the rest of the run is second order as
advertised. Measured with it: **1.898**.

It is enabled by default and can be switched off to observe the degradation:

```cpp
FdConfig cfg;
cfg.scheme = FdScheme::CrankNicolson;
cfg.rannacher_steps = 0;   // 2 by default
```

Note this is visible at *coarse* time steps. At 800 time steps the oscillation
has long since been damped by sheer step count and the spatial error dominates,
which is why the demo table shows the two variants agreeing there. The
convergence test deliberately refines time while holding space fine, so the
effect is not masked.

---

## Quickstart

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/convergence_tests      # 18 checks
./build/convergence_demo       # prices one contract every way, with errors
```

No CMake? There is no dependency beyond a C++20 compiler:

```bash
./build.sh               # g++ or clang++
build.bat                # MSVC; finds vcvars64.bat itself
```

Header-only. To use it in something else, add `include/` to your path.

```cpp
#include "convergence/option.hpp"
#include "convergence/finite_difference.hpp"

using namespace convergence;
Option  opt{100.0, 1.0, OptionType::Call, Exercise::European};
Market  mkt{100.0, 0.05, 0.20, 0.0};

FdConfig cfg;                       // Crank-Nicolson + Rannacher by default
auto r = finite_difference(opt, mkt, cfg);
```

---

## What is implemented

**Lattices** — `include/convergence/lattice.hpp`
Binomial in three parameterisations (Cox–Ross–Rubinstein, Jarrow–Rudd, Tian) and
the Boyle trinomial. American exercise by taking the payoff maximum at every
node. The tree refuses to price when the time step drives the risk-neutral
probability outside [0,1], rather than returning an arbitrageable number.

**Finite differences** — `include/convergence/finite_difference.hpp`
The whole θ-family: explicit (θ=0), Crank–Nicolson (θ=½), implicit (θ=1), with
Rannacher start-up. Tridiagonal solves by the Thomas algorithm. The explicit
scheme reports its own stability ratio, which must not exceed ½.

**Finite elements** — `include/convergence/finite_element.hpp`
Galerkin with P1 hat functions, both consistent and lumped mass matrices, on the
same θ time-stepping. Lumping the mass matrix reproduces the finite-difference
scheme to 2e-6 — an equivalence the test suite demonstrates rather than
mentions.

**Monte Carlo** — `include/convergence/monte_carlo.hpp`
Exact terminal sampling, Euler–Maruyama and Milstein, with antithetic variates
and a control variate on the terminal spot. Measured variance reduction: **7.5×**
tighter standard error at the same path count. Every result carries a standard
error and a 95% interval; a Monte Carlo price quoted without one is not a
result.

---

## Design notes worth the words

**Everything is solved in log-spot.** In `S` the PDE coefficients carry `S` and
`S²`, so truncation error varies across the grid and the matrix must be rebuilt
as the grid changes. Substituting `x = ln S` makes every coefficient constant:
one tridiagonal matrix serves every time step and the scheme is uniformly
accurate.

**The control-variate beta is estimated on a pilot run**, then held fixed.
Estimating it on the same paths it corrects biases the result.

**Monte Carlo refuses American exercise.** Forward simulation cannot value an
optimal stopping problem; that needs Longstaff–Schwartz regression. It throws
rather than returning a number that looks like a price.

**American finite differences use explicit projection**, not a full
linear-complementarity solve, so they are first order in time near the free
boundary even under Crank–Nicolson. Stated because it is a real limitation, not
hidden because it is inconvenient.

---

## Layout

```
include/convergence/option.hpp             contract, market, closed form and Greeks
include/convergence/lattice.hpp            binomial x3, trinomial
include/convergence/finite_difference.hpp  theta-family, Thomas solver, Rannacher
include/convergence/finite_element.hpp     Galerkin P1, consistent and lumped mass
include/convergence/monte_carlo.hpp        exact / Euler / Milstein, variance reduction
src/main.cpp                               comparison table
tests/test_convergence.cpp                 18 checks, orders and edge cases
```

## References

- Black, F. and Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
- Cox, J., Ross, S. and Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.*
- Boyle, P. (1986). *Option Valuation Using a Three-Jump Process.*
- Rannacher, R. (1984). *Finite element solution of diffusion problems with irregular data.*
- Giles, M. and Carter, R. (2006). *Convergence analysis of Crank–Nicolson and Rannacher time-marching.*

## Licence

MIT.
