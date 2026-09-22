# convergence-lab

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=c%2B%2B&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![header-only](https://img.shields.io/badge/header--only-yes-success)](include/convergence)
[![dependencies](https://img.shields.io/badge/dependencies-none-success)](#dependencies)
[![tests](https://img.shields.io/badge/checks-28%2F28-brightgreen)](tests/test_convergence.cpp)
[![licence](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)

The Black–Scholes PDE solved four ways in C++20 — lattices, finite differences,
finite elements and Monte Carlo — with every scheme measured against the closed
form, and **every convergence rate verified rather than asserted**.

The name is the point. Getting close to the analytic price is easy and proves
little. What this repository tests is whether each scheme converges at the rate
its theory promises — and it documents one well-known case where the textbook
rate is simply not achieved.

The same standard applies to the Monte Carlo side, where the question is not the
rate but the constant: six variance reduction techniques, each measured against
plain sampling rather than asserted to help, and each checked for the bias it
might have introduced while it was tightening the interval.

```
$ ./build/convergence_tests
...
28/28 checks passed
```

**Contents** · [Why rates](#why-convergence-rates-not-prices) ·
[The Crank–Nicolson trap](#the-cranknicolson-trap) ·
[Variance reduction](#variance-reduction-measured) · [Quickstart](#quickstart) ·
[Dependencies](#dependencies) · [What is implemented](#what-is-implemented) ·
[Design notes](#design-notes-worth-the-words) · [Layout](#layout) ·
[References](#references)

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
| Monte Carlo standard error | 0.497 | 0.5 |

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

## Variance reduction, measured

A variance reduction is a claim about a constant, and a claim about a constant
is worth exactly as much as the measurement behind it. Every technique below is
run on the same contract, at the same path count, from the same seed, so the
only thing changing between rows is the estimator. FACTOR is how much the
standard error shrinks against plain sampling.

| Technique | `McConfig` | K = 100 | K = 160 |
|---|---|---|---|
| none | — | 1.0× | 1.0× |
| Antithetic variates | `antithetic` | 2.0× | 1.5× |
| Control variate, terminal spot | `control_variate` | 2.6× | 1.1× |
| Control variate, delta hedge | `delta_control_variate` | **9.2×** | 3.3× |
| Stratified sampling | `stratified` | **36.8×** | 4.8× |
| Importance sampling | `importance_sampling` | 0.8× | **10.3×** |
| Moment matching | `moment_matching` | 1.0× † | 1.0× † |
| Conditional MC, t\_c = T/2 | `conditional_fraction` | 1.5× | 2.7× |
| Conditional MC, t\_c = T/10 | `conditional_fraction` | 3.6× | 11.3× |
| **Every unbiased technique at once** | | **88.1×** | **19.9×** |

200k paths, exact sampling, one-year call, S = 100, r = 5%, σ = 20%. Reproduce
with `./build/convergence_demo`.

Two rows are worth more than their numbers.

**Importance sampling at 0.8× is not a bug.** The automatic shift is −d₂, which
centres the terminal spot on the strike. That is the right move for a payoff
that is almost always zero — hence 10.3× on the out-of-the-money call — and the
wrong move for one that finishes in the money half the time. A technique that
only ever helps is a technique that has not been measured honestly.

**Moment matching reports no gain because it cannot see its own.** † Rescaling
the terminal spots couples the paths, so the variance estimator, which assumes
they are independent, keeps quoting the plain figure. Measured across 400 seeds
the true spread is about 2.5× tighter than the interval it reports. It also
biases the price at O(1/N) — roughly +0.009 on a price of 10.45 at 500 paths.
Conservative rather than wrong, but no longer a measurement, which is why it is
the one technique off by default.

And a limit worth seeing: conditional Monte Carlo simulates to t\_c and closes
the remaining leg in closed form. Push t\_c to zero and there is nothing left to
simulate — the estimator *is* Black–Scholes, and the standard error is exactly
zero. The other rows are all working towards that.

Five of the six are unbiased, and the test suite does not take that on trust:
each is run over 120 seeds and its mean error checked against the spread of its
own estimator.

---

## Quickstart

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/convergence_tests      # 28 checks
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

## Dependencies

**None.** No Boost, no Eigen, no QuantLib, no test framework, no package
manager. A C++20 compiler and its standard library are the whole toolchain,
which is why `build.sh` is four lines and the library is five headers you can
drop into a project.

That is a deliberate choice, not an omission. A linear algebra library would
hide the Thomas algorithm behind a solver call, and the point here is that the
tridiagonal solve, the θ-weighting and the Rannacher start-up are visible and
auditable. Everything below is cited in full:

### C++ standard library

| Header | Used for | Where |
|---|---|---|
| [`<vector>`](https://en.cppreference.com/w/cpp/container/vector) | `std::vector<double>` for the log-spot grid, tridiagonal bands, lattice layers and path buffers — the only container in the project | all four method headers |
| [`<cmath>`](https://en.cppreference.com/w/cpp/header/cmath) | `std::exp`, `std::log`, `std::sqrt`, `std::pow` for the SDE and lattice parameterisations; `std::erfc` for the normal CDF (rather than `0.5*(1+erf(x/√2))`, for accuracy in the far tail); `std::fmax` for payoffs and American projection; `std::log2` for the observed-order slopes | `option.hpp`, `monte_carlo.hpp`, tests |
| [`<random>`](https://en.cppreference.com/w/cpp/header/random) | `std::mt19937_64` seeded explicitly for reproducible runs — the raw 64-bit stream only. Normals come from inverting the CDF, not from `std::normal_distribution`, which is not specified to produce the same numbers on two implementations; a seed here means the same paths on every platform, and stratification needs the quantile function anyway | `monte_carlo.hpp` |
| [`<algorithm>`](https://en.cppreference.com/w/cpp/header/algorithm) | `std::lower_bound` + `std::distance` to locate spot in the grid for interpolation; `std::min`/`std::max` to clamp the Rannacher step count and the pilot-run size | `finite_difference.hpp`, `finite_element.hpp`, `monte_carlo.hpp` |
| [`<stdexcept>`](https://en.cppreference.com/w/cpp/header/stdexcept) | `std::invalid_argument` and `std::runtime_error` — the tree throwing on a risk-neutral probability outside [0,1], and Monte Carlo refusing American exercise, both go through here | all method headers |
| [`<cstdint>`](https://en.cppreference.com/w/cpp/header/cstdint) | `std::int64_t` path counts and `std::uint64_t` seeds, sized rather than implementation-defined | `monte_carlo.hpp` |
| [`<cstdio>`](https://en.cppreference.com/w/cpp/header/cstdio) | `std::printf`/`std::snprintf` for the comparison table and the CSV the project page is plotted from | `src/`, `tests/`, `tools/` only |
| [`<string>`](https://en.cppreference.com/w/cpp/header/string) | `std::string`, `std::to_string` in test and demo labels | `src/`, `tests/`, `tools/` only |

The five library headers pull in the first six rows only; `<cstdio>` and
`<string>` never reach the library itself.

### Language features

C++20 is the standard, though the library stays conservative within it:
`[[nodiscard]]` on every result-returning function, `inline constexpr`
constants, `noexcept` where it holds, aggregate initialisation for the `Option`,
`Market` and config structs, and designated-initialiser-friendly defaults so
`FdConfig cfg;` is already a sensible scheme.

### Build and test

| Tool | Role | Required? |
|---|---|---|
| Any C++20 compiler (g++ 10+, clang++ 12+, MSVC 19.29+) | the entire toolchain | yes |
| [CMake](https://cmake.org/) ≥ 3.16 | `INTERFACE` target, warnings, CTest registration | no — `build.sh` / `build.bat` do the same job |
| CTest | `ctest` runs the same 28 checks | no |

Tests are hand-rolled: `tests/test_convergence.cpp` counts its own assertions
and prints `28/28 checks passed`. No GoogleTest, no Catch2, nothing to install
before you can verify the claims in the table above.

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
Exact terminal sampling, Euler–Maruyama and Milstein, under the six variance
reduction techniques tabulated [above](#variance-reduction-measured): antithetic
variates, control variates (terminal spot, and a delta-hedge martingale),
stratified sampling of the terminal Brownian increment with a Brownian-bridge
fill, importance sampling by Girsanov drift shift, moment matching, and
conditional Monte Carlo. They compose, subject to the scheme rule in the design
notes below.

Normals come from inverting the CDF over the raw 64-bit stream rather than from
`std::normal_distribution`, which is not specified to produce the same numbers
on two implementations. A seed here means the same paths on every platform, and
stratification needs the quantile function regardless.

Every result carries a standard error and a 95% interval; a Monte Carlo price
quoted without one is not a result. Under stratification the naive 1/√N formula
would be wrong, so the standard error is built from within-stratum variances
instead, and a test checks it against the spread the estimator really has.

---

## Design notes worth the words

**Everything is solved in log-spot.** In `S` the PDE coefficients carry `S` and
`S²`, so truncation error varies across the grid and the matrix must be rebuilt
as the grid changes. Substituting `x = ln S` makes every coefficient constant:
one tridiagonal matrix serves every time step and the scheme is uniformly
accurate.

**The control-variate betas are estimated on an independent pilot run** — its
own seed, and no stratification — then held fixed. Estimating them on the same
paths they correct biases the result.

**Monte Carlo refuses American exercise.** Forward simulation cannot value an
optimal stopping problem; that needs Longstaff–Schwartz regression. It throws
rather than returning a number that looks like a price.

**Monte Carlo also refuses a variance reduction that does not match its
scheme.** Four of the six are built on the exact dynamics: both control
variates take their means from them (`E[S_T]` is the forward, and
`exp(-(r-q)t) S_t` is a martingale), moment matching rescales towards that same
forward, and conditional Monte Carlo closes the remaining leg with the
Black–Scholes formula. Euler and Milstein reproduce none of it — their mean
terminal spot is `S₀(1 + (r-q)Δt)^n`, and the clamp at zero moves even that. The
combination does not merely lose accuracy, it shifts the price by a constant
that no number of paths removes, so it throws. Antithetic variates,
stratification and importance sampling are properties of the normals rather
than of the dynamics, and compose with any scheme.

**Control variate coefficients are chosen by measurement, not by argument.**
Two controls can be collinear — at a single step the delta hedge is an affine
function of the terminal spot, and under antithetic sampling on a linear scheme
it is constant — and a singular fit hands back two enormous betas that cancel
in the pilot and cancel nowhere else. Under importance sampling the same thing
happens without any correlation test noticing, because both controls are
multiplied by a heavy-tailed likelihood ratio. So the pilot is split: every
candidate set of controls, including the empty one, is fitted on the first half
and scored by the variance it actually delivers on the second. A set that only
looks good where it was fitted loses to using no control at all.

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
include/convergence/monte_carlo.hpp        exact / Euler / Milstein, six variance reductions
src/main.cpp                               comparison table
tools/convergence_study.cpp                emits docs/convergence.csv
tests/test_convergence.cpp                 28 checks, orders, bias and edge cases
docs/                                      project page, plotted from convergence.csv
```

## References

- Black, F. and Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.*
- Lamberton, D. and Lapeyre, B. (1996). *Introduction to Stochastic Calculus Applied to Finance.*
- Cox, J., Ross, S. and Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.*
- Boyle, P. (1986). *Option Valuation Using a Three-Jump Process.*
- Ciarlet, P.G. (1978). *The Finite Element Method for Elliptic Problems.*
- Achdou, Y. and Pironneau, O. (2005). *Computational Methods for Option Pricing.*
- Rannacher, R. (1984). *Finite element solution of diffusion problems with irregular data.*
- Giles, M. and Carter, R. (2006). *Convergence analysis of Crank–Nicolson and Rannacher time-marching.*
- Talay, D. and Tubaro, L. (1990). *Expansion of the global error for numerical schemes solving stochastic differential equations.*

## Licence

MIT. See [LICENSE](LICENSE).

---

<sub>**Keywords:** quantitative finance · computational finance · Black–Scholes ·
option pricing · derivatives pricing · numerical methods · numerical analysis ·
convergence analysis · order of accuracy · finite difference method ·
Crank–Nicolson · Rannacher time-marching · theta scheme · Thomas algorithm ·
finite element method · Galerkin P1 · mass lumping · binomial tree ·
Cox–Ross–Rubinstein · Jarrow–Rudd · Tian · trinomial tree · Boyle ·
Monte Carlo simulation · Euler–Maruyama · Milstein scheme · antithetic variates ·
control variate · variance reduction · PDE solver · American options ·
header-only · C++20</sub>
