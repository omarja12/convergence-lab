// Finite differences on the Black-Scholes PDE.
//
// The equation is solved in log-spot, x = ln(S), and in time-to-maturity,
// tau = T - t:
//
//     dV/dtau = 0.5*sig^2 * d2V/dx2 + (r - q - 0.5*sig^2) * dV/dx - r*V
//
// The transformation is not cosmetic. In S the coefficients carry S and S^2,
// so the truncation error varies across the grid and the matrix must be rebuilt
// whenever the grid changes. In x every coefficient is constant, one tridiagonal
// matrix serves every time step, and the scheme is uniformly accurate.
//
// One theta parameter covers the whole family:
//   theta = 0    explicit          O(dtau) + O(dx^2), conditionally stable
//   theta = 1/2  Crank-Nicolson    O(dtau^2) + O(dx^2), unconditionally stable
//   theta = 1    implicit          O(dtau) + O(dx^2), unconditionally stable

#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "option.hpp"

namespace convergence {

enum class FdScheme { Explicit, Implicit, CrankNicolson };

[[nodiscard]] inline double theta_of(FdScheme s) noexcept {
    switch (s) {
        case FdScheme::Explicit: return 0.0;
        case FdScheme::CrankNicolson: return 0.5;
        case FdScheme::Implicit: return 1.0;
    }
    return 1.0;
}

struct FdConfig {
    int space_steps{400};   // interior + boundary nodes = space_steps + 1
    int time_steps{400};
    double num_stdev{6.0};  // half-width of the log-spot grid, in std deviations
    FdScheme scheme{FdScheme::CrankNicolson};

    // Rannacher start-up: run the first few steps fully implicit, as half-steps,
    // before handing over to Crank-Nicolson.
    //
    // This is not a refinement, it is a correctness fix. Crank-Nicolson is
    // second order only for smooth data. A vanilla payoff has a kink at the
    // strike, which excites high-frequency modes, and CN's amplification factor
    // tends to -1 as the frequency rises: those modes are not damped, they
    // merely alternate sign. The result is oscillation near the strike and an
    // observed order closer to 1 than 2. Implicit Euler damps them completely,
    // so a short implicit start-up kills the modes and the rest of the run is
    // second order as advertised.
    //
    // Set to 0 to disable and observe the degradation directly.
    int rannacher_steps{2};
};

struct FdResult {
    double price{};
    double stability_ratio{};  // explicit scheme only; must be <= 0.5 to be stable
    bool stability_ok{true};
};

// Thomas algorithm: O(n) solve of a tridiagonal system.
//
// No pivoting. That is safe here and nowhere in general: the theta-scheme
// matrices below are diagonally dominant for theta > 0, which is exactly the
// condition under which the unpivoted elimination is stable.
inline void solve_tridiagonal(const std::vector<double>& lower,
                              const std::vector<double>& diag,
                              const std::vector<double>& upper,
                              std::vector<double>& rhs,
                              std::vector<double>& scratch) {
    const std::size_t n = diag.size();
    if (n == 0) return;
    scratch.resize(n);

    double beta = diag[0];
    if (beta == 0.0) throw std::runtime_error("tridiagonal: zero pivot");
    rhs[0] /= beta;

    for (std::size_t i = 1; i < n; ++i) {
        scratch[i] = upper[i - 1] / beta;
        beta = diag[i] - lower[i] * scratch[i];
        if (beta == 0.0) throw std::runtime_error("tridiagonal: zero pivot");
        rhs[i] = (rhs[i] - lower[i] * rhs[i - 1]) / beta;
    }
    for (std::size_t i = n - 1; i-- > 0;) {
        rhs[i] -= scratch[i + 1] * rhs[i + 1];
    }
}

[[nodiscard]] inline FdResult finite_difference(const Option& opt, const Market& mkt,
                                                const FdConfig& cfg) {
    mkt.validate();
    if (cfg.space_steps < 3 || cfg.time_steps < 1) {
        throw std::invalid_argument("finite_difference: grid too coarse");
    }

    const double sig = mkt.volatility, r = mkt.rate, q = mkt.dividend, T = opt.maturity;
    const double theta = theta_of(cfg.scheme);

    // Grid in log-spot, centred on the forward-adjusted initial log-spot so the
    // region of interest is not pushed to one edge for long maturities.
    const double x0 = std::log(mkt.spot);
    const double half_width = cfg.num_stdev * sig * std::sqrt(T);
    const double x_min = x0 - half_width, x_max = x0 + half_width;

    const auto n = static_cast<std::size_t>(cfg.space_steps);
    const double dx = (x_max - x_min) / static_cast<double>(n);
    const double dt = T / cfg.time_steps;

    const double a = 0.5 * sig * sig;          // diffusion
    const double b = r - q - 0.5 * sig * sig;  // convection

    // Operator coefficients: L V_i = alpha*V_{i-1} + beta*V_i + gamma*V_{i+1}
    const double alpha = a / (dx * dx) - b / (2.0 * dx);
    const double beta_c = -2.0 * a / (dx * dx) - r;
    const double gamma = a / (dx * dx) + b / (2.0 * dx);

    FdResult out;
    out.stability_ratio = a * dt / (dx * dx);
    // The explicit scheme is only conditionally stable. Report it rather than
    // letting the caller discover it from a price that has exploded.
    out.stability_ok = (cfg.scheme != FdScheme::Explicit) || (out.stability_ratio <= 0.5);

    std::vector<double> x(n + 1), v(n + 1);
    for (std::size_t i = 0; i <= n; ++i) {
        x[i] = x_min + static_cast<double>(i) * dx;
        v[i] = opt.payoff(std::exp(x[i]));
    }

    // Interior system has n-1 unknowns; the two boundary nodes are imposed.
    const std::size_t m = n - 1;
    std::vector<double> lower(m), diag(m), upper(m), rhs(m), scratch(m);

    const double s_lo = std::exp(x_min), s_hi = std::exp(x_max);

    // One theta-step of length h, advancing to time-to-maturity tau_end.
    // Taking dt and theta as arguments is what lets the Rannacher start-up run
    // fully implicit half-steps before the main loop takes over.
    auto advance = [&](double h, double th) {
        for (std::size_t k = 0; k < m; ++k) {
            lower[k] = -th * h * alpha;
            diag[k] = 1.0 - th * h * beta_c;
            upper[k] = -th * h * gamma;
        }
        return [&, h, th](double tau_end) {
            // Dirichlet boundaries from the known asymptotics of the contract.
            // Deep out-of-the-money the option is worthless; deep in-the-money a
            // call is the discounted forward and a put is worth the discounted
            // strike less the spot.
            double v_lo, v_hi;
            if (opt.type == OptionType::Call) {
                v_lo = 0.0;
                v_hi = s_hi * std::exp(-q * tau_end) - opt.strike * std::exp(-r * tau_end);
            } else {
                v_lo = opt.strike * std::exp(-r * tau_end) - s_lo * std::exp(-q * tau_end);
                v_hi = 0.0;
            }
            if (opt.exercise == Exercise::American) {
                v_lo = std::fmax(v_lo, opt.payoff(s_lo));
                v_hi = std::fmax(v_hi, opt.payoff(s_hi));
            }

            // Explicit part: (I + (1-theta)*h*L) applied to the current layer.
            const double w = (1.0 - th) * h;
            for (std::size_t k = 0; k < m; ++k) {
                const std::size_t i = k + 1;
                rhs[k] = v[i] + w * (alpha * v[i - 1] + beta_c * v[i] + gamma * v[i + 1]);
            }
            // Boundary contributions of the implicit part move to the right side.
            rhs[0] += th * h * alpha * v_lo;
            rhs[m - 1] += th * h * gamma * v_hi;

            if (th != 0.0) {
                solve_tridiagonal(lower, diag, upper, rhs, scratch);
            }
            for (std::size_t k = 0; k < m; ++k) v[k + 1] = rhs[k];
            v[0] = v_lo;
            v[n] = v_hi;

            // American: project onto the obstacle after each step. This is
            // explicit projection, not a full linear-complementarity solve, so
            // it is first order in time near the free boundary even under
            // Crank-Nicolson.
            if (opt.exercise == Exercise::American) {
                for (std::size_t i = 0; i <= n; ++i) {
                    v[i] = std::fmax(v[i], opt.payoff(std::exp(x[i])));
                }
            }
        };
    };

    // Rannacher start-up applies only where it is needed: a scheme that already
    // damps (implicit) gains nothing, and the explicit scheme is a different
    // stability problem entirely.
    const int rannacher = (cfg.scheme == FdScheme::CrankNicolson)
                              ? std::max(0, cfg.rannacher_steps)
                              : 0;
    const int smoothed = std::min(rannacher, cfg.time_steps);

    {
        auto half_implicit = advance(0.5 * dt, 1.0);
        for (int step = 1; step <= smoothed; ++step) {
            half_implicit((step - 0.5) * dt);
            half_implicit(step * dt);
        }
    }
    {
        auto main_step = advance(dt, theta);
        for (int step = smoothed + 1; step <= cfg.time_steps; ++step) {
            main_step(step * dt);
        }
    }

    // x0 sits exactly on a node by construction, at index n/2 when n is even.
    // Interpolate rather than assume, so an odd space_steps is still correct.
    const auto it = std::lower_bound(x.begin(), x.end(), x0);
    const auto idx = static_cast<std::size_t>(std::distance(x.begin(), it));
    if (idx == 0) {
        out.price = v[0];
    } else {
        const double t_ = (x0 - x[idx - 1]) / (x[idx] - x[idx - 1]);
        out.price = (1.0 - t_) * v[idx - 1] + t_ * v[idx];
    }
    return out;
}

}  // namespace convergence
