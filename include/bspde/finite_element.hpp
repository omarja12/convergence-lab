// Galerkin finite elements on the log-transformed Black-Scholes PDE.
//
// Same equation as finite_difference.hpp, in weak form. Multiply by a test
// function and integrate the second derivative by parts:
//
//   (dV/dtau, phi) = -a (V', phi') + b (V', phi) - r (V, phi)
//
// with a = sig^2/2 and b = r - q - sig^2/2. Writing V in the P1 hat basis gives
//
//   M dV/dtau = A V,     A = -a K + b B - r M
//
// and M, K, B are all tridiagonal, so the theta time-stepping reuses the same
// Thomas solver as the finite-difference module.
//
// The difference from finite differences is the mass matrix. FD implicitly uses
// the identity; Galerkin uses M, which couples neighbouring nodes. Lumping M to
// its row sums recovers something close to the FD scheme -- that equivalence is
// worth being able to demonstrate, so both are offered.

#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "finite_difference.hpp"  // solve_tridiagonal, FdScheme, theta_of
#include "option.hpp"

namespace bspde {

enum class MassMatrix {
    Consistent,  // true Galerkin: M = h/6 * tridiag(1, 4, 1)
    Lumped,      // row-summed to diagonal h; cheaper, and close to the FD scheme
};

struct FemConfig {
    int elements{400};
    int time_steps{400};
    double num_stdev{6.0};
    FdScheme scheme{FdScheme::CrankNicolson};
    MassMatrix mass{MassMatrix::Consistent};
};

struct FemResult {
    double price{};
    int elements{};
};

[[nodiscard]] inline FemResult finite_element(const Option& opt, const Market& mkt,
                                              const FemConfig& cfg) {
    mkt.validate();
    if (cfg.elements < 3 || cfg.time_steps < 1) {
        throw std::invalid_argument("finite_element: mesh too coarse");
    }

    const double sig = mkt.volatility, r = mkt.rate, q = mkt.dividend, T = opt.maturity;
    const double theta = theta_of(cfg.scheme);

    const double x0 = std::log(mkt.spot);
    const double half_width = cfg.num_stdev * sig * std::sqrt(T);
    const double x_min = x0 - half_width, x_max = x0 + half_width;

    const auto n = static_cast<std::size_t>(cfg.elements);
    const double h = (x_max - x_min) / static_cast<double>(n);
    const double dt = T / cfg.time_steps;

    const double a = 0.5 * sig * sig;
    const double b = r - q - 0.5 * sig * sig;

    // Element matrices assembled for a uniform mesh, per interior row.
    //   M = h/6 * (1, 4, 1)    consistent mass
    //   M = h   * (0, 1, 0)    lumped mass
    //   K = 1/h * (-1, 2, -1)  stiffness
    //   B =       (-1/2, 0, 1/2)  convection, B_ij = integral(phi_j' phi_i)
    const bool lumped = (cfg.mass == MassMatrix::Lumped);
    const double m_off = lumped ? 0.0 : h / 6.0;
    const double m_dia = lumped ? h : 4.0 * h / 6.0;

    // A = -a*K + b*B - r*M, row-constant for a uniform mesh.
    const double a_low = -a * (-1.0 / h) + b * (-0.5) - r * m_off;
    const double a_dia = -a * (2.0 / h) + b * (0.0) - r * m_dia;
    const double a_upp = -a * (-1.0 / h) + b * (0.5) - r * m_off;

    std::vector<double> x(n + 1), v(n + 1);
    for (std::size_t i = 0; i <= n; ++i) {
        x[i] = x_min + static_cast<double>(i) * h;
        v[i] = opt.payoff(std::exp(x[i]));
    }

    const std::size_t m = n - 1;  // interior unknowns
    std::vector<double> lower(m), diag(m), upper(m), rhs(m), scratch(m);
    for (std::size_t k = 0; k < m; ++k) {
        lower[k] = m_off - theta * dt * a_low;
        diag[k] = m_dia - theta * dt * a_dia;
        upper[k] = m_off - theta * dt * a_upp;
    }

    for (int step = 1; step <= cfg.time_steps; ++step) {
        const double tau = step * dt;
        const double s_lo = std::exp(x_min), s_hi = std::exp(x_max);
        double v_lo, v_hi;
        if (opt.type == OptionType::Call) {
            v_lo = 0.0;
            v_hi = s_hi * std::exp(-q * tau) - opt.strike * std::exp(-r * tau);
        } else {
            v_lo = opt.strike * std::exp(-r * tau) - s_lo * std::exp(-q * tau);
            v_hi = 0.0;
        }
        if (opt.exercise == Exercise::American) {
            v_lo = std::fmax(v_lo, opt.payoff(s_lo));
            v_hi = std::fmax(v_hi, opt.payoff(s_hi));
        }

        // Right side: (M + (1-theta)*dt*A) V^n, boundary terms moved across.
        const double w = (1.0 - theta) * dt;
        for (std::size_t k = 0; k < m; ++k) {
            const std::size_t i = k + 1;
            rhs[k] = (m_off * v[i - 1] + m_dia * v[i] + m_off * v[i + 1])
                     + w * (a_low * v[i - 1] + a_dia * v[i] + a_upp * v[i + 1]);
        }
        rhs[0] -= (m_off - theta * dt * a_low) * v_lo;
        rhs[m - 1] -= (m_off - theta * dt * a_upp) * v_hi;

        solve_tridiagonal(lower, diag, upper, rhs, scratch);
        for (std::size_t k = 0; k < m; ++k) v[k + 1] = rhs[k];
        v[0] = v_lo;
        v[n] = v_hi;

        if (opt.exercise == Exercise::American) {
            for (std::size_t i = 0; i <= n; ++i) {
                v[i] = std::fmax(v[i], opt.payoff(std::exp(x[i])));
            }
        }
    }

    const auto it = std::lower_bound(x.begin(), x.end(), x0);
    const auto idx = static_cast<std::size_t>(std::distance(x.begin(), it));
    double price;
    if (idx == 0) {
        price = v[0];
    } else {
        const double t_ = (x0 - x[idx - 1]) / (x[idx] - x[idx - 1]);
        price = (1.0 - t_) * v[idx - 1] + t_ * v[idx];
    }
    return {price, cfg.elements};
}

}  // namespace bspde
