// Monte Carlo for the same contract.
//
// Three ways of getting the terminal spot:
//
//   Exact     S_T = S_0 exp((r-q-sig^2/2)T + sig sqrt(T) Z). Geometric Brownian
//             motion is integrable, so there is no discretisation error at all
//             and one step suffices. Any bias you see is pure sampling noise.
//   Euler     Euler-Maruyama on dS = (r-q)S dt + sig S dW. Weak order 1.
//   Milstein  Adds the 0.5 sig^2 S (dW^2 - dt) correction. Strong order 1.
//
// Euler and Milstein are not needed to price a European option. They are here
// because the whole point of a discretisation scheme is what it costs you, and
// that is only visible when an exact answer sits next to it.
//
// Every result carries a standard error. A Monte Carlo price quoted without one
// is not a result, it is a number.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "option.hpp"

namespace convergence {

enum class McScheme { Exact, Euler, Milstein };

struct McConfig {
    std::int64_t paths{200000};
    int steps{1};  // ignored by Exact, which needs only one
    McScheme scheme{McScheme::Exact};
    bool antithetic{true};
    bool control_variate{true};
    std::uint64_t seed{20170401};
};

struct McResult {
    double price{};
    double standard_error{};
    std::int64_t paths{};
    // 95% interval, reported so a claim of agreement can be checked rather
    // than eyeballed.
    [[nodiscard]] double ci_low() const noexcept { return price - 1.959964 * standard_error; }
    [[nodiscard]] double ci_high() const noexcept { return price + 1.959964 * standard_error; }
};

namespace detail {

// One path to maturity under the chosen scheme.
inline double terminal_spot(McScheme scheme, double s0, double drift, double sig,
                            double T, int steps, const std::vector<double>& z) {
    if (scheme == McScheme::Exact) {
        return s0 * std::exp((drift - 0.5 * sig * sig) * T + sig * std::sqrt(T) * z[0]);
    }
    const double dt = T / steps;
    const double sqrt_dt = std::sqrt(dt);
    double s = s0;
    for (int i = 0; i < steps; ++i) {
        const double dw = sqrt_dt * z[static_cast<std::size_t>(i)];
        if (scheme == McScheme::Euler) {
            s += drift * s * dt + sig * s * dw;
        } else {  // Milstein
            s += drift * s * dt + sig * s * dw + 0.5 * sig * sig * s * (dw * dw - dt);
        }
        // Euler can step a positive process negative. Clamping is the standard
        // fix and it is a real source of bias, not a formality.
        if (s < 0.0) s = 0.0;
    }
    return s;
}

}  // namespace detail

[[nodiscard]] inline McResult monte_carlo(const Option& opt, const Market& mkt,
                                          const McConfig& cfg) {
    mkt.validate();
    if (opt.exercise == Exercise::American) {
        throw std::invalid_argument(
            "monte_carlo: American exercise needs a regression method "
            "(Longstaff-Schwartz); forward simulation alone cannot price it");
    }
    if (cfg.paths < 2) throw std::invalid_argument("monte_carlo: need at least 2 paths");
    const int steps = (cfg.scheme == McScheme::Exact) ? 1 : cfg.steps;
    if (steps < 1) throw std::invalid_argument("monte_carlo: steps must be positive");

    const double T = opt.maturity, sig = mkt.volatility;
    const double drift = mkt.rate - mkt.dividend;
    const double disc = std::exp(-mkt.rate * T);

    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    // Control variate: the discounted terminal spot has known expectation
    // S_0 * exp(-q*T). Its correlation with the payoff is high for a call, so
    // subtracting the beta-weighted deviation removes much of the variance.
    // Beta is estimated on a small pilot run and then held fixed, so it stays
    // independent of the sample it corrects -- estimating it on the same paths
    // biases the result.
    // E[S_T] under the risk-neutral measure is the forward.
    const double forward = mkt.spot * std::exp((mkt.rate - mkt.dividend) * T);
    double beta = 0.0;

    std::vector<double> z(static_cast<std::size_t>(steps));
    auto draw_payoff_pair = [&](double& payoff, double& control) {
        for (auto& zi : z) zi = gauss(rng);
        const double s = detail::terminal_spot(cfg.scheme, mkt.spot, drift, sig, T, steps, z);
        payoff = opt.payoff(s);
        control = s;
        if (cfg.antithetic) {
            for (auto& zi : z) zi = -zi;
            const double s2 = detail::terminal_spot(cfg.scheme, mkt.spot, drift, sig, T, steps, z);
            payoff = 0.5 * (payoff + opt.payoff(s2));
            control = 0.5 * (control + s2);
        }
    };

    if (cfg.control_variate) {
        const std::int64_t pilot = std::min<std::int64_t>(cfg.paths / 10, 20000);
        if (pilot > 2) {
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            for (std::int64_t i = 0; i < pilot; ++i) {
                double py, cv;
                draw_payoff_pair(py, cv);
                sx += cv; sy += py; sxx += cv * cv; sxy += cv * py;
            }
            const double np = static_cast<double>(pilot);
            const double cov = sxy / np - (sx / np) * (sy / np);
            const double var = sxx / np - (sx / np) * (sx / np);
            if (var > 0.0) beta = cov / var;
        }
    }

    double sum = 0.0, sum_sq = 0.0;
    for (std::int64_t i = 0; i < cfg.paths; ++i) {
        double payoff, control;
        draw_payoff_pair(payoff, control);
        const double adjusted = payoff - beta * (control - forward);
        sum += adjusted;
        sum_sq += adjusted * adjusted;
    }

    const double np = static_cast<double>(cfg.paths);
    const double mean = sum / np;
    // Sample variance of the mean, with Bessel's correction.
    const double var = (sum_sq / np - mean * mean) * np / (np - 1.0);

    McResult out;
    out.price = disc * mean;
    out.standard_error = disc * std::sqrt(std::fmax(var, 0.0) / np);
    out.paths = cfg.paths;
    return out;
}

}  // namespace convergence
