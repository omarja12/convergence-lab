// Lattice methods: binomial (three parameterisations) and Boyle trinomial.
//
// All of them discretise the same risk-neutral dynamics, and all converge to
// Black-Scholes. They differ in how the moments are matched, and the
// differences are visible in how they converge rather than in where they land:
// CRR oscillates as the strike moves between nodes, Jarrow-Rudd does not.

#pragma once

#include <stdexcept>
#include <vector>

#include "option.hpp"

namespace convergence {

enum class BinomialModel {
    CoxRossRubinstein,  // u = 1/d, the standard choice
    JarrowRudd,         // p = 1/2, drift carried by the nodes
    Tian,               // matches the first three moments exactly
};

struct LatticeResult {
    double price{};
    int steps{};
};

namespace detail {

struct LatticeParams {
    double up{}, down{}, p_up{};
};

inline LatticeParams binomial_params(BinomialModel model, double dt, double r,
                                     double q, double sig) {
    const double nu = r - q - 0.5 * sig * sig;
    const double sqrt_dt = std::sqrt(dt);
    const double growth = std::exp((r - q) * dt);
    LatticeParams lp;

    switch (model) {
        case BinomialModel::CoxRossRubinstein: {
            lp.up = std::exp(sig * sqrt_dt);
            lp.down = 1.0 / lp.up;
            lp.p_up = (growth - lp.down) / (lp.up - lp.down);
            break;
        }
        case BinomialModel::JarrowRudd: {
            // Equal probabilities; the drift is absorbed into the node spacing.
            lp.up = std::exp(nu * dt + sig * sqrt_dt);
            lp.down = std::exp(nu * dt - sig * sqrt_dt);
            lp.p_up = 0.5;
            break;
        }
        case BinomialModel::Tian: {
            const double v = std::exp(sig * sig * dt);
            const double rad = std::sqrt(v * v + 2.0 * v - 3.0);
            lp.up = 0.5 * growth * v * (v + 1.0 + rad);
            lp.down = 0.5 * growth * v * (v + 1.0 - rad);
            lp.p_up = (growth - lp.down) / (lp.up - lp.down);
            break;
        }
    }
    return lp;
}

}  // namespace detail

// Backward induction on a recombining binomial tree.
//
// Only one vector is held: the layer being folded back. An American option is
// handled by taking the max against the payoff at every node, which is exactly
// the discrete obstacle problem the PDE formulation solves variationally.
[[nodiscard]] inline LatticeResult binomial(const Option& opt, const Market& mkt,
                                            int steps,
                                            BinomialModel model = BinomialModel::CoxRossRubinstein) {
    mkt.validate();
    if (steps < 1) throw std::invalid_argument("steps must be at least 1");

    const double dt = opt.maturity / steps;
    const auto lp = detail::binomial_params(model, dt, mkt.rate, mkt.dividend, mkt.volatility);

    if (lp.p_up < 0.0 || lp.p_up > 1.0) {
        throw std::runtime_error(
            "binomial: risk-neutral probability outside [0,1] - time step too large "
            "for this volatility; the tree would admit arbitrage");
    }

    const double disc = std::exp(-mkt.rate * dt);

    // Terminal layer: spot after j up-moves and (steps-j) down-moves.
    std::vector<double> value(static_cast<std::size_t>(steps) + 1);
    for (int j = 0; j <= steps; ++j) {
        const double s = mkt.spot * std::pow(lp.up, j) * std::pow(lp.down, steps - j);
        value[static_cast<std::size_t>(j)] = opt.payoff(s);
    }

    for (int n = steps - 1; n >= 0; --n) {
        for (int j = 0; j <= n; ++j) {
            const auto k = static_cast<std::size_t>(j);
            value[k] = disc * (lp.p_up * value[k + 1] + (1.0 - lp.p_up) * value[k]);
            if (opt.exercise == Exercise::American) {
                const double s = mkt.spot * std::pow(lp.up, j) * std::pow(lp.down, n - j);
                value[k] = std::fmax(value[k], opt.payoff(s));
            }
        }
    }
    return {value[0], steps};
}

// Boyle (1986) trinomial tree.
//
// The extra branch buys a free parameter in the node spacing. Taking
// dx = sig*sqrt(3*dt) is the usual choice: it keeps all three probabilities
// positive and gives the same O(1/N) convergence with a smaller constant.
[[nodiscard]] inline LatticeResult trinomial(const Option& opt, const Market& mkt, int steps) {
    mkt.validate();
    if (steps < 1) throw std::invalid_argument("steps must be at least 1");

    const double dt = opt.maturity / steps;
    const double sig = mkt.volatility;
    const double nu = mkt.rate - mkt.dividend - 0.5 * sig * sig;
    const double dx = sig * std::sqrt(3.0 * dt);

    const double sig2dt = sig * sig * dt;
    const double p_up = 0.5 * ((sig2dt + nu * nu * dt * dt) / (dx * dx) + nu * dt / dx);
    const double p_dn = 0.5 * ((sig2dt + nu * nu * dt * dt) / (dx * dx) - nu * dt / dx);
    const double p_md = 1.0 - p_up - p_dn;

    if (p_up < 0.0 || p_dn < 0.0 || p_md < 0.0) {
        throw std::runtime_error("trinomial: negative probability - time step too large");
    }

    const double disc = std::exp(-mkt.rate * dt);
    const double log_s = std::log(mkt.spot);

    // Index i runs over 2*steps+1 nodes, centred on the initial log-spot.
    const auto width = static_cast<std::size_t>(2 * steps + 1);
    std::vector<double> value(width);
    for (int i = -steps; i <= steps; ++i) {
        value[static_cast<std::size_t>(i + steps)] = opt.payoff(std::exp(log_s + i * dx));
    }

    // A scratch layer is required here, unlike the binomial. Each node reads its
    // neighbour below as well as above, so updating in place would consume a
    // value already overwritten in this same sweep and silently return a wrong
    // price rather than failing.
    std::vector<double> next(width);
    for (int n = steps - 1; n >= 0; --n) {
        for (int i = -n; i <= n; ++i) {
            const auto k = static_cast<std::size_t>(i + steps);
            double v = disc * (p_up * value[k + 1] + p_md * value[k] + p_dn * value[k - 1]);
            if (opt.exercise == Exercise::American) {
                v = std::fmax(v, opt.payoff(std::exp(log_s + i * dx)));
            }
            next[k] = v;
        }
        for (int i = -n; i <= n; ++i) {
            const auto k = static_cast<std::size_t>(i + steps);
            value[k] = next[k];
        }
    }
    return {value[static_cast<std::size_t>(steps)], steps};
}

}  // namespace convergence
