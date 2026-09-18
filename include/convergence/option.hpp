// Shared description of the contract and the market, plus the closed-form
// price that every numerical scheme in this library is measured against.
//
// The analytic solution is the oracle. No scheme here is trusted because it
// looks plausible; each one is required to converge to this, at its own
// theoretical rate.

#pragma once

#include <cmath>
#include <stdexcept>

namespace convergence {

enum class OptionType { Call, Put };
enum class Exercise { European, American };

struct Option {
    double strike{100.0};
    double maturity{1.0};  // years
    OptionType type{OptionType::Call};
    Exercise exercise{Exercise::European};

    // Terminal condition of the PDE, and the early-exercise obstacle.
    [[nodiscard]] double payoff(double spot) const noexcept {
        return type == OptionType::Call ? std::fmax(spot - strike, 0.0)
                                        : std::fmax(strike - spot, 0.0);
    }
};

struct Market {
    double spot{100.0};
    double rate{0.05};        // continuously compounded
    double volatility{0.20};  // annualised
    double dividend{0.0};     // continuous dividend yield q

    void validate() const {
        if (spot <= 0.0) throw std::invalid_argument("spot must be positive");
        if (volatility <= 0.0) throw std::invalid_argument("volatility must be positive");
    }
};

// 1/sqrt(2) and 1/sqrt(2*pi), written out rather than taken from <cmath>'s
// M_* macros, which are not standard and are absent on MSVC unless
// _USE_MATH_DEFINES is set before the include.
inline constexpr double kInvSqrt2 = 0.70710678118654752440;
inline constexpr double kInvSqrt2Pi = 0.39894228040143267794;

// Standard normal CDF via erfc. Using erfc rather than 0.5*(1+erf(x/sqrt2))
// keeps precision in the far left tail, where the naive form cancels.
[[nodiscard]] inline double norm_cdf(double x) noexcept {
    return 0.5 * std::erfc(-x * kInvSqrt2);
}

[[nodiscard]] inline double norm_pdf(double x) noexcept {
    return kInvSqrt2Pi * std::exp(-0.5 * x * x);
}

struct Greeks {
    double price{};
    double delta{};
    double gamma{};
    double vega{};
    double theta{};
    double rho{};
};

// Black-Scholes-Merton with a continuous dividend yield.
[[nodiscard]] inline Greeks analytic(const Option& opt, const Market& mkt) {
    mkt.validate();
    const double S = mkt.spot, K = opt.strike, T = opt.maturity;
    const double r = mkt.rate, q = mkt.dividend, sig = mkt.volatility;

    if (T <= 0.0) {  // expired: worth its payoff, no sensitivities
        return Greeks{opt.payoff(S), 0.0, 0.0, 0.0, 0.0, 0.0};
    }

    const double sqrtT = std::sqrt(T);
    const double vol_sqrtT = sig * sqrtT;
    const double d1 = (std::log(S / K) + (r - q + 0.5 * sig * sig) * T) / vol_sqrtT;
    const double d2 = d1 - vol_sqrtT;
    const double df_r = std::exp(-r * T);
    const double df_q = std::exp(-q * T);

    Greeks g;
    if (opt.type == OptionType::Call) {
        g.price = S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2);
        g.delta = df_q * norm_cdf(d1);
        g.rho = K * T * df_r * norm_cdf(d2);
        g.theta = -S * df_q * norm_pdf(d1) * sig / (2.0 * sqrtT)
                  - r * K * df_r * norm_cdf(d2) + q * S * df_q * norm_cdf(d1);
    } else {
        g.price = K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1);
        g.delta = -df_q * norm_cdf(-d1);
        g.rho = -K * T * df_r * norm_cdf(-d2);
        g.theta = -S * df_q * norm_pdf(d1) * sig / (2.0 * sqrtT)
                  + r * K * df_r * norm_cdf(-d2) - q * S * df_q * norm_cdf(-d1);
    }
    g.gamma = df_q * norm_pdf(d1) / (S * vol_sqrtT);
    g.vega = S * df_q * norm_pdf(d1) * sqrtT;
    return g;
}

[[nodiscard]] inline double analytic_price(const Option& opt, const Market& mkt) {
    return analytic(opt, mkt).price;
}

}  // namespace convergence
