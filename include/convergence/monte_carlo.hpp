// Monte Carlo for the same contract, and the variance reduction that decides
// whether it is usable.
//
// Three ways of getting the spot forward in time:
//
//   Exact     S_{t+dt} = S_t exp((r-q-sig^2/2)dt + sig sqrt(dt) Z). Geometric
//             Brownian motion is integrable, so there is no discretisation
//             error at all, at any number of steps. Any bias is sampling noise.
//   Euler     Euler-Maruyama on dS = (r-q)S dt + sig S dW. Weak order 1.
//   Milstein  Adds the 0.5 sig^2 S (dW^2 - dt) correction. Strong order 1.
//
// Euler and Milstein are not needed to price a European option. They are here
// because the whole point of a discretisation scheme is what it costs you, and
// that is only visible when an exact answer sits next to it.
//
// On top of the scheme sit six variance reduction techniques, each switchable
// and all composable:
//
//   Antithetic variates    Price the path driven by Z and by -Z, average the
//                          two. Exact for the linear part of the payoff, so it
//                          removes the odd component of the integrand.
//   Control variates       Subtract beta times the deviation of a statistic
//                          whose mean is known. Two are offered: the terminal
//                          spot (mean = the forward) and a delta-hedge
//                          martingale (mean = 0, and it is the better of the
//                          two once the path has more than a couple of steps).
//   Stratified sampling    Force one draw per stratum of the terminal
//                          Brownian increment instead of hoping the sample
//                          covers it. The rest of the path is filled in by
//                          Brownian bridge, so the stratified coordinate is
//                          the one that actually drives S_T.
//   Importance sampling    Shift the terminal normal by theta and reweight by
//                          the Girsanov likelihood ratio. For an option that
//                          is nearly always worthless this is the difference
//                          between an answer and noise.
//   Moment matching        Rescale the terminal spots so their sample mean is
//                          the forward exactly. Cheap, effective, and it
//                          introduces an O(1/N) bias -- see the note below.
//   Conditional MC         Simulate only to t_c = fraction * T and close the
//                          remaining leg with Black-Scholes. Integrating out
//                          the last piece of randomness cannot raise the
//                          variance (Rao-Blackwell) and here it collapses it:
//                          at fraction = 0 the estimator is the closed form
//                          and the standard error is exactly zero.
//
// Two honesty notes, because a variance reduction that quietly breaks the
// estimator is worse than none:
//
//   * The control variate coefficients are fitted on an independent pilot run
//     with a different seed, never on the sample they correct. Fitting beta on
//     the same paths biases the price.
//   * Moment matching is the one technique here that is not unbiased, and it is
//     off by default for that reason. Rescaling couples the paths, so the
//     estimator picks up an O(1/N) bias -- at 500 paths on the at-the-money
//     call it is around +0.009, against a price of 10.45. The coupling also
//     hides from the variance estimator: measured over 400 seeds the true
//     spread is about 2.5x tighter than the reported standard error, so the
//     interval is conservative rather than wrong, but it is no longer a
//     measurement of anything. Moment matching also makes the terminal-spot
//     control variate redundant, since it has already forced that mean.
//
// Every result carries a standard error. A Monte Carlo price quoted without one
// is not a result, it is a number. Under stratification the naive formula would
// be wrong -- the draws are no longer independent -- so the standard error is
// computed from the within-stratum variances instead.
//
// Normals come from inverting the standard normal CDF over the raw 64-bit
// stream rather than from <random>'s distributions, which are not specified to
// produce the same numbers on two implementations. A seed here means the same
// paths on every platform.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "option.hpp"

namespace convergence {

enum class McScheme { Exact, Euler, Milstein };

struct McConfig {
    std::int64_t paths{200000};
    int steps{1};  // Exact is unbiased at any number; Euler and Milstein are not
    McScheme scheme{McScheme::Exact};

    // -------------------------------------------------- variance reduction
    bool antithetic{true};
    bool control_variate{true};         // terminal spot, mean = forward
    bool delta_control_variate{false};  // delta-hedge martingale, mean = 0
    bool stratified{false};
    int strata{0};  // 0 = choose automatically (>= 32 paths per stratum)
    bool importance_sampling{false};
    double importance_shift{0.0};  // 0 = choose automatically (centres S_T on K)
    bool moment_matching{false};   // biased at O(1/N); see the header note
    // Simulate to conditional_fraction * T and close the rest in closed form.
    // 1.0 is plain Monte Carlo, 0.0 is the analytic price with zero variance.
    double conditional_fraction{1.0};

    std::uint64_t seed{20170401};
};

struct McResult {
    double price{};
    double standard_error{};
    std::int64_t paths{};

    // What the run actually did, so a reported number can be audited.
    int strata_used{1};
    double beta_spot{};         // fitted coefficient on the terminal-spot control
    double beta_delta{};        // fitted coefficient on the delta-hedge control
    double shift_used{};        // theta of the importance sampling measure change
    double moment_factor{1.0};  // terminal rescaling applied by moment matching

    // 95% interval, reported so a claim of agreement can be checked rather
    // than eyeballed.
    [[nodiscard]] double ci_low() const noexcept { return price - 1.959964 * standard_error; }
    [[nodiscard]] double ci_high() const noexcept { return price + 1.959964 * standard_error; }
};

namespace detail {

// Inverse standard normal CDF: Acklam's rational approximation followed by one
// Halley step against erfc, which takes the relative error to around 1e-15.
// Stratification and portable path generation both need the inverse, not just
// a way to draw normals.
[[nodiscard]] inline double inv_norm_cdf(double p) noexcept {
    static constexpr double a[6] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                    -2.759285104469687e+02, 1.383577518672690e+02,
                                    -3.066479806614716e+01, 2.506628277459239e+00};
    static constexpr double b[5] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                    -1.556989798598866e+02, 6.680131188771972e+01,
                                    -1.328068155288572e+01};
    static constexpr double c[6] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                    -2.400758277161838e+00, -2.549732539343734e+00,
                                    4.374664141464968e+00,  2.938163982698783e+00};
    static constexpr double d[4] = {7.784695709041462e-03, 3.224671290700398e-01,
                                    2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double p_low = 0.02425;

    double x;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= 1.0 - p_low) {
        const double q = p - 0.5, r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    // Halley: e = Phi(x) - p, u = e * sqrt(2 pi) * exp(x^2 / 2).
    const double e = norm_cdf(x) - p;
    const double u = e / kInvSqrt2Pi * std::exp(0.5 * x * x);
    return x - u / (1.0 + 0.5 * x * u);
}

// Uniform on the open interval (0, 1) from the top 53 bits, offset by half an
// ulp so that neither endpoint can be produced and the log in the tail branch
// of the inverse CDF is always finite.
[[nodiscard]] inline double uniform01(std::mt19937_64& rng) noexcept {
    return (static_cast<double>(rng() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// One sample: its value, and the two controls whose means are known.
struct McSample {
    double value{};    // discounted payoff (or its conditional expectation)
    double c_spot{};   // terminal spot, mean = forward to the simulated horizon
    double c_delta{};  // delta-hedge P&L, mean = 0
};

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
    if (cfg.steps < 1) throw std::invalid_argument("monte_carlo: steps must be positive");
    if (cfg.conditional_fraction < 0.0 || cfg.conditional_fraction > 1.0) {
        throw std::invalid_argument(
            "monte_carlo: conditional_fraction must lie in [0, 1]");
    }
    // Four of the six reductions are built on the exact dynamics. The control
    // variates take their means from them -- E[S_T] is the forward, and
    // exp(-(r-q)t) S_t is a martingale. Moment matching rescales towards that
    // same forward. Conditional Monte Carlo closes the remaining leg with the
    // Black-Scholes formula. Euler and Milstein satisfy none of it: their mean
    // terminal spot is S_0 (1 + (r-q) dt)^n, not the forward, and the clamp at
    // zero moves even that. Used together they do not merely lose accuracy,
    // they shift the price by a constant, which no amount of sampling removes.
    //
    // Antithetic variates, stratification and importance sampling are
    // properties of the normals rather than of the dynamics, so they compose
    // with any scheme and are left alone here.
    if (cfg.scheme != McScheme::Exact) {
        const char* why = nullptr;
        if (cfg.control_variate || cfg.delta_control_variate) {
            why = "the control variates take their means from the exact dynamics";
        } else if (cfg.moment_matching) {
            why = "moment matching rescales towards the exact forward";
        } else if (cfg.conditional_fraction < 1.0) {
            why = "conditional Monte Carlo closes the tail with the Black-Scholes formula";
        }
        if (why != nullptr) {
            throw std::invalid_argument(
                std::string("monte_carlo: ") + why +
                ", which is not what Euler or Milstein simulate. The combination would "
                "bias the price rather than only tighten it. Use McScheme::Exact, or keep "
                "to antithetic variates, stratification and importance sampling, which "
                "hold under any scheme");
        }
    }

    const double T = opt.maturity, sig = mkt.volatility;
    const double r = mkt.rate, q = mkt.dividend;
    const double drift = r - q;
    const double disc_T = std::exp(-r * T);

    // The simulated horizon. Everything past it is integrated out in closed
    // form, which is the conditional Monte Carlo estimator.
    const double horizon = cfg.conditional_fraction * T;
    const double tail = T - horizon;

    McResult out;
    out.paths = cfg.paths;

    // Conditioning all the way back to time zero leaves nothing to simulate:
    // the estimator is the closed form and its variance is zero. Worth
    // returning rather than dividing by a zero horizon.
    if (horizon <= 0.0 || T <= 0.0) {
        out.price = analytic_price(opt, mkt);
        out.standard_error = 0.0;
        out.strata_used = 1;
        return out;
    }

    const int n = cfg.steps;
    const double dt = horizon / n;
    const double sqrt_h = std::sqrt(horizon);
    const double disc_h = std::exp(-r * horizon);
    // E[S_h] under the risk-neutral measure: the forward to the simulated
    // horizon, and the known mean of the terminal-spot control.
    const double forward_h = mkt.spot * std::exp(drift * horizon);

    // Strata for the terminal Brownian increment.
    int strata = 1;
    if (cfg.stratified) {
        strata = cfg.strata;
        if (strata <= 0) {
            strata = static_cast<int>(std::min<std::int64_t>(
                std::max<std::int64_t>(cfg.paths / 32, 1), 1024));
        }
        if (strata < 1) throw std::invalid_argument("monte_carlo: strata must be positive");
        if (static_cast<std::int64_t>(strata) * 2 > cfg.paths) {
            throw std::invalid_argument(
                "monte_carlo: need at least 2 paths per stratum to estimate a "
                "within-stratum variance");
        }
    }
    out.strata_used = strata;

    // Importance sampling shift. The automatic choice centres the terminal
    // log-spot on the strike, which for a single step is exactly -d2. It is
    // the right shift for an option that is usually worthless; for one that is
    // deep in the money it moves probability the wrong way, so the caller can
    // set importance_shift instead.
    double theta = 0.0;
    if (cfg.importance_sampling) {
        theta = cfg.importance_shift;
        if (theta == 0.0) {
            theta = (std::log(opt.strike / mkt.spot) - (drift - 0.5 * sig * sig) * horizon) /
                    (sig * sqrt_h);
        }
    }
    out.shift_used = theta;

    std::mt19937_64 rng(cfg.seed);
    std::vector<double> z(static_cast<std::size_t>(n));   // z[0] terminal, z[1..] bridge
    std::vector<double> w(static_cast<std::size_t>(n) + 1);  // Brownian path on [0, horizon]

    // One branch of a path: sign = +1 for the draw, -1 for its antithetic.
    auto run_branch = [&](double sign, double mm) -> detail::McSample {
        const double z0 = sign * z[0];
        const double zt = z0 + theta;
        // Girsanov weight for shifting the terminal normal. Exactly 1 when
        // theta is 0, so the unshifted path needs no special case.
        const double L = std::exp(-theta * z0 - 0.5 * theta * theta);

        // Brownian bridge: pin the endpoint the stratified draw chose, then
        // fill the interior left to right from its conditional law.
        w[0] = 0.0;
        w[static_cast<std::size_t>(n)] = sqrt_h * zt;
        for (int i = 1; i < n; ++i) {
            const double t_prev = (i - 1) * dt, t_i = i * dt;
            const double span = horizon - t_prev;
            const double mean = w[static_cast<std::size_t>(i - 1)] +
                                dt / span * (w[static_cast<std::size_t>(n)] -
                                             w[static_cast<std::size_t>(i - 1)]);
            const double var = dt * (horizon - t_i) / span;
            w[static_cast<std::size_t>(i)] =
                mean + std::sqrt(std::fmax(var, 0.0)) * sign * z[static_cast<std::size_t>(i)];
        }

        double s = mkt.spot;
        double c_delta = 0.0;
        double m_prev = mkt.spot;  // exp(-(r-q)*0) * S_0, a Q-martingale at t=0

        for (int i = 0; i < n; ++i) {
            double delta_i = 0.0;
            if (cfg.delta_control_variate && s > 0.0) {
                Option rest = opt;
                rest.maturity = T - i * dt;
                Market here = mkt;
                here.spot = s;
                delta_i = analytic(rest, here).delta;
            }

            const double dw = w[static_cast<std::size_t>(i + 1)] - w[static_cast<std::size_t>(i)];
            if (cfg.scheme == McScheme::Exact) {
                s *= std::exp((drift - 0.5 * sig * sig) * dt + sig * dw);
            } else if (cfg.scheme == McScheme::Euler) {
                s += drift * s * dt + sig * s * dw;
            } else {  // Milstein
                s += drift * s * dt + sig * s * dw + 0.5 * sig * sig * s * (dw * dw - dt);
            }
            // Euler can step a positive process negative. Clamping is the
            // standard fix and it is a real source of bias, not a formality.
            if (s < 0.0) s = 0.0;

            if (cfg.delta_control_variate) {
                // exp(-(r-q)t) S_t is a Q-martingale for any dividend yield, so
                // the hedge sum has mean zero exactly. Its correlation with the
                // payoff is what makes it the strongest control here.
                const double m_cur = std::exp(-drift * (i + 1) * dt) * s;
                c_delta += delta_i * (m_cur - m_prev);
                m_prev = m_cur;
            }
        }

        const double s_end = s * mm;

        double value;
        if (tail <= 0.0) {
            value = disc_T * opt.payoff(s_end);
        } else if (s_end <= 0.0) {
            // Absorbed at zero: the remaining leg is worth its payoff at zero.
            value = disc_T * opt.payoff(0.0);
        } else {
            Option rest = opt;
            rest.maturity = tail;
            Market here = mkt;
            here.spot = s_end;
            value = disc_h * analytic_price(rest, here);
        }

        return detail::McSample{L * value, L * s_end, L * c_delta};
    };

    // A whole sample: one branch, or the average of a branch and its
    // antithetic. stratum < 0 means draw the terminal uniform unstratified.
    auto draw_sample = [&](int stratum, double mm) -> detail::McSample {
        const double u = (stratum < 0)
                             ? detail::uniform01(rng)
                             : (static_cast<double>(stratum) + detail::uniform01(rng)) /
                                   static_cast<double>(strata);
        z[0] = detail::inv_norm_cdf(u);
        for (int i = 1; i < n; ++i) z[static_cast<std::size_t>(i)] = detail::inv_norm_cdf(detail::uniform01(rng));

        detail::McSample a = run_branch(+1.0, mm);
        if (!cfg.antithetic) return a;
        // Negating the draw negates the stratified coordinate too, which maps
        // stratum j onto stratum (strata-1-j): the design survives.
        const detail::McSample b = run_branch(-1.0, mm);
        return detail::McSample{0.5 * (a.value + b.value), 0.5 * (a.c_spot + b.c_spot),
                                0.5 * (a.c_delta + b.c_delta)};
    };

    // Paths per stratum, with the remainder spread over the first strata.
    std::vector<std::int64_t> per(static_cast<std::size_t>(strata),
                                  cfg.paths / strata);
    for (std::int64_t i = 0; i < cfg.paths % strata; ++i) ++per[static_cast<std::size_t>(i)];

    // A full pass over the design, deterministic in the seed so it can be
    // repeated verbatim -- which is what moment matching needs.
    auto sweep = [&](double mm, auto&& on_sample) {
        rng.seed(cfg.seed);
        for (int j = 0; j < strata; ++j) {
            for (std::int64_t k = 0; k < per[static_cast<std::size_t>(j)]; ++k) {
                on_sample(j, draw_sample(cfg.stratified ? j : -1, mm));
            }
        }
    };

    // ------------------------------------------------ control variate betas
    //
    // Fitted on an independent pilot: a different seed, and no stratification,
    // so the coefficients carry no information about the sample they correct.
    //
    // Three things make this more than a one-line regression.
    //
    // The pilot is stored and centred in a second pass rather than accumulated
    // as raw moments. The terminal spot sits around 100 with a standard
    // deviation of about 20, so s11 - (sx1)^2/n discards most of its
    // significant digits.
    //
    // The two controls can be collinear. At a single step the delta hedge is
    // an affine function of the terminal spot and nothing more, so the normal
    // equations are singular; solved anyway they hand back two enormous betas
    // that cancel each other in the pilot and cancel nowhere else.
    //
    // And collinearity is not the only way that happens. Under importance
    // sampling both controls are multiplied by a heavy-tailed likelihood
    // ratio, which is enough to make a merely ill-conditioned fit produce the
    // same cancelling pair without ever tripping a correlation test. So the
    // choice of which controls to use is not argued, it is measured: the pilot
    // is split, each candidate set is fitted on the first half and scored by
    // the variance it actually delivers on the second, and the winner is
    // refitted on the whole pilot. A set that only looks good where it was
    // fitted loses to using no control at all, which is the right answer.
    constexpr std::int64_t kMinPilot = 64;
    double beta_spot = 0.0, beta_delta = 0.0;
    if (cfg.control_variate || cfg.delta_control_variate) {
        const std::int64_t pilot = std::min<std::int64_t>(cfg.paths / 10, 20000);
        if (pilot >= kMinPilot) {
            rng.seed(cfg.seed ^ 0x9E3779B97F4A7C15ULL);
            std::vector<detail::McSample> sample(static_cast<std::size_t>(pilot));
            for (auto& sp : sample) sp = draw_sample(-1, 1.0);

            // Least squares of value on the requested controls over [lo, hi).
            auto fit = [&](std::size_t lo, std::size_t hi, bool w1, bool w2, double& b1,
                           double& b2) {
                b1 = 0.0;
                b2 = 0.0;
                const double n = static_cast<double>(hi - lo);
                double mx1 = 0, mx2 = 0, my = 0;
                for (std::size_t k = lo; k < hi; ++k) {
                    mx1 += sample[k].c_spot; mx2 += sample[k].c_delta; my += sample[k].value;
                }
                mx1 /= n; mx2 /= n; my /= n;
                double c11 = 0, c12 = 0, c22 = 0, c1y = 0, c2y = 0;
                double q11 = 0, q22 = 0;
                for (std::size_t k = lo; k < hi; ++k) {
                    const double x1 = sample[k].c_spot - mx1;
                    const double x2 = sample[k].c_delta - mx2;
                    const double y = sample[k].value - my;
                    c11 += x1 * x1; c12 += x1 * x2; c22 += x2 * x2;
                    c1y += x1 * y;  c2y += x2 * y;
                    q11 += sample[k].c_spot * sample[k].c_spot;
                    q22 += sample[k].c_delta * sample[k].c_delta;
                }
                // A control has to vary to be worth anything, and "varies" has
                // to be judged against its own scale: a control that is
                // constant up to rounding still has a positive computed
                // variance, and dividing by it is how a beta of 1e14 happens.
                const bool u1 = w1 && c11 > 1e-10 * q11;
                const bool u2 = w2 && c22 > 1e-10 * q22;
                if (u1 && u2) {
                    const double rho_sq = (c12 * c12) / (c11 * c22);
                    if (rho_sq < 1.0 - 1e-8) {
                        const double det = c11 * c22 - c12 * c12;
                        b1 = (c1y * c22 - c2y * c12) / det;
                        b2 = (c2y * c11 - c1y * c12) / det;
                        return;
                    }
                    b2 = c2y / c22;  // identical information; keep the hedge
                    return;
                }
                if (u1) b1 = c1y / c11;
                if (u2) b2 = c2y / c22;
            };

            // Variance the fitted pair actually delivers over [lo, hi). The
            // constant beta*mean shifts the mean, not the spread, so it is
            // left out.
            auto score = [&](std::size_t lo, std::size_t hi, double b1, double b2) {
                const double n = static_cast<double>(hi - lo);
                double sum = 0, sum_sq = 0;
                for (std::size_t k = lo; k < hi; ++k) {
                    const double v =
                        sample[k].value - b1 * sample[k].c_spot - b2 * sample[k].c_delta;
                    sum += v;
                    sum_sq += v * v;
                }
                const double mean = sum / n;
                return std::fmax(sum_sq / n - mean * mean, 0.0);
            };

            const std::size_t np = sample.size();
            const std::size_t half = np / 2;

            // Candidate control sets, including the empty one.
            const bool w1 = cfg.control_variate;
            const bool w2 = cfg.delta_control_variate;
            const bool masks[4][2] = {{false, false}, {w1, false}, {false, w2}, {w1, w2}};
            double best_score = 0.0;
            int best = 0;
            for (int m = 0; m < 4; ++m) {
                double b1 = 0, b2 = 0;
                fit(0, half, masks[m][0], masks[m][1], b1, b2);
                const double v = score(half, np, b1, b2);
                if (m == 0 || v < best_score) { best_score = v; best = m; }
            }
            // Refit the winning structure on the whole pilot.
            fit(0, np, masks[best][0], masks[best][1], beta_spot, beta_delta);
        }
    }
    out.beta_spot = beta_spot;
    out.beta_delta = beta_delta;

    // ---------------------------------------------------- moment matching
    //
    // First pass measures the sample mean of the terminal spot under the same
    // stratified design; the second rescales every terminal spot so that mean
    // is the forward exactly.
    double mm = 1.0;
    if (cfg.moment_matching) {
        std::vector<double> sum(static_cast<std::size_t>(strata), 0.0);
        sweep(1.0, [&](int j, const detail::McSample& s) {
            sum[static_cast<std::size_t>(j)] += s.c_spot;
        });
        double est = 0.0;
        for (int j = 0; j < strata; ++j) {
            est += (sum[static_cast<std::size_t>(j)] /
                    static_cast<double>(per[static_cast<std::size_t>(j)])) /
                   static_cast<double>(strata);
        }
        if (est > 0.0) mm = forward_h / est;
    }
    out.moment_factor = mm;

    // ------------------------------------------------------- the estimator
    std::vector<double> sum(static_cast<std::size_t>(strata), 0.0);
    std::vector<double> sum_sq(static_cast<std::size_t>(strata), 0.0);

    sweep(mm, [&](int j, const detail::McSample& s) {
        const double adjusted =
            s.value - beta_spot * (s.c_spot - forward_h) - beta_delta * s.c_delta;
        sum[static_cast<std::size_t>(j)] += adjusted;
        sum_sq[static_cast<std::size_t>(j)] += adjusted * adjusted;
    });

    // Stratified mean and its variance: each stratum carries probability
    // 1/strata, so the estimator is the unweighted average of the stratum
    // means and its variance is the sum of their scaled within-stratum
    // variances. With one stratum this is exactly the usual formula.
    double price = 0.0, var_mean = 0.0;
    const double p_j = 1.0 / static_cast<double>(strata);
    for (int j = 0; j < strata; ++j) {
        const auto idx = static_cast<std::size_t>(j);
        const double nj = static_cast<double>(per[idx]);
        const double mean_j = sum[idx] / nj;
        price += p_j * mean_j;
        if (nj > 1.0) {
            // Sample variance with Bessel's correction.
            const double var_j = (sum_sq[idx] / nj - mean_j * mean_j) * nj / (nj - 1.0);
            var_mean += p_j * p_j * std::fmax(var_j, 0.0) / nj;
        }
    }

    out.price = price;
    out.standard_error = std::sqrt(std::fmax(var_mean, 0.0));
    return out;
}

}  // namespace convergence
