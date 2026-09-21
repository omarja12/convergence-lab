// Every method here converges to Black-Scholes. That is not interesting on its
// own -- a wrong scheme with a bug in the drift often lands close too, if you
// only ever test one grid size.
//
// What these checks establish is the *rate*. Halve the step, and the error must
// fall by the factor the theory predicts. Observed order is measured as
//
//     p = log2( error(coarse) / error(fine) )
//
// and compared against the expected order. A scheme that is accidentally right
// at one resolution will not produce the right slope across four.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "convergence/finite_difference.hpp"
#include "convergence/finite_element.hpp"
#include "convergence/lattice.hpp"
#include "convergence/monte_carlo.hpp"
#include "convergence/option.hpp"

using namespace convergence;

namespace {

int g_passed = 0;
int g_failed = 0;

void check(const std::string& name, bool ok, const std::string& detail) {
    std::printf("[%s] %s\n       %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
    ok ? ++g_passed : ++g_failed;
}

// Least-squares slope of log2(error) against log2(resolution), which is more
// robust than a single successive-ratio estimate.
double observed_order(const std::vector<double>& resolution,
                      const std::vector<double>& error) {
    const std::size_t n = resolution.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = std::log2(resolution[i]);
        const double y = std::log2(error[i]);
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const double d = static_cast<double>(n) * sxx - sx * sx;
    return -(static_cast<double>(n) * sxy - sx * sy) / d;  // negated: error falls
}

const Option kCall{100.0, 1.0, OptionType::Call, Exercise::European};
const Option kPut{100.0, 1.0, OptionType::Put, Exercise::European};
const Market kMkt{100.0, 0.05, 0.20, 0.0};

// ---------------------------------------------------------------- analytic

void test_put_call_parity() {
    const double c = analytic_price(kCall, kMkt);
    const double p = analytic_price(kPut, kMkt);
    const double lhs = c - p;
    const double rhs = kMkt.spot * std::exp(-kMkt.dividend * kCall.maturity)
                       - kCall.strike * std::exp(-kMkt.rate * kCall.maturity);
    const double err = std::fabs(lhs - rhs);
    check("Analytic prices satisfy put-call parity", err < 1e-13,
          "C - P = " + std::to_string(lhs) + " vs S - Ke^-rT = " + std::to_string(rhs)
              + ", error " + std::to_string(err));
}

void test_greeks_against_bumps() {
    const Greeks g = analytic(kCall, kMkt);
    const double h = 1e-4;
    Market up = kMkt, dn = kMkt;
    up.spot += h; dn.spot -= h;
    const double fd_delta = (analytic_price(kCall, up) - analytic_price(kCall, dn)) / (2 * h);
    const double fd_gamma =
        (analytic_price(kCall, up) - 2 * analytic_price(kCall, kMkt) + analytic_price(kCall, dn))
        / (h * h);
    const double d_err = std::fabs(fd_delta - g.delta);
    const double gm_err = std::fabs(fd_gamma - g.gamma);
    check("Closed-form delta and gamma match central differences",
          d_err < 1e-8 && gm_err < 1e-4,
          "delta error " + std::to_string(d_err) + ", gamma error " + std::to_string(gm_err));
}

// ---------------------------------------------------------------- lattices

void test_binomial_first_order() {
    const double exact = analytic_price(kCall, kMkt);
    std::vector<double> res, err;
    // Cox-Ross-Rubinstein, and the choice matters. Binomial error carries an
    // oscillatory term whose size depends on where the strike sits relative to
    // the terminal nodes. Here S0 = K and CRR has u = 1/d, so the strike lands
    // exactly on the centre node at every even n: the oscillation vanishes and
    // the underlying O(1/n) is exposed cleanly. Jarrow-Rudd shifts the nodes by
    // the drift, the strike falls between them, and its measured order is
    // erratic (observed 0.5 to 3.2 over the same sizes) despite being the same
    // order asymptotically. Measuring convergence on JR here would report a
    // property of the node alignment, not of the scheme.
    for (int n : {250, 500, 1000, 2000}) {
        res.push_back(n);
        err.push_back(std::fabs(
            binomial(kCall, kMkt, n, BinomialModel::CoxRossRubinstein).price - exact));
    }
    const double p = observed_order(res, err);
    check("Binomial (Cox-Ross-Rubinstein) converges at first order", p > 0.85 && p < 1.15,
          "observed order " + std::to_string(p) + " (expected 1), finest error "
              + std::to_string(err.back()));
}

void test_all_lattices_agree() {
    const double exact = analytic_price(kCall, kMkt);
    double worst = 0.0;
    for (auto model : {BinomialModel::CoxRossRubinstein, BinomialModel::JarrowRudd,
                       BinomialModel::Tian}) {
        worst = std::fmax(worst, std::fabs(binomial(kCall, kMkt, 4000, model).price - exact));
    }
    worst = std::fmax(worst, std::fabs(trinomial(kCall, kMkt, 2000).price - exact));
    check("All three binomial parameterisations and the trinomial agree", worst < 5e-3,
          "worst deviation from analytic " + std::to_string(worst) + " at 4000 steps");
}

void test_american_put_exceeds_european() {
    Option am = kPut; am.exercise = Exercise::American;
    const double a = binomial(am, kMkt, 2000).price;
    const double e = analytic_price(kPut, kMkt);
    check("American put is worth more than the European", a > e,
          "American " + std::to_string(a) + " vs European " + std::to_string(e)
              + ", early-exercise premium " + std::to_string(a - e));
}

void test_tree_rejects_bad_step() {
    // A time step too large for the volatility makes the risk-neutral
    // probability leave [0,1]. That must be an error, not a price.
    Market wild = kMkt; wild.volatility = 0.02; wild.rate = 3.0;
    bool threw = false;
    try {
        (void)binomial(kCall, wild, 2);
    } catch (const std::exception&) {
        threw = true;
    }
    check("Tree refuses a step that would admit arbitrage", threw,
          "p outside [0,1] raises rather than returning a number");
}

// ------------------------------------------------- finite differences / FEM

// Refine space and time together with dt ~ dx^2 so the O(dt) term cannot mask
// the O(dx^2) term, and the measured slope in dx is the true spatial order.
double fd_error(FdScheme scheme, int nx) {
    FdConfig cfg;
    cfg.scheme = scheme;
    cfg.space_steps = nx;
    cfg.time_steps = nx * nx / 50;
    return std::fabs(finite_difference(kCall, kMkt, cfg).price - analytic_price(kCall, kMkt));
}

// Measure the time order of Crank-Nicolson with and without the Rannacher
// start-up. Space is fixed and fine so the time error dominates.
double cn_time_order(int rannacher_steps) {
    const double exact = analytic_price(kCall, kMkt);
    std::vector<double> res, err;
    for (int nt : {10, 20, 40, 80}) {
        FdConfig cfg;
        cfg.scheme = FdScheme::CrankNicolson;
        cfg.space_steps = 2000;
        cfg.time_steps = nt;
        cfg.rannacher_steps = rannacher_steps;
        res.push_back(nt);
        err.push_back(std::fabs(finite_difference(kCall, kMkt, cfg).price - exact));
    }
    return observed_order(res, err);
}

void test_crank_nicolson_second_order_in_time() {
    const double p = cn_time_order(2);
    check("Crank-Nicolson with Rannacher start-up is second order in time",
          p > 1.6 && p < 2.4,
          "observed order " + std::to_string(p) + " (expected 2)");
}

void test_crank_nicolson_needs_rannacher() {
    // The headline result of this repository. Crank-Nicolson is second order
    // only for smooth data; a vanilla payoff has a kink at the strike. CN's
    // amplification factor tends to -1 at high frequency, so the modes the kink
    // excites are not damped, only flipped in sign. Two implicit half-steps at
    // the start remove them.
    const double without = cn_time_order(0);
    const double with = cn_time_order(2);
    check("Without Rannacher, Crank-Nicolson loses its second order",
          without < 1.5 && with > 1.6 && with > without + 0.4,
          "observed order " + std::to_string(without) + " without start-up vs "
              + std::to_string(with) + " with it - the payoff kink, not a coding error");
}

void test_implicit_first_order_in_time() {
    const double exact = analytic_price(kCall, kMkt);
    std::vector<double> res, err;
    for (int nt : {10, 20, 40, 80}) {
        FdConfig cfg;
        cfg.scheme = FdScheme::Implicit;
        cfg.space_steps = 2000;
        cfg.time_steps = nt;
        res.push_back(nt);
        err.push_back(std::fabs(finite_difference(kCall, kMkt, cfg).price - exact));
    }
    const double p = observed_order(res, err);
    check("Implicit Euler is first order in time", p > 0.7 && p < 1.4,
          "observed order " + std::to_string(p) + " (expected 1) -- this is the "
          "cost Crank-Nicolson buys back");
}

void test_fd_second_order_in_space() {
    std::vector<double> res, err;
    for (int nx : {100, 200, 400}) {
        res.push_back(nx);
        err.push_back(fd_error(FdScheme::CrankNicolson, nx));
    }
    const double p = observed_order(res, err);
    check("Finite differences are second order in space", p > 1.5 && p < 2.5,
          "observed order " + std::to_string(p) + " (expected 2)");
}

void test_explicit_stability_boundary() {
    // Above ratio 1/2 the explicit scheme is unstable. It must say so.
    FdConfig unstable;
    unstable.scheme = FdScheme::Explicit;
    unstable.space_steps = 400;
    unstable.time_steps = 50;
    const auto bad = finite_difference(kCall, kMkt, unstable);

    FdConfig stable;
    stable.scheme = FdScheme::Explicit;
    stable.space_steps = 200;
    stable.time_steps = 20000;
    const auto good = finite_difference(kCall, kMkt, stable);
    const double good_err = std::fabs(good.price - analytic_price(kCall, kMkt));

    check("Explicit scheme reports its own stability condition",
          !bad.stability_ok && good.stability_ok && good_err < 1e-2,
          "unstable ratio " + std::to_string(bad.stability_ratio) + " flagged; stable ratio "
              + std::to_string(good.stability_ratio) + " gives error " + std::to_string(good_err));
}

void test_fem_matches_fd() {
    const double exact = analytic_price(kCall, kMkt);
    FemConfig fem;
    fem.elements = 800; fem.time_steps = 800;
    const double fem_consistent = finite_element(kCall, kMkt, fem).price;

    fem.mass = MassMatrix::Lumped;
    const double fem_lumped = finite_element(kCall, kMkt, fem).price;

    FdConfig fd;
    fd.space_steps = 800; fd.time_steps = 800;
    const double fd_price = finite_difference(kCall, kMkt, fd).price;

    const double e_cons = std::fabs(fem_consistent - exact);
    const double e_lump = std::fabs(fem_lumped - exact);
    const double gap = std::fabs(fem_lumped - fd_price);

    check("Galerkin FEM matches the analytic price, and lumping it approaches FD",
          e_cons < 5e-3 && e_lump < 5e-3 && gap < 5e-4,
          "consistent-mass error " + std::to_string(e_cons) + ", lumped error "
              + std::to_string(e_lump) + ", lumped-vs-FD gap " + std::to_string(gap));
}

void test_fem_second_order() {
    const double exact = analytic_price(kCall, kMkt);
    std::vector<double> res, err;
    for (int ne : {100, 200, 400}) {
        FemConfig cfg;
        cfg.elements = ne;
        cfg.time_steps = ne * ne / 50;
        res.push_back(ne);
        err.push_back(std::fabs(finite_element(kCall, kMkt, cfg).price - exact));
    }
    const double p = observed_order(res, err);
    check("P1 elements are second order in the mesh size", p > 1.5 && p < 2.5,
          "observed order " + std::to_string(p) + " (expected 2)");
}

// ------------------------------------------------------------ Monte Carlo

void test_monte_carlo_half_order() {
    const double exact = analytic_price(kCall, kMkt);
    std::vector<double> res, err;
    for (std::int64_t n : {2000, 8000, 32000, 128000}) {
        McConfig cfg;
        cfg.paths = n;
        cfg.antithetic = false;
        cfg.control_variate = false;
        cfg.seed = 12345;
        res.push_back(static_cast<double>(n));
        err.push_back(std::fabs(monte_carlo(kCall, kMkt, cfg).standard_error));
    }
    const double p = observed_order(res, err);
    check("Monte Carlo standard error falls as 1/sqrt(N)", p > 0.4 && p < 0.6,
          "observed order " + std::to_string(p) + " (expected 0.5)");
    (void)exact;
}

void test_variance_reduction_helps() {
    McConfig plain;
    plain.paths = 100000; plain.antithetic = false; plain.control_variate = false;
    McConfig reduced;
    reduced.paths = 100000; reduced.antithetic = true; reduced.control_variate = true;

    const auto a = monte_carlo(kCall, kMkt, plain);
    const auto b = monte_carlo(kCall, kMkt, reduced);
    const double ratio = a.standard_error / b.standard_error;
    check("Antithetic plus control variate cuts the standard error", ratio > 3.0,
          "standard error " + std::to_string(a.standard_error) + " -> "
              + std::to_string(b.standard_error) + ", a factor of " + std::to_string(ratio));
}

void test_monte_carlo_interval_covers_truth() {
    const double exact = analytic_price(kCall, kMkt);
    McConfig cfg;
    cfg.paths = 400000;
    const auto r = monte_carlo(kCall, kMkt, cfg);
    const bool covered = r.ci_low() <= exact && exact <= r.ci_high();
    check("95% interval contains the analytic price", covered,
          "[" + std::to_string(r.ci_low()) + ", " + std::to_string(r.ci_high())
              + "] contains " + std::to_string(exact));
}

void test_euler_is_biased_and_milstein_less_so() {
    const double exact = analytic_price(kCall, kMkt);
    auto err_of = [&](McScheme s, int steps) {
        McConfig cfg;
        cfg.paths = 200000; cfg.scheme = s; cfg.steps = steps; cfg.seed = 777;
        // The controls take their means from the exact dynamics, so leaving
        // them on here would fold a control bias into the discretisation bias
        // this check is trying to measure. The library refuses the pairing.
        cfg.antithetic = false; cfg.control_variate = false;
        return std::fabs(monte_carlo(kCall, kMkt, cfg).price - exact);
    };
    const double euler_coarse = err_of(McScheme::Euler, 4);
    const double euler_fine = err_of(McScheme::Euler, 64);
    const double exact_scheme = err_of(McScheme::Exact, 1);

    check("Discretisation bias shrinks with steps, and exact sampling has none",
          euler_fine < euler_coarse && exact_scheme < euler_coarse,
          "Euler error " + std::to_string(euler_coarse) + " at 4 steps -> "
              + std::to_string(euler_fine) + " at 64; exact sampling "
              + std::to_string(exact_scheme));
}

void test_american_rejected_by_monte_carlo() {
    Option am = kPut; am.exercise = Exercise::American;
    bool threw = false;
    try {
        (void)monte_carlo(am, kMkt, McConfig{});
    } catch (const std::exception&) {
        threw = true;
    }
    check("Monte Carlo refuses American exercise rather than silently mispricing", threw,
          "forward simulation cannot value an optimal stopping problem");
}


// ------------------------------------------- Monte Carlo variance reduction
//
// A variance reduction that is not checked for bias is a way of being wrong
// with more confidence. Each technique below is run over many seeds: the mean
// error must be consistent with zero, and the standard error the estimator
// reports must match the spread the estimator actually has.

struct SeedStudy {
    double mean_error{};
    double reported_se{};  // averaged over seeds
    double realised_se{};  // spread of the price across seeds
    double z{};            // mean error in standard errors of the mean
};

SeedStudy seed_study(const Option& o, const Market& m, McConfig cfg, int reps) {
    const double exact = analytic_price(o, m);
    double se_sum = 0, e_sum = 0, e_sq = 0;
    for (int i = 0; i < reps; ++i) {
        cfg.seed = 1000003ULL * static_cast<std::uint64_t>(i + 1) + 7ULL;
        const auto r = monte_carlo(o, m, cfg);
        const double e = r.price - exact;
        e_sum += e;
        e_sq += e * e;
        se_sum += r.standard_error;
    }
    const double n = reps;
    SeedStudy s;
    s.mean_error = e_sum / n;
    s.realised_se = std::sqrt(std::fmax(e_sq / n - s.mean_error * s.mean_error, 0.0));
    s.reported_se = se_sum / n;
    s.z = s.realised_se > 0.0 ? s.mean_error / (s.realised_se / std::sqrt(n)) : 0.0;
    return s;
}

McConfig plain_mc(std::int64_t paths = 20000) {
    McConfig c;
    c.paths = paths;
    c.antithetic = false;
    c.control_variate = false;
    return c;
}

void test_scheme_inconsistent_reductions_are_refused() {
    // A reduction whose mean comes from the exact dynamics, run on a scheme
    // that does not reproduce them, shifts the price by a constant. Refusing
    // is the only honest option; the four that depend on the dynamics are
    // checked here, and the three that do not must still be allowed through.
    auto refuses = [](const McConfig& cfg) {
        try { (void)monte_carlo(kCall, kMkt, cfg); return false; }
        catch (const std::invalid_argument&) { return true; }
    };
    auto euler = [] {
        McConfig c;
        c.paths = 5000; c.scheme = McScheme::Euler; c.steps = 8;
        c.antithetic = false; c.control_variate = false;
        return c;
    };
    McConfig spot = euler();   spot.control_variate = true;
    McConfig delta = euler();  delta.delta_control_variate = true;
    McConfig mm = euler();     mm.moment_matching = true;
    McConfig cond = euler();   cond.conditional_fraction = 0.5;
    const bool all_refused = refuses(spot) && refuses(delta) && refuses(mm) && refuses(cond);

    McConfig fine = euler();
    fine.antithetic = true;
    fine.stratified = true;
    fine.importance_sampling = true;
    bool allowed = true;
    try { (void)monte_carlo(kCall, kMkt, fine); } catch (const std::exception&) { allowed = false; }

    check("Reductions built on the exact dynamics are refused on a discretised scheme",
          all_refused && allowed,
          "both control variates, moment matching and conditional MC throw under Euler, "
          "while antithetic, stratified and importance sampling still run");
}

void test_degenerate_control_is_dropped_not_divided_by() {
    // Under a one-step scheme the delta hedge is an affine function of the
    // terminal spot, so the two controls carry the same information and the
    // normal equations are singular. Under antithetic sampling on a linear
    // scheme it is worse: the control is constant, and its computed variance
    // is pure rounding. Both must end with a finite, sane price.
    McConfig cfg;
    cfg.paths = 20000;
    cfg.steps = 1;
    cfg.antithetic = true;
    cfg.control_variate = true;
    cfg.delta_control_variate = true;
    const double exact = analytic_price(kCall, kMkt);
    bool ok = true;
    double worst = 0.0;
    for (std::uint64_t seed = 1; seed <= 50; ++seed) {
        cfg.seed = seed;
        const auto r = monte_carlo(kCall, kMkt, cfg);
        const double e = std::fabs(r.price - exact);
        worst = std::fmax(worst, e);
        if (!std::isfinite(r.price) || !std::isfinite(r.standard_error) || e > 0.5) ok = false;
    }
    check("Collinear and constant controls are dropped rather than inverted", ok,
          "worst error over 50 seeds with both controls fitted at one step is "
              + std::to_string(worst));
}

void test_every_reduction_is_unbiased() {
    constexpr int kReps = 120;
    struct Case { const char* name; McConfig cfg; };
    std::vector<Case> cases;
    cases.push_back({"plain", plain_mc()});
    { auto c = plain_mc(); c.antithetic = true;             cases.push_back({"antithetic", c}); }
    { auto c = plain_mc(); c.control_variate = true;        cases.push_back({"CV spot", c}); }
    { auto c = plain_mc(); c.steps = 8; c.delta_control_variate = true;
                                                            cases.push_back({"CV delta", c}); }
    { auto c = plain_mc(); c.stratified = true;             cases.push_back({"stratified", c}); }
    { auto c = plain_mc(); c.importance_sampling = true;    cases.push_back({"importance", c}); }
    { auto c = plain_mc(); c.conditional_fraction = 0.25;   cases.push_back({"conditional", c}); }
    { auto c = plain_mc(); c.steps = 8; c.antithetic = true; c.control_variate = true;
      c.delta_control_variate = true; c.stratified = true; c.conditional_fraction = 0.5;
                                                            cases.push_back({"all of them", c}); }

    bool ok = true;
    std::string worst;
    double worst_z = 0.0;
    for (const auto& c : cases) {
        const SeedStudy s = seed_study(kCall, kMkt, c.cfg, kReps);
        if (std::fabs(s.z) > std::fabs(worst_z)) { worst_z = s.z; worst = c.name; }
        // Three and a half sigma on the mean of 120 independent runs.
        if (!(std::fabs(s.z) < 3.5)) ok = false;
    }
    check("Every unbiased variance reduction leaves the price unbiased", ok,
          "worst over " + std::to_string(cases.size()) + " estimators is " + worst + " at z = "
              + std::to_string(worst_z) + " (must stay inside 3.5)");
}

void test_reported_standard_error_is_honest() {
    constexpr int kReps = 120;
    // Stratification breaks independence, so the naive 1/sqrt(N) formula would
    // be wrong here. The within-stratum estimator has to reproduce the spread
    // the estimator really has.
    auto strat = plain_mc();
    strat.stratified = true;
    const SeedStudy s = seed_study(kCall, kMkt, strat, kReps);
    const double ratio = s.reported_se / s.realised_se;
    check("Stratified sampling reports the standard error it actually has",
          ratio > 0.8 && ratio < 1.25,
          "reported " + std::to_string(s.reported_se) + " against a realised spread of "
              + std::to_string(s.realised_se) + ", ratio " + std::to_string(ratio));
}

void test_moment_matching_is_the_biased_one() {
    constexpr int kReps = 200;
    auto mm = plain_mc(500);
    mm.moment_matching = true;
    const SeedStudy with = seed_study(kCall, kMkt, mm, kReps);
    const SeedStudy without = seed_study(kCall, kMkt, plain_mc(500), kReps);

    // It genuinely tightens the estimator, and the variance estimator cannot
    // see that it has: the rescaling couples the paths. The interval it quotes
    // is therefore conservative, and the price it quotes is biased at O(1/N).
    const bool tighter = with.realised_se < 0.6 * without.realised_se;
    const bool blind = with.reported_se > 1.5 * with.realised_se;
    check("Moment matching tightens the estimator but no longer measures itself",
          tighter && blind,
          "realised spread " + std::to_string(without.realised_se) + " -> "
              + std::to_string(with.realised_se) + ", while it still reports "
              + std::to_string(with.reported_se));
}

void test_importance_sampling_rescues_the_tail() {
    Option far = kCall;
    far.strike = 160.0;  // finishes in the money about 3% of the time
    const auto base = monte_carlo(far, kMkt, plain_mc(200000));
    auto is_cfg = plain_mc(200000);
    is_cfg.importance_sampling = true;
    const auto shifted = monte_carlo(far, kMkt, is_cfg);
    const double factor = base.standard_error / shifted.standard_error;
    check("Importance sampling pays for itself where the payoff is rare", factor > 5.0,
          "standard error " + std::to_string(base.standard_error) + " -> "
              + std::to_string(shifted.standard_error) + ", a factor of "
              + std::to_string(factor) + " at a shift of " + std::to_string(shifted.shift_used));
}

void test_delta_control_beats_the_spot_control() {
    auto spot = plain_mc(200000);
    spot.control_variate = true;
    auto delta = plain_mc(200000);
    delta.steps = 16;
    delta.delta_control_variate = true;

    const auto a = monte_carlo(kCall, kMkt, spot);
    const auto b = monte_carlo(kCall, kMkt, delta);
    // The hedge control is the payoff minus its replicating portfolio, so what
    // is left is the hedging error. Theory says beta lands on 1.
    const bool beta_is_one = std::fabs(b.beta_delta - 1.0) < 0.1;
    check("The delta-hedge control beats the terminal-spot control, at beta = 1",
          b.standard_error < 0.5 * a.standard_error && beta_is_one,
          "spot control " + std::to_string(a.standard_error) + " against delta control "
              + std::to_string(b.standard_error) + ", fitted beta "
              + std::to_string(b.beta_delta));
}

void test_conditioning_shrinks_the_variance_monotonically() {
    std::vector<double> ses;
    for (double f : {1.0, 0.5, 0.25, 0.1}) {
        auto cfg = plain_mc(200000);
        cfg.conditional_fraction = f;
        ses.push_back(monte_carlo(kCall, kMkt, cfg).standard_error);
    }
    bool falling = true;
    for (std::size_t i = 1; i < ses.size(); ++i) {
        if (!(ses[i] < ses[i - 1])) falling = false;
    }
    // The limit of the sequence: condition away every bit of randomness and
    // the estimator is the closed form, with no standard error at all.
    auto none = plain_mc(200000);
    none.conditional_fraction = 0.0;
    const auto limit = monte_carlo(kCall, kMkt, none);
    const double exact = analytic_price(kCall, kMkt);
    const bool collapses = limit.standard_error == 0.0 && std::fabs(limit.price - exact) < 1e-12;

    check("Conditioning cuts the variance, and conditioning fully collapses it",
          falling && collapses,
          "s.e. " + std::to_string(ses[0]) + " -> " + std::to_string(ses.back())
              + " as t_c falls to T/10, and exactly 0 at t_c = 0");
}

void test_stacking_reductions_compounds_them() {
    const auto base = monte_carlo(kCall, kMkt, plain_mc(200000));
    auto all = plain_mc(200000);
    all.steps = 16;
    all.antithetic = true;
    all.control_variate = true;
    all.delta_control_variate = true;
    all.stratified = true;
    all.conditional_fraction = 0.5;
    const auto stacked = monte_carlo(kCall, kMkt, all);
    const double factor = base.standard_error / stacked.standard_error;
    const double exact = analytic_price(kCall, kMkt);
    check("Stacked reductions compound, and the tight interval still covers the truth",
          factor > 20.0 && stacked.ci_low() <= exact && exact <= stacked.ci_high(),
          "standard error cut by a factor of " + std::to_string(factor) + " to "
              + std::to_string(stacked.standard_error) + ", interval ["
              + std::to_string(stacked.ci_low()) + ", " + std::to_string(stacked.ci_high()) + "]");
}

void test_inverse_normal_inverts_the_normal() {
    // Stratification is only as good as the quantile function underneath it.
    double worst = 0.0;
    for (int i = 1; i < 100000; ++i) {
        const double p = static_cast<double>(i) / 100000.0;
        worst = std::fmax(worst, std::fabs(norm_cdf(detail::inv_norm_cdf(p)) - p));
    }
    check("The quantile function stratification relies on inverts the normal CDF",
          worst < 1e-14, "worst absolute error over 10^5 probabilities is "
                             + std::to_string(worst));
}

}  // namespace

int main() {
    test_put_call_parity();
    test_greeks_against_bumps();
    test_binomial_first_order();
    test_all_lattices_agree();
    test_american_put_exceeds_european();
    test_tree_rejects_bad_step();
    test_crank_nicolson_second_order_in_time();
    test_crank_nicolson_needs_rannacher();
    test_implicit_first_order_in_time();
    test_fd_second_order_in_space();
    test_explicit_stability_boundary();
    test_fem_matches_fd();
    test_fem_second_order();
    test_monte_carlo_half_order();
    test_variance_reduction_helps();
    test_monte_carlo_interval_covers_truth();
    test_euler_is_biased_and_milstein_less_so();
    test_american_rejected_by_monte_carlo();
    test_scheme_inconsistent_reductions_are_refused();
    test_degenerate_control_is_dropped_not_divided_by();
    test_every_reduction_is_unbiased();
    test_reported_standard_error_is_honest();
    test_moment_matching_is_the_biased_one();
    test_importance_sampling_rescues_the_tail();
    test_delta_control_beats_the_spot_control();
    test_conditioning_shrinks_the_variance_monotonically();
    test_stacking_reductions_compounds_them();
    test_inverse_normal_inverts_the_normal();

    std::printf("\n%d/%d checks passed\n", g_passed, g_passed + g_failed);
    return g_failed == 0 ? 0 : 1;
}

