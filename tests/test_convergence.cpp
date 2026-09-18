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

#include "bspde/finite_difference.hpp"
#include "bspde/finite_element.hpp"
#include "bspde/lattice.hpp"
#include "bspde/monte_carlo.hpp"
#include "bspde/option.hpp"

using namespace bspde;

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

    std::printf("\n%d/%d checks passed\n", g_passed, g_passed + g_failed);
    return g_failed == 0 ? 0 : 1;
}

