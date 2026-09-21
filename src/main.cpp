// Prices one contract every way the library knows, and prints the error of
// each against the closed form.
//
// The table is the point: the methods do not merely agree, they disagree by
// amounts that match what their discretisation error should be.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "convergence/finite_difference.hpp"
#include "convergence/finite_element.hpp"
#include "convergence/lattice.hpp"
#include "convergence/monte_carlo.hpp"
#include "convergence/option.hpp"

using namespace convergence;

namespace {

void row(const std::string& method, const std::string& config, double price, double exact,
         const std::string& note = "") {
    std::printf("  %-28s %-18s %12.8f  %+11.2e  %s\n", method.c_str(), config.c_str(), price,
                price - exact, note.c_str());
}


// Every technique on the same contract, same path count, same seed, so the
// only thing that changes between rows is the estimator. FACTOR is how much
// the standard error shrinks against plain sampling.
//
// One caveat the table cannot show: antithetic sampling evaluates two paths
// per sample, and conditional Monte Carlo evaluates a Black-Scholes formula
// per sample. Equal path counts are not equal work.
void variance_reduction_table(const std::string& title, const Option& opt, const Market& mkt) {
    const double exact = analytic_price(opt, mkt);
    constexpr std::int64_t kPaths = 200000;

    McConfig base;
    base.paths = kPaths;
    base.antithetic = false;
    base.control_variate = false;
    base.seed = 90210;

    const double se0 = monte_carlo(opt, mkt, base).standard_error;

    std::printf("  %s   %lld paths, exact sampling, closed form %.8f\n", title.c_str(),
                static_cast<long long>(kPaths), exact);
    std::printf("  %-34s %12s  %11s  %10s  %8s\n", "TECHNIQUE", "PRICE", "ERROR", "S.E.",
                "FACTOR");
    std::printf("  %s\n", std::string(82, '-').c_str());

    auto vr_row = [&](const std::string& name, const McConfig& cfg) {
        const auto r = monte_carlo(opt, mkt, cfg);
        std::printf("  %-34s %12.8f  %+11.2e  %10.3e  %7.1fx\n", name.c_str(), r.price,
                    r.price - exact, r.standard_error,
                    r.standard_error > 0.0 ? se0 / r.standard_error : INFINITY);
    };

    vr_row("none (plain sampling)", base);

    { McConfig c = base; c.antithetic = true;
      vr_row("antithetic variates", c); }
    { McConfig c = base; c.control_variate = true;
      vr_row("control variate: terminal spot", c); }
    { McConfig c = base; c.steps = 16; c.delta_control_variate = true;
      vr_row("control variate: delta hedge", c); }
    { McConfig c = base; c.stratified = true;
      vr_row("stratified sampling", c); }
    { McConfig c = base; c.importance_sampling = true;
      vr_row("importance sampling", c); }
    { McConfig c = base; c.moment_matching = true;
      vr_row("moment matching", c); }
    { McConfig c = base; c.conditional_fraction = 0.5;
      vr_row("conditional MC (t_c = T/2)", c); }
    { McConfig c = base; c.conditional_fraction = 0.1;
      vr_row("conditional MC (t_c = T/10)", c); }
    { McConfig c = base; c.steps = 16; c.antithetic = true; c.control_variate = true;
      c.delta_control_variate = true; c.stratified = true; c.conditional_fraction = 0.5;
      vr_row("every unbiased technique at once", c); }

    {
        McConfig c = base;
        c.conditional_fraction = 0.0;
        const auto r = monte_carlo(opt, mkt, c);
        std::printf("  %-34s %12.8f  %+11.2e  %10.3e  %8s\n", "conditional MC (t_c = 0)", r.price,
                    r.price - exact, r.standard_error, "exact");
    }
    std::printf("  (t_c = 0 conditions away all the randomness: the estimator is the closed\n"
                "   form itself, which is the limit every other row is working towards)\n");
    std::printf("  (moment matching reports no gain because the variance estimator cannot see\n"
                "   one: rescaling couples the paths. Its true spread is about 2.5x tighter,\n"
                "   at the cost of an O(1/N) bias, which is why it is off by default)\n");
    std::printf("  (importance sampling below 1.0x is not a bug. The automatic shift centres\n"
                "   the terminal spot on the strike, which is the wrong move for an option\n"
                "   that already finishes in the money half the time)\n\n");
}

}  // namespace

int main() {
    const Option call{100.0, 1.0, OptionType::Call, Exercise::European};
    const Market mkt{100.0, 0.05, 0.20, 0.0};

    const Greeks g = analytic(call, mkt);
    std::printf("European call  S=%.0f K=%.0f T=%.0f r=%.0f%% q=%.0f%% sig=%.0f%%\n\n",
                mkt.spot, call.strike, call.maturity, 100 * mkt.rate, 100 * mkt.dividend,
                100 * mkt.volatility);
    std::printf("  Closed form   %.10f   delta %.6f  gamma %.6f  vega %.6f\n\n", g.price,
                g.delta, g.gamma, g.vega);

    std::printf("  %-28s %-18s %12s  %11s\n", "METHOD", "RESOLUTION", "PRICE", "ERROR");
    std::printf("  %s\n", std::string(80, '-').c_str());

    const double exact = g.price;

    row("Binomial (CRR)", "2000 steps", binomial(call, mkt, 2000).price, exact);
    row("Binomial (Jarrow-Rudd)", "2000 steps",
        binomial(call, mkt, 2000, BinomialModel::JarrowRudd).price, exact);
    row("Binomial (Tian)", "2000 steps",
        binomial(call, mkt, 2000, BinomialModel::Tian).price, exact);
    row("Trinomial (Boyle)", "1000 steps", trinomial(call, mkt, 1000).price, exact);

    {
        FdConfig cfg;
        cfg.scheme = FdScheme::Explicit;
        cfg.space_steps = 200;
        cfg.time_steps = 20000;
        const auto r = finite_difference(call, mkt, cfg);
        char note[64];
        std::snprintf(note, sizeof note, "ratio %.4f %s", r.stability_ratio,
                      r.stability_ok ? "stable" : "UNSTABLE");
        row("FD explicit", "200 x 20000", r.price, exact, note);
    }
    {
        FdConfig cfg;
        cfg.scheme = FdScheme::Implicit;
        cfg.space_steps = 800;
        cfg.time_steps = 800;
        row("FD implicit", "800 x 800", finite_difference(call, mkt, cfg).price, exact);
    }
    {
        FdConfig cfg;
        cfg.scheme = FdScheme::CrankNicolson;
        cfg.space_steps = 800;
        cfg.time_steps = 800;
        cfg.rannacher_steps = 0;
        row("FD Crank-Nicolson", "800 x 800", finite_difference(call, mkt, cfg).price, exact,
            "no start-up");
        cfg.rannacher_steps = 2;
        row("FD Crank-Nicolson", "800 x 800", finite_difference(call, mkt, cfg).price, exact,
            "Rannacher");
    }
    {
        FemConfig cfg;
        cfg.elements = 800;
        cfg.time_steps = 800;
        row("FEM P1 (consistent mass)", "800 elements", finite_element(call, mkt, cfg).price,
            exact);
        cfg.mass = MassMatrix::Lumped;
        row("FEM P1 (lumped mass)", "800 elements", finite_element(call, mkt, cfg).price, exact);
    }
    {
        McConfig cfg;
        cfg.paths = 500000;
        cfg.antithetic = false;
        cfg.control_variate = false;
        const auto r = monte_carlo(call, mkt, cfg);
        char note[80];
        std::snprintf(note, sizeof note, "s.e. %.2e", r.standard_error);
        row("Monte Carlo (plain)", "500k paths", r.price, exact, note);

        cfg.antithetic = true;
        cfg.control_variate = true;
        const auto r2 = monte_carlo(call, mkt, cfg);
        std::snprintf(note, sizeof note, "s.e. %.2e  (%.1fx tighter)", r2.standard_error,
                      r.standard_error / r2.standard_error);
        row("Monte Carlo (anti + CV)", "500k paths", r2.price, exact, note);

        // The controls are turned back off here on purpose: their means come
        // from the exact dynamics, which Euler does not reproduce, so leaving
        // them on would fold a control bias into the row that is supposed to
        // show the discretisation bias alone. The library refuses that
        // combination rather than letting it through.
        cfg.antithetic = false;
        cfg.control_variate = false;
        cfg.scheme = McScheme::Euler;
        cfg.steps = 8;
        const auto r3 = monte_carlo(call, mkt, cfg);
        row("Monte Carlo (Euler)", "500k x 8 steps", r3.price, exact, "discretisation bias");
    }

    std::printf("\n");

    variance_reduction_table("Variance reduction, at the money (K = 100)", call, mkt);

    Option otm = call;
    otm.strike = 160.0;
    variance_reduction_table("Variance reduction, far out of the money (K = 160)", otm, mkt);

    Option amer = call;
    amer.type = OptionType::Put;
    amer.exercise = Exercise::American;
    const double euro_put = analytic_price(Option{100.0, 1.0, OptionType::Put}, mkt);
    const double amer_put = binomial(amer, mkt, 4000).price;
    std::printf("  American put   %.8f   European put %.8f   early exercise %+.8f\n", amer_put,
                euro_put, amer_put - euro_put);
    std::printf("  (no closed form exists for the American put; the lattice is the reference)\n");

    return 0;
}
