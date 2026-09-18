// Prices one contract every way the library knows, and prints the error of
// each against the closed form.
//
// The table is the point: the methods do not merely agree, they disagree by
// amounts that match what their discretisation error should be.

#include <cstdio>
#include <string>

#include "bspde/finite_difference.hpp"
#include "bspde/finite_element.hpp"
#include "bspde/lattice.hpp"
#include "bspde/monte_carlo.hpp"
#include "bspde/option.hpp"

using namespace bspde;

namespace {

void row(const std::string& method, const std::string& config, double price, double exact,
         const std::string& note = "") {
    std::printf("  %-28s %-18s %12.8f  %+11.2e  %s\n", method.c_str(), config.c_str(), price,
                price - exact, note.c_str());
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

        cfg.scheme = McScheme::Euler;
        cfg.steps = 8;
        const auto r3 = monte_carlo(call, mkt, cfg);
        row("Monte Carlo (Euler)", "500k x 8 steps", r3.price, exact, "discretisation bias");
    }

    std::printf("\n");

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
