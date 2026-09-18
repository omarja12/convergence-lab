// Emits the convergence study behind the project page, as CSV on stdout.
//
// The resolutions here deliberately match tests/test_convergence.cpp exactly.
// Fitting an order is sensitive to which points you fit it on, so sampling a
// different range here would put a second, disagreeing set of numbers on the
// website. One study, one set of figures, quoted everywhere.
//
// The chart on the website is drawn from this output. Regenerate it with
//
//     ./build/convergence_study > docs/convergence.csv
//
// so that nothing shown to a reader is a number somebody typed in by hand.

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

const Option kCall{100.0, 1.0, OptionType::Call, Exercise::European};
const Market kMkt{100.0, 0.05, 0.20, 0.0};

void emit(const std::string& series, double resolution, double error) {
    std::printf("%s,%.0f,%.12e\n", series.c_str(), resolution, error);
}

}  // namespace

int main() {
    const double exact = analytic_price(kCall, kMkt);
    std::printf("series,resolution,error\n");

    // Lattices: error against number of steps.
    for (int n : {250, 500, 1000, 2000}) {
        emit("Binomial CRR", n,
             std::fabs(binomial(kCall, kMkt, n, BinomialModel::CoxRossRubinstein).price - exact));
    }

    // Time discretisation, space held fine so the time error dominates.
    for (int nt : {10, 20, 40, 80}) {
        FdConfig implicit_cfg;
        implicit_cfg.scheme = FdScheme::Implicit;
        implicit_cfg.space_steps = 2000;
        implicit_cfg.time_steps = nt;
        emit("Implicit Euler", nt,
             std::fabs(finite_difference(kCall, kMkt, implicit_cfg).price - exact));

        FdConfig cn_plain;
        cn_plain.scheme = FdScheme::CrankNicolson;
        cn_plain.space_steps = 2000;
        cn_plain.time_steps = nt;
        cn_plain.rannacher_steps = 0;
        emit("Crank-Nicolson (no start-up)", nt,
             std::fabs(finite_difference(kCall, kMkt, cn_plain).price - exact));

        FdConfig cn_rann = cn_plain;
        cn_rann.rannacher_steps = 2;
        emit("Crank-Nicolson + Rannacher", nt,
             std::fabs(finite_difference(kCall, kMkt, cn_rann).price - exact));
    }

    // Spatial discretisation, dt ~ dx^2 so the time term cannot mask it.
    for (int nx : {100, 200, 400}) {
        FdConfig fd;
        fd.scheme = FdScheme::CrankNicolson;
        fd.space_steps = nx;
        fd.time_steps = nx * nx / 50;
        emit("Finite difference (space)", nx,
             std::fabs(finite_difference(kCall, kMkt, fd).price - exact));

        FemConfig fem;
        fem.elements = nx;
        fem.time_steps = nx * nx / 50;
        emit("P1 finite elements", nx,
             std::fabs(finite_element(kCall, kMkt, fem).price - exact));
    }

    // Monte Carlo: the standard error, which is the honest measure of a
    // stochastic method. A single realised error is a draw, not a rate.
    for (std::int64_t n : {2000, 8000, 32000, 128000}) {
        McConfig cfg;
        cfg.paths = n;
        cfg.antithetic = false;
        cfg.control_variate = false;
        cfg.seed = 12345;
        emit("Monte Carlo std. error", static_cast<double>(n),
             monte_carlo(kCall, kMkt, cfg).standard_error);
    }

    return 0;
}
