// Canary per-token log-probability (pure host, no model). The decode reports
// exp() of it as transcribe_token::p. See src/arch/canary/token_logprob.h.

#include "arch/canary/token_logprob.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void expect(const char * what, bool ok) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++g_failures;
    }
}

bool near(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

}  // namespace

int main() {
    using transcribe::canary::token_logprob;

    // Uniform row: every id is 1/n.
    const std::vector<float> uniform(4, 2.5f);
    expect("uniform", near(token_logprob(uniform.data(), 4, 2), std::log(0.25f)));

    // Matches log(softmax) computed the long way.
    const std::vector<float> row = { 1.0f, 3.0f, -2.0f, 0.5f };
    double sum = 0.0;
    for (float v : row) {
        sum += std::exp(static_cast<double>(v));
    }
    for (int id = 0; id < 4; ++id) {
        const float want = static_cast<float>(row[id] - std::log(sum));
        expect("matches log softmax", near(token_logprob(row.data(), 4, id), want));
    }

    // A clear winner is near 0; a near tie is near log(1/2).
    const std::vector<float> clear = { 30.0f, 0.0f, 0.0f };
    expect("clear winner", token_logprob(clear.data(), 3, 0) > -1e-6f);
    const std::vector<float> tie = { 10.0f, 10.0f, -40.0f };
    expect("near tie", near(token_logprob(tie.data(), 3, 0), std::log(0.5f), 1e-4f));

    // Large logits do not overflow (the max is subtracted first).
    const std::vector<float> large = { 1e4f, 1e4f - 1.0f };
    expect("large logits finite", std::isfinite(token_logprob(large.data(), 2, 1)));

    // Already log-softmaxed input (debug dumps) is a fixed point.
    std::vector<float> logp(row.size());
    for (size_t i = 0; i < row.size(); ++i) {
        logp[i] = static_cast<float>(row[i] - std::log(sum));
    }
    expect("idempotent", near(token_logprob(logp.data(), 4, 1), logp[1]));

    // Out of range / empty -> -inf.
    expect("id out of range", std::isinf(token_logprob(row.data(), 4, 4)));
    expect("negative id", std::isinf(token_logprob(row.data(), 4, -1)));
    expect("empty row", std::isinf(token_logprob(nullptr, 0, 0)));

    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("canary_token_logprob_unit: ok\n");
    return 0;
}
