// arch/canary/token_logprob.h - how sure canary's greedy decode was of each token.
//
// INTERNAL. The decode picks each token by argmax and never normalizes the
// logits, so a run had no score: a clear reading and a guess over noise looked
// the same to the caller. log_softmax(logits)[id] = logits[id] - logsumexp(logits)
// is one pass over the vocabulary (~16k floats, tens of microseconds on CPU), so
// the decode keeps it per token and reports exp() of it as transcribe_token::p.
// It is taken on the raw logits, before any vocabulary bias: it says what the
// model heard, not what the bias chose.

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace transcribe::canary {

// Natural-log probability of `id` under softmax(logits[0, n)). -inf for an id
// out of range or an empty / non-finite row.
inline float token_logprob(const float * logits, int64_t n, int id) {
    if (logits == nullptr || n <= 0 || id < 0 || id >= n) {
        return -std::numeric_limits<float>::infinity();
    }
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int64_t i = 0; i < n; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    if (!std::isfinite(max_logit)) {
        return -std::numeric_limits<float>::infinity();
    }
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        sum += std::exp(static_cast<double>(logits[i] - max_logit));
    }
    return static_cast<float>(static_cast<double>(logits[id] - max_logit) - std::log(sum));
}

}  // namespace transcribe::canary
