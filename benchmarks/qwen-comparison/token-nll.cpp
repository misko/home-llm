// Score the same token IDs with each merged Q4_K_M model through pinned llama.cpp.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static double token_nll(const float * logits, int vocabulary, int target) {
    if (target < 0 || target >= vocabulary) throw std::runtime_error("target outside vocabulary");
    double maximum = *std::max_element(logits, logits + vocabulary);
    double total = 0;
    for (int i = 0; i < vocabulary; ++i) total += std::exp(double(logits[i]) - maximum);
    double value = maximum + std::log(total) - logits[target];
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite NLL");
    return value;
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test") {
            float uniform[] = {0, 0, 0, 0};
            if (std::abs(token_nll(uniform, 4, 2) - std::log(4.0)) > 1e-12) return 1;
            float shifted[] = {10000, 10000, 10000, 10000};
            if (std::abs(token_nll(shifted, 4, 2) - std::log(4.0)) > 1e-10) return 1;
            std::cout << "NLL identity and numerical-stability checks passed\n";
            return 0;
        }
        const bool reference = argc == 5 && std::string(argv[1]) == "--reference";
        if (argc != 4 && !reference) throw std::runtime_error("usage: token-nll [--reference] MODEL.gguf TOKENS.uint32 OUTPUT");
        const int offset = reference ? 1 : 0;
        constexpr int context = 512, windows = 256, targets_per_window = 256;
        std::ifstream input(argv[2 + offset], std::ios::binary | std::ios::ate);
        const size_t token_count = reference ? size_t(input.tellg()) / sizeof(llama_token) : context * windows;
        if (reference && (token_count == 0 || token_count > context)) throw std::runtime_error("reference must have 1..512 tokens");
        input.seekg(0);
        std::vector<llama_token> tokens(token_count);
        static_assert(sizeof(llama_token) == sizeof(uint32_t));
        input.read(reinterpret_cast<char *>(tokens.data()), tokens.size() * sizeof(llama_token));
        if (input.gcount() != std::streamsize(tokens.size() * sizeof(llama_token))) throw std::runtime_error("insufficient frozen tokens");
        ggml_backend_load_all();
        bool gpu = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
            gpu = gpu || ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU;
        if (!gpu) throw std::runtime_error("NLL evaluation requires the shared GPU; refusing silent CPU fallback");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = -1;
        auto * model = llama_model_load_from_file(argv[1 + offset], mp);
        if (!model) throw std::runtime_error("model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = context; cp.n_batch = context; cp.n_ubatch = context;
        cp.n_seq_max = 1; cp.n_threads = 4; cp.n_threads_batch = 4;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("context creation failed");
        const int vocabulary = llama_vocab_n_tokens(llama_model_get_vocab(model));
        auto batch = llama_batch_init(context, 0, 1);
        std::vector<double> losses;
        double sum = 0;
        for (int window = 0; window < (reference ? 1 : windows); ++window) {
            llama_memory_clear(llama_get_memory(ctx), true);
            batch.n_tokens = reference ? int(token_count) : context;
            for (int j = 0; j < batch.n_tokens; ++j) {
                auto token = tokens[window * context + j];
                if (token < 0 || token >= vocabulary) throw std::runtime_error("input token outside vocabulary");
                batch.token[j] = token; batch.pos[j] = j;
                batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
                batch.logits[j] = reference ? j == batch.n_tokens - 1 : j >= 255 && j < 511;
            }
            if (llama_decode(ctx, batch) != 0) throw std::runtime_error("decode failed");
            if (reference) {
                const float * logits = llama_get_logits_ith(ctx, -1);
                if (!logits) throw std::runtime_error("reference logits missing");
                std::ofstream out(argv[3 + offset], std::ios::binary);
                out.write(reinterpret_cast<const char *>(logits), vocabulary * sizeof(float));
                out.close();
                if (!out) throw std::runtime_error("reference write failed");
                llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
                return 0;
            }
            double loss = 0;
            for (int j = 255; j < 511; ++j) {
                const float * logits = llama_get_logits_ith(ctx, j);
                if (!logits) throw std::runtime_error("required token logits missing");
                loss += token_nll(logits, vocabulary, tokens[window * context + j + 1]);
            }
            sum += loss; losses.push_back(loss / targets_per_window);
            std::cerr << "NLL window " << window + 1 << "/" << windows << "\n";
        }
        std::ofstream out(argv[3]);
        out << std::setprecision(12) << "{\"target_tokens\":" << windows * targets_per_window
            << ",\"context_tokens\":512,\"nll\":" << sum / (windows * targets_per_window)
            << ",\"window_nll\":[";
        for (size_t i = 0; i < losses.size(); ++i) out << (i ? "," : "") << losses[i];
        out << "]}\n";
        out.close();
        if (!out) throw std::runtime_error("could not persist NLL result");
        llama_batch_free(batch); llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
