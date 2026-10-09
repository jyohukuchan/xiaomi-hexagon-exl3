#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct trace_state {
    std::filesystem::path directory;
    std::string stop;
    bool stopped = false;
};

static bool trace(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & state = *static_cast<trace_state *>(opaque);
    if (state.stopped) return false;
    const std::string name = tensor->name;
    const bool selected = name == "model.input_embed" || name == "inp_embd" ||
        name.rfind("l_out-", 0) == 0 || name.rfind("Qcur", 0) == 0 || name.rfind("Kcur", 0) == 0 ||
        name.rfind("attn_output-", 0) == 0 || name.rfind("attn_norm-", 0) == 0 ||
        (name.size() > 2 && (name.substr(name.size() - 2) == "-0" || name.substr(name.size() - 2) == "-1"));
    if (ask) return selected || name == state.stop;
    if (selected && (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16)) {
        std::vector<unsigned char> data(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
        std::ofstream output(state.directory / (name + ".bin"), std::ios::binary);
        output.write(reinterpret_cast<const char *>(data.data()), data.size());
        if (!output) { std::fprintf(stderr, "Trace write failed: %s\n", name.c_str()); return false; }
        std::ofstream metadata(state.directory / (name + ".json"));
        metadata << "{\"type\":" << int(tensor->type) << ",\"ne\":[";
        for (int d = 0; d < 4; ++d) metadata << (d ? "," : "") << tensor->ne[d];
        metadata << "],\"nb\":[";
        for (int d = 0; d < 4; ++d) metadata << (d ? "," : "") << tensor->nb[d];
        metadata << "]}\n";
        std::printf("TRACE %s %zu\n", name.c_str(), data.size());
        std::fflush(stdout);
    }
    if (name == state.stop) { state.stopped = true; return false; }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 6) { std::fprintf(stderr, "Usage: trace_runtime MODEL DIRECTORY GPU_LAYERS STOP_TENSOR TOKEN...\n"); return 1; }
    trace_state state{argv[2], argv[4]};
    if (std::filesystem::exists(state.directory) && !std::filesystem::is_empty(state.directory)) {
        std::fprintf(stderr, "Trace directory must be empty\n"); return 1;
    }
    std::filesystem::create_directories(state.directory);
    std::vector<llama_token> tokens;
    for (int i = 5; i < argc; ++i) tokens.push_back(std::stoi(argv[i]));
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = std::stoi(argv[3]);
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    cp.cb_eval = trace; cp.cb_eval_user_data = &state;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 3; }
    const int status = llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()));
    std::printf("TRACE_DONE status=%d stopped=%d\n", status, state.stopped);
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return state.stopped ? 0 : 4;
}
