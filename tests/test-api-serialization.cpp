#include "server-task.h"
#include <iostream>
#include <stdexcept>

static void check(bool valid, const char * message) {
    if (!valid) throw std::runtime_error(message);
}

int main() {
    try {
        server_task_result_cmpl_final result{};
        result.oaicompat_created = 123456;
        result.oaicompat_cmpl_id = "local-test";
        result.oaicompat_model = "index-translate-2b-exl3";
        result.n_prompt_tokens = 21;
        result.n_prompt_tokens_cache = 0;
        result.n_decoded = 4;
        result.stream = false;
        auto normal = result.to_json_oaicompat();
        check(normal.is_object() && normal["choices"].size() == 1, "Non-stream completion changed");
        check(normal["usage"]["total_tokens"] == 25, "Non-stream usage changed");
        for (bool include : {false, true}) {
            result.stream = true;
            result.include_usage = include;
            for (bool chat : {false, true}) {
                auto chunks = chat ? result.to_json_oaicompat_chat_stream() : result.to_json_oaicompat();
                check(chunks.is_array() && chunks.size() == (include ? 2u : 1u), "Wrong stream chunk count");
                check(chunks[0]["choices"].size() == 1 && chunks[0]["usage"].is_null(), "Usage attached to a choice chunk");
                check(chunks[0]["choices"][0]["finish_reason"] == "length", "Missing final choice");
                for (const auto & chunk : chunks) {
                    check(chunk["created"] == 123456 && chunk["id"] == "local-test", "Stream identity changed");
                }
                if (include) {
                    check(chunks[1]["choices"].empty(), "Usage chunk must have no choices");
                    check(chunks[1]["usage"]["total_tokens"] == 25, "Wrong stream usage");
                }
            }
        }
        server_task_result_cmpl_partial partial{};
        task_result_state state(common_chat_parser_params{});
        state.created = 123456;
        partial.is_begin = true;
        partial.update(state);
        check(partial.oaicompat_created == state.created, "Partial result lost request timestamp");
        partial.oaicompat_cmpl_id = "local-test";
        partial.oaicompat_model = "index-translate-2b-exl3";
        partial.n_decoded = 1;
        auto chunk = partial.to_json_oaicompat();
        check(chunk["usage"].is_null() && chunk["created"] == 123456, "Partial completion identity/usage");
        auto chat_chunks = partial.to_json_oaicompat_chat();
        check(!chat_chunks.empty() && chat_chunks[0]["usage"].is_null() && chat_chunks[0]["created"] == 123456, "Partial chat identity/usage");
        std::cout << "PASS: completion/chat stream usage and request timestamps\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
