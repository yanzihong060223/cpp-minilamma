add_executable(yan_minillama_tests
    backend_test.cc
    tensor_test.cc
    quantized_tensor_test.cc
    test_quant.cc
    load_test.cc
    ops_test.cc
    forward_test.cc
    thread_pool_test.cc
    tokenizer_test.cc
    kvcache_test.cc
    radix_tree_test.cc
    context_test.cc
    sampler_test.cc
    terminal_test.cc
    chatsession_test.cc
    prompt_build_test.cc
    gguf_prompt_build_test.cc
    gguf_test.cc
    gguf_load_test.cc
)
target_link_libraries(yan_minillama_tests PRIVATE
    yan_minillama
    GTest::gtest_main
)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_executable(gguf_inspect gguf_inspect.cc)
target_link_libraries(gguf_inspect PRIVATE yan_minillama)

add_executable(gguf_generate gguf_generate.cc)
target_link_libraries(gguf_generate PRIVATE yan_minillama)

add_executable(betch_mark_test betch_mark_test.cc)
target_link_libraries(betch_mark_test PRIVATE yan_minillama)

add_executable(gguf_run gguf_run.cc)
target_link_libraries(gguf_run PRIVATE yan_minillama)
add_test(NAME mini-llama-tests.GgufInteractive
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gguf_run_test.py"
            "$<TARGET_FILE:gguf_run>" "$<TARGET_FILE:gguf_generate>")
set_tests_properties(mini-llama-tests.GgufInteractive PROPERTIES TIMEOUT 30)

add_executable(gguf_tokenizer_compare gguf_tokenizer_compare.cc)
target_link_libraries(gguf_tokenizer_compare PRIVATE yan_minillama)
add_test(NAME mini-llama-tests.GgufTokenizerExport
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gguf_tokenizer_export_test.py"
            "$<TARGET_FILE:gguf_tokenizer_compare>")
set_tests_properties(mini-llama-tests.GgufTokenizerExport PROPERTIES TIMEOUT 30)

add_test(NAME mini-llama-tests.BenchCli
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/bench_cli_test.py"
            "$<TARGET_FILE:mini-llama>")
set_tests_properties(mini-llama-tests.BenchCli PROPERTIES TIMEOUT 60)

# Numerical forward regressions create their own temporary models and dumps.
add_test(NAME mini-llama-tests.golden-logits
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/test_golden.py"
            --binary "$<TARGET_FILE:mini-llama>")
add_test(NAME mini-llama-tests.golden-logits-threaded
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/test_threaded_cli.py"
            --binary "$<TARGET_FILE:mini-llama>")
add_test(NAME mini-llama-tests.golden-logits-compare
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/test_compare_logits.py")
set_tests_properties(
    mini-llama-tests.golden-logits
    mini-llama-tests.golden-logits-threaded
    mini-llama-tests.golden-logits-compare
    PROPERTIES TIMEOUT 120 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(yan_minillama_tests PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(yan_minillama_tests PRIVATE /W4)
endif()

include(GoogleTest)
gtest_discover_tests(yan_minillama_tests
    TEST_PREFIX "mini-llama-tests."
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    NO_PRETTY_VALUES
    PROPERTIES TIMEOUT 30
)
