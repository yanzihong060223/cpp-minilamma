add_executable(yan_minillama_tests
    tensor_test.cc
    quantized_tensor_test.cc
    load_test.cc
    ops_test.cc
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

add_executable(gguf_run gguf_run.cc)
target_link_libraries(gguf_run PRIVATE yan_minillama)
add_test(NAME GgufInteractive
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gguf_run_test.py"
            "$<TARGET_FILE:gguf_run>" "$<TARGET_FILE:gguf_generate>")
set_tests_properties(GgufInteractive PROPERTIES TIMEOUT 30)

add_executable(gguf_tokenizer_compare gguf_tokenizer_compare.cc)
target_link_libraries(gguf_tokenizer_compare PRIVATE yan_minillama)
add_test(NAME GgufTokenizerExport
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gguf_tokenizer_export_test.py"
            "$<TARGET_FILE:gguf_tokenizer_compare>")
set_tests_properties(GgufTokenizerExport PROPERTIES TIMEOUT 30)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(yan_minillama_tests PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(yan_minillama_tests PRIVATE /W4)
endif()

include(GoogleTest)
gtest_discover_tests(yan_minillama_tests
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 30
    NO_PRETTY_VALUES
    PROPERTIES TIMEOUT 30
)
