add_executable(yan_minillama_tests
    tensor_test.cc
    ops_test.cc
    tokenizer_test.cc
    kvcache_test.cc
    radix_tree_test.cc
    context_test.cc
    terminal_test.cc
    chatsession_test.cc
    prompt_build_test.cc
)
target_link_libraries(yan_minillama_tests PRIVATE
    yan_minillama
    GTest::gtest_main
)

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
