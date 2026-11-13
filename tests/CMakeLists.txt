# Each file is an independent process; no GPU state leaks between CTest modes.
foreach(part runtime matmul linear quant ops kv_cache attention forward forward_buffer forward_linear)
    add_executable(test_cuda_${part} test_cuda_${part}.cc)
    target_link_libraries(test_cuda_${part} PRIVATE yan_minillama)
    if(MINI_LLAMA_CUDA)
        target_link_libraries(test_cuda_${part} PRIVATE CUDA::cudart CUDA::cublas)
    endif()
endforeach()
function(cuda_case name part mode)
    add_test(NAME ${name} COMMAND test_cuda_${part} ${mode}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(${name} PROPERTIES LABELS "cuda-validation" TIMEOUT 120)
endfunction()
cuda_case(cuda_runtime runtime runtime)
cuda_case(cuda_matmul matmul matmul)
cuda_case(cuda_linear linear f32)
cuda_case(cuda_quant_q8_0 quant q8_0)
cuda_case(cuda_quant_q4 quant q4)
cuda_case(cuda_forward_quant quant forward)
cuda_case(cuda_forward_q4 quant forward_q4)
cuda_case(cuda_ops ops ops)
cuda_case(cuda_kv_cache kv_cache kv)
cuda_case(cuda_attention attention direct)
cuda_case(cuda_kv_attention attention kv)
cuda_case(cuda_forward_attention attention forward)
cuda_case(cuda_forward forward forward)
cuda_case(cuda_forward_buffer forward_buffer f32)
cuda_case(cuda_forward_device_resident forward_buffer q8_0)
cuda_case(cuda_forward_linear forward_linear linear)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
foreach(script backend_cli cuda_chat_smoke)
    if(script STREQUAL "backend_cli")
        set(case_name backend-cli-smoke)
    else()
        set(case_name cuda-chat-smoke)
    endif()
    add_test(NAME ${case_name}
        COMMAND ${CMAKE_COMMAND} -E env MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_${script}.py
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(${case_name} PROPERTIES LABELS "cuda-validation" TIMEOUT 600)
endforeach()

add_executable(test_bench_counters test_bench_counters.cc)
target_link_libraries(test_bench_counters PRIVATE yan_minillama)
add_test(NAME strict-bench-counters COMMAND test_bench_counters
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME strict-bench-harness
    COMMAND ${CMAKE_COMMAND} -E env MINI_LLAMA_BIN=$<TARGET_FILE:mini-llama>
        ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/test_bench.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(strict-bench-counters strict-bench-harness PROPERTIES TIMEOUT 120)
