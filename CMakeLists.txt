cmake_minimum_required(VERSION 3.20)
project(cpp_minilamma LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(YAN_MINILLAMA_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

add_library(yan_minillama STATIC
    src/tensor.cc
    src/ops.cc
    matumal/matuml_ops.cc
)
target_include_directories(yan_minillama PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(yan_minillama PRIVATE -Wall -Wextra -Wpedantic)
elseif(MSVC)
    target_compile_options(yan_minillama PRIVATE /W4)
endif()

if(YAN_MINILLAMA_ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "Sanitizers require GCC or Clang.")
    endif()
    # PUBLIC also enables these options for executables linked to this library.
    target_compile_options(yan_minillama PUBLIC
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(yan_minillama PUBLIC -fsanitize=address,undefined)
endif()

include(CTest)
if(BUILD_TESTING)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
        message(FATAL_ERROR
            "Google Test is missing. Run: git clone --depth 1 --branch v1.18.0 "
            "https://github.com/google/googletest.git third_party/googletest"
        )
    endif()
    set(BUILD_GMOCK OFF CACHE BOOL "Build Google Mock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Install Google Test" FORCE)
    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "Use the shared MSVC runtime" FORCE)
    endif()
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
    add_subdirectory(test)
endif()
