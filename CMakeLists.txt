cmake_minimum_required(VERSION 3.12)
project(vector_search CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The shared library exports only the ves_* C API (faiss is statically linked
# with hidden visibility): everything is hidden unless annotated
# VECTOR_SEARCH_API.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -fvisibility-inlines-hidden")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# GPU support. OFF (default): every GPU entry point is a stub returning -7.
# ON requires a CUDA toolkit and faiss's GPU build; the GPU code is
# written against the modern faiss GPU API (>= 1.7, checked against the 1.14
# headers) and is NOT compiled/validated on this machine.
option(VES_WITH_CUDA "Build GPU paths against the faiss GPU API" OFF)
option(VES_BUILD_TESTS "Build the smoke test" ON)
option(VES_USE_FETCHED_FAISS "Fetch and build faiss from source (needs no system faiss)" ON)
set(VES_FAISS_VERSION "1.14.3" CACHE STRING "faiss tag used by VES_USE_FETCHED_FAISS")

# ---------------------------------------------------------------------------
# faiss
# ---------------------------------------------------------------------------
if(VES_USE_FETCHED_FAISS)
    include(FetchContent)
    FetchContent_Declare(
        faiss
        GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
        GIT_TAG v${VES_FAISS_VERSION}
    )
    set(FAISS_ENABLE_GPU ${VES_WITH_CUDA} CACHE BOOL "" FORCE)
    set(FAISS_ENABLE_CUVS OFF CACHE BOOL "" FORCE)
    # faiss >= 1.13 defaults Apple Metal GPU support ON on Darwin/arm64;
    # it needs the Metal toolchain and is not part of this library's surface.
    set(FAISS_ENABLE_METAL OFF CACHE BOOL "" FORCE)
    set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
    set(FAISS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(FAISS_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(FAISS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
    set(FAISS_ENABLE_RAFT OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    # faiss is statically embedded into libvector_search (the library links
    # no external faiss symbols).
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error")
    FetchContent_MakeAvailable(faiss)
else()
    find_package(faiss REQUIRED)
endif()

if(VES_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# ---------------------------------------------------------------------------
# vector_search
# ---------------------------------------------------------------------------
# faiss 1.6.5 arm64 workaround (see the .cpp); also linked by tests that link
# libfaiss directly and need the symbol resolved.
add_library(ves_aarch64_shim STATIC src/faiss_aarch64_shim.cpp)
target_link_libraries(ves_aarch64_shim PUBLIC faiss)

add_library(vector_search SHARED
    src/vector_search.cpp
    src/ves_search_filter.cpp
)
target_link_libraries(vector_search PRIVATE ves_aarch64_shim)
target_include_directories(vector_search
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(vector_search PRIVATE VECTOR_SEARCH_EXPORTS)
if(VES_WITH_CUDA)
    target_compile_definitions(vector_search PUBLIC VES_WITH_CUDA)
    target_link_libraries(vector_search PRIVATE faiss CUDA::cudart)
else()
    target_link_libraries(vector_search PRIVATE faiss)
endif()
set_target_properties(vector_search PROPERTIES
    OUTPUT_NAME vector_search
    VERSION 0.1.0
    SOVERSION 0
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)
# the C API must be the only visible surface
set_target_properties(vector_search PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(VES_BUILD_TESTS)
    enable_testing()
    add_executable(ves_smoke_test tests/ves_smoke_test.cpp)
    target_link_libraries(ves_smoke_test PRIVATE vector_search)
    add_test(NAME ves_smoke_test COMMAND ves_smoke_test)

    add_executable(ves_filter_crosscheck tests/ves_filter_crosscheck.cpp)
    target_link_libraries(ves_filter_crosscheck PRIVATE vector_search ves_aarch64_shim faiss)
    add_test(NAME ves_filter_crosscheck COMMAND ves_filter_crosscheck)
endif()
