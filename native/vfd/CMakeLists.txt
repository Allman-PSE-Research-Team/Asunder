cmake_minimum_required(VERSION 3.15)
project(asunder_vfd LANGUAGES CXX)

add_library(asunder_vfd SHARED vfd.cpp vfd_c.cpp)
target_compile_features(asunder_vfd PRIVATE cxx_std_17)
set_target_properties(asunder_vfd PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)
install(TARGETS asunder_vfd RUNTIME DESTINATION . LIBRARY DESTINATION .)
