cmake_minimum_required(VERSION 3.15)
project(cpgeo VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found. Parallel features will be disabled.")
endif()

# Optimization flags for Release build
if(MSVC)
    # MSVC: /O2 (默认) 或 /Ox (最大速度优化)
    add_compile_options($<$<CONFIG:Release>:/O2>)
    # 可选：启用内联函数扩展
    add_compile_options($<$<CONFIG:Release>:/Ob2>)
    # 可选：启用全程序优化 (Link Time Optimization)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
    # GCC/Clang: -O3 (最大优化) 或 -Ofast (最激进优化，可能不符合标准)
    add_compile_options($<$<CONFIG:Release>:-O3>)
    # 可选：启用链接时优化
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" OFF)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Platform-specific settings
# Note: CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS disabled to avoid conflicts with LTO
# Use cpgeo_API macro in headers instead

# Add subdirectories
add_subdirectory(src/cpp)

# Installation rules
include(GNUInstallDirs)
install(TARGETS cpgeo
    EXPORT cpgeoTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY src/cpp/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets for CMake package
install(EXPORT cpgeoTargets
    FILE cpgeoTargets.cmake
    NAMESPACE cpgeo::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpgeo
)

# Create package config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpgeoConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cpgeoConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpgeo
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cpgeoConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cpgeoConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cpgeoConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpgeo
)
