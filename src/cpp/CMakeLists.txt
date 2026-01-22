cmake_minimum_required(VERSION 3.15)

# Create the cpgeo shared library
add_library(cpgeo SHARED
    # 源文件
    cpgeo.cpp
    src/triangulation.cpp
    src/sphere_triangulation.cpp
    src/cpgeo_mapping.cpp
    src/space_tree.cpp
    src/tensor.cpp
    src/mesh_utils.cpp
    src/mesh_edge_flip.cpp
    
    # 头文件（让 IDE 能显示和索引）
    include/triangulation.h
    include/sphere_triangulation.h
    include/cpgeo_mapping.h
    include/space_tree.h
    include/tensor.h
    include/mesh_utils.h
    include/mesh_edge_flip.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cpgeo.h
)

# Add alias for consistent naming
add_library(cpgeo::cpgeo ALIAS cpgeo)

# Set library properties
set_target_properties(cpgeo PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "cpgeo"
    PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cpgeo.h
)

# Include directories
target_include_directories(cpgeo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compile definitions
target_compile_definitions(cpgeo
    PRIVATE
        cpgeo_EXPORTS
)

# Compiler warnings
if(MSVC)
    target_compile_options(cpgeo PRIVATE /W4)
else()
    target_compile_options(cpgeo PRIVATE -Wall -Wextra -Wpedantic)
endif()

# C++23 standard
set_property(TARGET cpgeo PROPERTY CXX_STANDARD 23)
set_property(TARGET cpgeo PROPERTY CXX_STANDARD_REQUIRED ON)

# Link OpenMP if found (optional, can be removed if not needed)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpgeo PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "Linking cpgeo with OpenMP")
endif()

# Link libraries (add any dependencies here)
# target_link_libraries(cpgeo PUBLIC/PRIVATE dependency_name)

# ==================== Executable ====================
# Create the cpgeo executable for testing/demo
add_executable(cpgeo_demo
    tests/main.cpp
)

# Include directories for the demo
target_include_directories(cpgeo_demo PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Link the executable with the library
target_link_libraries(cpgeo_demo PRIVATE cpgeo::cpgeo)

# C++20 standard for demo
set_property(TARGET cpgeo_demo PROPERTY CXX_STANDARD 20)
set_property(TARGET cpgeo_demo PROPERTY CXX_STANDARD_REQUIRED ON)

# Set output name
set_target_properties(cpgeo_demo PROPERTIES
    OUTPUT_NAME "cpgeo_demo"
)

# Install the executable
install(TARGETS cpgeo_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)