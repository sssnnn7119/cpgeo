cmake_minimum_required(VERSION 3.15)

# Create the cpgeo shared library
add_library(cpgeo SHARED
    # 源文件
    cpgeo.cpp      # 主文件，包含 C API 导出
    src/basisfunc.cpp   # B-spline 基函数
    src/tensor.cpp    # 多维数组实现
    src/mappingfunc.cpp  # 映射函数实现
    src/multidimfunc.cpp
    
    # 头文件（让 IDE 能显示和索引）
    ../../include/cpgeo.h
    include/basisfunc.h
    include/tensor.h
    include/mappingfunc.h
    include/multidimfunc.h
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

# Link OpenMP if found
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

# Link the executable with the library
target_link_libraries(cpgeo_demo PRIVATE cpgeo::cpgeo)

# Set output name
set_target_properties(cpgeo_demo PROPERTIES
    OUTPUT_NAME "cpgeo_demo"
)

# Install the executable
install(TARGETS cpgeo_demo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)