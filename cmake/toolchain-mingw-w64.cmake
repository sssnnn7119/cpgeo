# Toolchain file for cross-compiling to Windows (x86_64) using mingw-w64 on Linux
# Usage: cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchain-mingw-w64.cmake

# Target operating system
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Cross compiler executables
set(CMAKE_C_COMPILER x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER x86_64-w64-mingw32-windres)

# Cross compiler prefix
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32)

# Search for programs only in the host system
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for headers and libraries only in the target system
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Static link C++ and GCC runtime libraries to avoid DLL dependencies
# This makes the output .exe/.dll self-contained (no need for libstdc++-6.dll etc.)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -static-libgcc")
