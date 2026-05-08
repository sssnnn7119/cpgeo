from pathlib import Path
import os


def _get_build_dir(build_root: Path, suffix: str = "") -> Path:
    """Get a platform-specific build directory path."""
    if suffix:
        return build_root.parent / f"{build_root.name}{suffix}"
    return build_root


def _run_cmd(cmd, cwd: Path):
    import subprocess

    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败：{' '.join(cmd)}")


def copy_lib(build_dir: Path, target_dir: Path, is_windows: bool) -> bool:
    """Copy library files from build_dir to target_dir. Returns True if found."""
    import shutil

    if not build_dir.exists():
        return False

    if is_windows:
        names = ["cpgeo.dll"]
    else:
        names = ["cpgeo.dll", "libcpgeo.so", "libcpgeo.dylib"]

    found = False
    for name in names:
        src = build_dir / name
        if src.exists():
            if name.startswith("lib") and name.endswith(".dll"):
                name = "cpgeo.dll"
            dst = target_dir / name
            print(f"复制 \t{src} \n-> \t{dst}")
            shutil.copy2(src, dst)
            found = True
    return found


def get_pe_imports(dll_path: Path) -> list[str]:
    import subprocess

    commands = [
        ["x86_64-w64-mingw32-objdump", "-p", str(dll_path)],
        ["objdump", "-p", str(dll_path)],
    ]

    for cmd in commands:
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
            break
        except (FileNotFoundError, subprocess.CalledProcessError):
            output = None
    if not output:
        return []

    deps = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("DLL Name:"):
            deps.append(line.split(":", 1)[1].strip())
    return deps


def copy_mingw_dependencies(dll_path: Path, target_dir: Path) -> bool:
    import shutil

    mingw_bin = Path(os.environ.get("MINGW_ROOT", "/usr/x86_64-w64-mingw32")) / "bin"
    if not mingw_bin.exists():
        mingw_bin = Path("/usr/x86_64-w64-mingw32/bin")
    if not mingw_bin.exists():
        print("警告：未找到 Mingw-w64 bin 目录。请确保已安装交叉编译工具链。")
        return False

    deps = get_pe_imports(dll_path)
    fallback_deps = ["libgomp-1.dll", "libwinpthread-1.dll", "libgcc_s_seh-1.dll", "libstdc++-6.dll"]
    if not deps:
        deps = fallback_deps.copy()
    else:
        for fd in fallback_deps:
            if fd not in deps:
                deps.append(fd)

    system_dlls = {
        "kernel32.dll",
        "user32.dll",
        "gdi32.dll",
        "advapi32.dll",
        "shell32.dll",
        "ole32.dll",
        "oleaut32.dll",
        "ws2_32.dll",
        "secur32.dll",
        "bcrypt.dll",
        "ntdll.dll",
        "crypt32.dll",
        "comdlg32.dll",
        "version.dll",
        "msvcrt.dll",
        "ucrtbase.dll",
        "imm32.dll",
        "winmm.dll",
        "shlwapi.dll",
    }

    search_dirs = [
        mingw_bin,
        mingw_bin.parent / "lib",
        Path("/usr/lib/gcc/x86_64-w64-mingw32"),
        Path("/usr/lib/gcc/x86_64-w64-mingw32/13-posix"),
        Path("/usr/lib/gcc/x86_64-w64-mingw32/13-win32"),
    ]

    def find_dep_path(dep_name: str) -> Path | None:
        for root in search_dirs:
            if not root.exists():
                continue
            candidate = root / dep_name
            if candidate.exists():
                return candidate
            candidate = next(root.rglob(dep_name), None)
            if candidate and candidate.exists():
                return candidate
        return None

    copied = False
    for dep in deps:
        if dep.lower() in system_dlls:
            continue
        src = find_dep_path(dep)
        if src:
            dst = target_dir / dep
            print(f"复制 Mingw 依赖 \t{src} \n-> \t{dst}")
            shutil.copy2(src, dst)
            copied = True
        else:
            print(f"未找到依赖文件 {dep}，请检查 Mingw-w64 工具链是否安装完整。")
    return copied


def copy_all_libs(compile_mode: str = "Release"):
    import sys

    ROOT = Path(__file__).parent
    TARGET_DIR = ROOT / "src" / "python" / "cpgeo" / "bin"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    is_windows = sys.platform == "win32"
    found_any = False

    if is_windows:
        windows_dirs = [
            ROOT / "build" / "bin",
            ROOT / "build" / "lib",
        ]
        for bd in windows_dirs:
            if copy_lib(bd, TARGET_DIR, is_windows=True):
                found_any = True
    else:
        native_dirs = [
            ROOT / "build" / "lib",
            ROOT / "build" / "bin",
        ]
        for bd in native_dirs:
            if copy_lib(bd, TARGET_DIR, is_windows=False):
                found_any = True

        mingw_dirs = [
            ROOT / "build-mingw" / "bin",
            ROOT / "build-mingw" / "lib",
        ]
        for bd in mingw_dirs:
            if copy_lib(bd, TARGET_DIR, is_windows=False):
                found_any = True

        mingw_dll = TARGET_DIR / "cpgeo.dll"
        if mingw_dll.exists() and copy_mingw_dependencies(mingw_dll, TARGET_DIR):
            print("Mingw-w64 依赖文件复制成功！")

    if not found_any:
        print("警告：未找到编译的库文件！")
        print("请先运行 compile.py 编译 C++ 代码。")
    else:
        print("库文件复制成功！")


def compile_native(compile_mode: str):
    import sys

    ROOT = Path(__file__).parent
    BUILD_DIR = ROOT / "build"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    cmake_mode = compile_mode.capitalize()
    cache_file = BUILD_DIR / "CMakeCache.txt"
    if cache_file.exists():
        cache_file.unlink()

    if sys.platform == "win32":
        _run_cmd(["cmake", ".."], BUILD_DIR)
        _run_cmd(["cmake", "--build", ".", "--config", cmake_mode], BUILD_DIR)
    else:
        _run_cmd(["cmake", "..", f"-DCMAKE_BUILD_TYPE={cmake_mode}", "-DCMAKE_CXX_FLAGS=-fopenmp"], BUILD_DIR)
        _run_cmd(["cmake", "--build", ".", "--target", "cpgeo"], BUILD_DIR)


def compile_mingw64(compile_mode: str):
    ROOT = Path(__file__).parent
    BUILD_DIR = _get_build_dir(ROOT / "build", "-mingw")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    if sys.platform != "linux":
        raise RuntimeError("mingw64 交叉编译仅在 Linux 上支持。")

    cache_file = BUILD_DIR / "CMakeCache.txt"
    if cache_file.exists():
        cache_file.unlink()

    toolchain = ROOT / "cmake" / "toolchain-mingw-w64.cmake"
    if not toolchain.exists():
        raise RuntimeError("未找到 Mingw-w64 工具链文件 cmake/toolchain-mingw-w64.cmake")

    cmake_mode = compile_mode.capitalize()
    _run_cmd([
        "cmake",
        "..",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
        f"-DCMAKE_BUILD_TYPE={cmake_mode}",
        "-DCMAKE_CXX_FLAGS=-fopenmp",
        "-DOpenMP_CXX_FLAGS=-fopenmp",
    ], BUILD_DIR)
    _run_cmd(["cmake", "--build", ".", "--target", "cpgeo"], BUILD_DIR)


if __name__ == "__main__":
    import sys
    import os

    compile_mode = 'release'  # 'Release' or 'Debug'

    build_cache = Path("build/CMakeCache.txt")
    if build_cache.exists():
        build_cache.unlink()

    print("=" * 50)
    print("编译原生库...")
    print("=" * 50)
    try:
        compile_native(compile_mode)
    except RuntimeError as e:
        print(f"错误：{e}")
        exit(1)

    if sys.platform != "win32":
        print()
        print("=" * 50)
        print("在 Linux 下进行 Mingw-w64 交叉编译...")
        print("=" * 50)
        try:
            compile_mingw64(compile_mode)
        except RuntimeError as e:
            print(f"错误：{e}")
            exit(1)

    print()
    print("=" * 50)
    print("复制库文件到 Python 包...")
    print("=" * 50)
    copy_all_libs(compile_mode)
