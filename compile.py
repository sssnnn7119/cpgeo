

from pathlib import Path

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
            dst = target_dir / name
            print(f"复制 \t{src} \n-> \t{dst}")
            shutil.copy2(src, dst)
            found = True
    return found


def copy_all_libs(compile_mode: str = 'Release'):
    import sys
    from pathlib import Path

    ROOT = Path(__file__).parent
    TARGET_DIR = ROOT / "src" / "python" / "cpgeo" / "bin"
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    is_windows = sys.platform == "win32"

    # 清空旧的 bin 目录，避免残留过期文件
    # import shutil
    # for f in TARGET_DIR.iterdir():
    #     if f.is_file():
    #         f.unlink()

    found_any = False

    if is_windows:
        # Windows：只复制 .dll
        cmake_config = compile_mode.capitalize()
        windows_dirs = [
            ROOT / "build" / "bin" / cmake_config,
            ROOT / "build" / "bin",
            ROOT / "build" / "lib",
        ]
        for bd in windows_dirs:
            if copy_lib(bd, TARGET_DIR, is_windows=True):
                found_any = True
    else:
        # Linux：复制原生 .so 
        native_dirs = [
            ROOT / "build" / "lib",
            ROOT / "build" / "bin",
        ]
        for bd in native_dirs:
            if copy_lib(bd, TARGET_DIR, is_windows=False):
                found_any = True


    if not found_any:
        print("警告：未找到编译的库文件！")
        print("请先运行 compile.py 编译 C++ 代码。")
    else:
        print("库文件复制成功！")


def compile_native(compile_mode: str):
    """编译当前平台的库（Linux -> .so, Windows -> .dll, macOS -> .dylib）"""
    import subprocess
    import sys
    from pathlib import Path

    ROOT = Path(__file__).parent
    BUILD_DIR = ROOT / "build"
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    is_windows = sys.platform == "win32"

    if is_windows:
        # Windows（多配置生成器如 Visual Studio），--config 需要首字母大写
        cmake_config = compile_mode.capitalize()
        subprocess.run(["cmake", ".."], cwd=BUILD_DIR, check=True)
        subprocess.run(["cmake", "--build", ".", "--config", cmake_config], cwd=BUILD_DIR, check=True)
    else:
        cmake_mode = compile_mode.capitalize()
        subprocess.run(["cmake", "..", f"-DCMAKE_BUILD_TYPE={cmake_mode}"], cwd=BUILD_DIR, check=True)
        result = subprocess.run(["cmake", "--build", ".", "--target", "cpgeo"], cwd=BUILD_DIR)
        if result.returncode != 0:
            raise RuntimeError("原生库编译失败！")
        result_demo = subprocess.run(["cmake", "--build", ".", "--target", "cpgeo_demo"], cwd=BUILD_DIR)
        if result_demo.returncode != 0:
            print("警告：demo 程序编译失败，但库已编译成功。")


if __name__ == "__main__":

    import sys

    compile_mode = 'release'  # 'Release' or 'Debug'

    import os
    if os.path.exists("build/CMakeCache.txt"):
        os.remove("build/CMakeCache.txt")  # 或者其他构建系统生成的文件，强制 CMake 重新生成构建文件

    # 编译原生库
    print("=" * 50)
    print("编译原生库...")
    print("=" * 50)
    try:
        compile_native(compile_mode)
    except RuntimeError as e:
        print(f"错误：{e}")
        exit(1)

    print()
    print("=" * 50)
    print("复制库文件到 Python 包...")
    print("=" * 50)
    copy_all_libs(compile_mode)
