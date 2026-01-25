

def copy_dll(compile_mode: str = 'Release'):
    import shutil
    from pathlib import Path

    # 项目根目录
    ROOT = Path(__file__).parent

    # 源文件位置（build 目录）
    BUILD_DIRS = [
        ROOT / "build" / "bin" / compile_mode,
    ]

    # 目标位置（Python 包目录）
    TARGET_DIR = ROOT / "src"/ "python" / "cpgeo" / "bin" 

    # DLL 文件名
    DLL_NAMES = ["cpgeo.dll", "libcpgeo.so", "libcpgeo.dylib"]

    """查找并复制 DLL 到 Python 包目录"""
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    found = False
    for build_dir in BUILD_DIRS:
        if not build_dir.exists():
            continue
        
        for dll_name in DLL_NAMES:
            dll_path = build_dir / dll_name
            if dll_path.exists():
                target_path = TARGET_DIR / dll_name
                print(f"复制 \t{dll_path} \n-> \t{target_path}")
                shutil.copy2(dll_path, target_path)
                found = True
    
    if not found:
        print("警告：未找到编译的 DLL 文件！")
        print("请先运行：")
        print("  mkdir build && cd build")
        print("  cmake .. -DCMAKE_BUILD_TYPE=Release")
        print("  cmake --build . --config Release")
    else:
        print("DLL 复制成功！")


def complile_cpp(compile_mode: str = 'Release'):
    import subprocess
    from pathlib import Path

    # 项目根目录
    ROOT = Path(__file__).parent

    # 构建目录
    BUILD_DIR = ROOT / "build"

    # # 删除旧的构建目录（如果存在）
    # if BUILD_DIR.exists():
    #     import shutil
    #     shutil.rmtree(BUILD_DIR)

    # 创建构建目录
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # 运行 CMake 配置和构建命令
    subprocess.run(["cmake", ".."], cwd=BUILD_DIR, check=True)
    subprocess.run(["cmake", "--build", ".", "--config", compile_mode], cwd=BUILD_DIR, check=True)


if __name__ == "__main__":

    compile_mode = 'Release'  # 'Release' or 'Debug'

    print("编译 C++ 代码...")
    complile_cpp(compile_mode)

    print("复制 DLL 文件...")
    copy_dll(compile_mode)
