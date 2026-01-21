import numpy as np
from cpgeo.capi import get_mesh_edges, extract_boundary_loops

def test_extract_edges():
    """测试提取网格边的功能"""
    # 创建一个简单的三角形网格：两个三角形形成一个四边形
    # 顶点坐标（用于参考）：
    # 0: (0,0), 1: (1,0), 2: (1,1), 3: (0,1)
    triangles = np.array([
        [0, 1, 2],  # 三角形1
        [0, 2, 3]   # 三角形2
    ], dtype=np.int32)

    print("输入三角形网格:")
    print(triangles)

    # 提取边
    edges = get_mesh_edges(triangles)
    print("\n提取的边 (顶点1, 顶点2, 计数):")
    print(edges)

    # 验证结果
    # 边应该是：
    # (0,1): 1次
    # (0,2): 2次 (共享边)
    # (0,3): 1次
    # (1,2): 1次
    # (2,3): 1次
    expected_edges = [
        [0, 1, 1],
        [0, 2, 2],
        [0, 3, 1],
        [1, 2, 1],
        [2, 3, 1]
    ]
    print("\n期望的边:")
    print(np.array(expected_edges))

def test_extract_boundary_loops():
    """测试提取边界环的功能"""
    # 使用相同的三角形网格
    triangles = np.array([
        [0, 1, 2],  # 三角形1
        [0, 2, 3]   # 三角形2
    ], dtype=np.int32)

    print("\n输入三角形网格:")
    print(triangles)

    # 提取边界环
    boundary_loops = extract_boundary_loops(triangles)
    print("\n提取的边界环:")
    for i, loop in enumerate(boundary_loops):
        print(f"环 {i+1}: {loop}")

    # 验证结果
    # 边界环应该是 [0, 1, 2, 3] 或类似的顺序
    print("\n期望的边界环: [0, 1, 2, 3] 或逆时针顺序")

if __name__ == "__main__":
    test_extract_edges()
    test_extract_boundary_loops()
