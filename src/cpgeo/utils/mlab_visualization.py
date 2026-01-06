import torch
import pyvista as pv


def show_quiver3d(R, N, hold=False):
    r = R.detach().cpu().numpy()
    n = N.detach().cpu().numpy()
    points = r.T  # (n, 3)
    vectors = n.T
    plotter = pv.Plotter()
    plotter.add_arrows(points, vectors)
    if not hold:
        plotter.show()


def show_mesh_normal(r, coo, hold=False, color=None):
    normal = torch.cross(r[:, coo[:, 1]] - r[:, coo[:, 0]],
                         r[:, coo[:, 2]] - r[:, coo[:, 0]], dim=0)
    normal = normal / normal.norm(dim=0)
    
    r_mid = (r[:, coo[:, 0]] + r[:, coo[:, 1]] + r[:, coo[:, 2]]) / 3
    r_mid_np = r_mid.detach().cpu().numpy().T
    normal_np = normal.detach().cpu().numpy().T
    plotter = pv.Plotter()
    if color is None:
        plotter.add_arrows(r_mid_np, normal_np)
    else:
        plotter.add_arrows(r_mid_np, normal_np, color=color)
    if not hold:
        plotter.show()


def show_surf(r, coo, hold=False, color=None):
    r_np = r.detach().cpu().numpy().T
    coo_np = coo.detach().cpu().numpy()
    mesh = pv.PolyData(r_np, faces=coo_np)
    plotter = pv.Plotter()
    if color is None:
        plotter.add_mesh(mesh, opacity=1)
    else:
        plotter.add_mesh(mesh, color=color, opacity=1)
    plotter.add_mesh(mesh, style='wireframe', color=(1.0/255, 1.0/255, 1.0/255), opacity=1)
    if not hold:
        plotter.show()


def show_surf2(r,
               coo,
               hold=False,
               color=(232.0 / 255, 232.0 / 255, 232.0 / 255),
               if_points=False):
    r0 = r.clone()[:, coo.unique()]
    r_np = r.detach().cpu().numpy().T
    coo_np = coo.detach().cpu().numpy()
    r_np[:, 2] = 0
    mesh = pv.PolyData(r_np, faces=coo_np)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=color, opacity=1)
    plotter.add_mesh(mesh, style='wireframe', color=(1.0/255, 1.0/255, 1.0/255), opacity=1)
    if if_points:
        r0_np = r0.detach().cpu().numpy().T
        r0_np[:, 2] = 0
        plotter.add_points(r0_np, color=(40.0/255, 120.0/255, 181.0/255), point_size=0.04)
    if not hold:
        plotter.show()


def show_plot(data):
    from matplotlib import pyplot as plt
    d = data.detach().cpu().numpy()
    plt.plot(d)


def show_points(r, hold=False, scale_factor=0.1, color=(1, 0, 0)):
    r_np = r.detach().cpu().numpy().T
    plotter = pv.Plotter()
    plotter.add_points(r_np, color=color, point_size=scale_factor)
    if not hold:
        plotter.show()


def show_points2(r, hold=False, scale_factor=0.1, color=(1, 0, 0)):
    from mayavi import mlab
    r2 = torch.zeros_like(r[0]).tolist()
    r = r.tolist()
    mlab.points3d(r[0], r[1], r2, scale_factor=scale_factor, color=color)
    if not hold:
        mlab.show()
