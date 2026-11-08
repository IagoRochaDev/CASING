import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

def createPlot(xlim = 0, ylim = 0, Xlim=1000, Ylim=1000):
    
    fig, ax = plt.subplots()
    plt.xlim([xlim,Xlim])
    plt.ylim([ylim,Ylim])
    ax.set_aspect("equal")
    plt.tight_layout()
    ax.axis("off")

    return plt, ax


def createBasicPlot():
    
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    plt.tight_layout()
    ax.axis("off")

    return plt, ax



def drawEdges(ax, edges, points, color='gray', zorder=0):
    
    for edge in edges:
        x_line=[points[edge[0]][0], points[edge[1]][0]]
        y_line=[points[edge[0]][1], points[edge[1]][1]]
        ax.plot(x_line, y_line, color=color, linewidth = 1, zorder = zorder)


def drawCirclesPerClassNoSpecies(data, points, no_classes, class_per_point, ax, offset = 1):
    color = cm.rainbow(np.linspace(0, 1, no_classes))
    
    for index, el in enumerate(data):

        circle = plt.Circle((el[offset], el[offset + 1]), el[offset + 2], zorder=el[offset + 2], color=color[class_per_point[index]], fill=True, alpha = 0.5)

        ax.add_patch(circle)

    min_x, max_x, min_y, max_y = get_data_bounds(points)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

pointsize = 2

def get_data_bounds( points):
    all_x = []
    all_y = []
    for point in points:
        x, y = point
        all_x.append(x)
        all_y.append(y)
    return min(all_x)-pointsize, max(all_x)+pointsize, min(all_y)-pointsize, max(all_y)+pointsize


def drawPoints(plt, points, n_components, labels, size=0.7, ax=None, flip_y=False):
    color = cm.rainbow(np.linspace(0, 1, n_components))
    np.random.shuffle(color)
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    color_labels = []

    if flip_y:
        max_y = max(y)
        y = [max_y - yi for yi in y]
    
    for index, point in enumerate(points):
        if(n_components==1):
            index=0
        color_labels.append(color[(int)(labels[index])])

    min_x, max_x, min_y, max_y = get_data_bounds(points)

    if flip_y:
        min_y, max_y = max_y, min_y
        
    plt.scatter(x, y, s=size, color=color_labels)
    
    min_x, max_x, min_y, max_y = get_data_bounds(points)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)


##############################   CASING 3D DRAWING


def drawPointCloudClusters(pcd, n_components: int, labels: np.ndarray, window_name: str = "CASING 3D Clusters", show_normals: bool = False):
    """
    Renderiza uma nuvem de pontos 3D no Open3D, atribuindo uma cor distinta para cada cluster.
    """
    import open3d as o3d
    cmap = cm.get_cmap("tab10" if n_components <= 10 else "tab20")
    max_label = max(labels.max(), 1)
    colors = cmap((labels % 20) / 20.0)[:, :3]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
        point_show_normal=show_normals,
        width=1024,
        height=768
    )


