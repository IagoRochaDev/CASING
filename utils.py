import numpy as np
from collections import Counter
from draw import *
from sklearn.cluster import DBSCAN



def distF(a, b, r_a, r_b):
    if(r_a < r_b):
        temp = r_a
        r_a = r_b
        r_b = temp

        temp_center = a
        a = b
        b = temp_center


    d = np.linalg.norm(a - b)
    
    extent = max(d + r_a + r_b, 2*r_a)
    overlap = np.clip(r_a + r_b - d, 0, 2*r_b)
    f = extent - overlap + d + r_a - r_b

    if d <= r_a - r_b:
        return f/(4*r_a - 4*r_b)
    elif r_a - r_b < d and d <= r_a + r_b:
        return (f - 4*r_a + 7*r_b)/(3* r_b)
    else:
        return f - 4*r_a + 2*r_b + 3
    

def DBScanCluster(points, eps):
    
    clustering = DBSCAN(eps).fit(points)
    
    plt, ax = createBasicPlot()

    print(f"Number of DBSCAN components: {len(Counter(clustering.labels_).keys())}")

    drawPoints(plt, points, len(Counter(clustering.labels_).keys()), clustering.labels_, size=2, ax=ax)
    return len(Counter(clustering.labels_).keys()), clustering.labels_


##############################   CASING 3D GEOMETRY UTILS


def prepare_casing_normals(pcd, k_neighbors: int = 30):
    """
    Processa a nuvem de pontos para calcular e orientar consistentemente os vetores normais
    caso não estejam presentes.
    """
    import open3d as o3d
    if not pcd.has_normals():
        print(f"\n[CASING Geometry] Estimando vetores normais (k={k_neighbors})...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )
        pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)
        print("-> Normais estimadas e orientadas com sucesso.")
    else:
        print("-> A nuvem de pontos já possui normais.")
    return pcd


def add_gaussian_noise(pcd, std_dev: float = 0.01):
    """Adiciona ruído gaussiano às coordenadas da nuvem de pontos."""
    import open3d as o3d
    points = np.asarray(pcd.points)
    noise = np.random.normal(0, std_dev, size=points.shape)
    pcd.points = o3d.utility.Vector3dVector(points + noise)
    return pcd
