import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from utils import prepare_casing_normals


def computeCASINGDistances(points: np.ndarray, normals: np.ndarray, density: float = 0.0,
                           threshold: float = 0.85, penalty_val: float = 10.0,
                           write: bool = False, filename: str = ""):
    """
    Calcula a matriz de distâncias CASING (Curvature-Aware SING).
    Integra a distância euclidiana 3D com a penalidade por descontinuidade geométrica/curvatura:
        - dot_product > threshold: pontos na mesma superfície suave (penalty = 1.0)
        - dot_product <= threshold: pontos cruzando aresta ou quina viva (penalty = penalty_val)
    Aplica também o fator de densidade herdado do SING original.
    """
    tree = cKDTree(points)
    n_points = len(points)
    lower_tri = []
    dist_mat = np.zeros((n_points, n_points))
    nn = np.empty(n_points)

    # 1. Distância ao vizinho mais próximo (nn) para cada ponto
    for i, point in enumerate(points):
        ds, _ = tree.query(point, 2)
        nn[i] = ds[1]

    # 2. Construção da matriz simétrica e lower triangular para persistência
    for i in range(n_points):
        dists = []
        for j in range(i):
            euclidean_dist = np.linalg.norm(points[i] - points[j])
            dot_product = np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0)

            # Penalidade de Curvatura com Threshold de tolerância a ruído
            penalty = 1.0 if dot_product > threshold else penalty_val

            # Fator de Densidade (SING Original)
            density_factor = 1.0
            if density > 0.0:
                min_nn = min(nn[i], nn[j])
                if min_nn > 1e-12:
                    density_factor = (max(nn[i], nn[j]) / min_nn) ** density

            # Fórmula CASING
            distance = ((euclidean_dist * penalty) / (nn[i] + nn[j])) * density_factor
            dist_mat[i][j] = distance
            dist_mat[j][i] = distance
            dists.append(distance)

        lower_tri.append(dists)

    if write and filename:
        np.savetxt(filename, dist_mat, delimiter=",")

    return dist_mat, lower_tri


def verify_curvature_penalty():
    """
    Teste unitário para validar que a penalidade geométrica por descontinuidade
    impede conexões entre faces com orientações normais distintas.
    """
    print("=" * 60)
    print("TESTE UNITÁRIO: PENALIDADE GEOMÉTRICA CASING")
    print("=" * 60)

    # Pontos A e B na mesma face (normais paralelas)
    pt_a = np.array([0.0, 0.0, 1.0])
    norm_a = np.array([0.0, 0.0, 1.0])

    pt_b = np.array([0.1, 0.0, 1.0])
    norm_b = np.array([0.0, 0.0, 1.0])

    # Ponto C na face adjacente (quina de 90 graus)
    pt_c = np.array([0.1, 0.0, 0.9])
    norm_c = np.array([1.0, 0.0, 0.0])

    pts = np.array([pt_a, pt_b, pt_c])
    norms = np.array([norm_a, norm_b, norm_c])

    dist_mat, _ = computeCASINGDistances(pts, norms, density=0.0)
    dist_ab = dist_mat[0, 1]
    dist_ac = dist_mat[0, 2]

    print(f"Distância CASING mesma face (A-B): {dist_ab:.4f}")
    print(f"Distância CASING quina (A-C):      {dist_ac:.4f}")

    assert dist_ac > dist_ab * 2.0, "Falha na penalização da quina!"
    print("-> SUCESSO: Aresta através da quina devidamente penalizada.")
    print("=" * 60)


if __name__ == "__main__":
    verify_curvature_penalty()
