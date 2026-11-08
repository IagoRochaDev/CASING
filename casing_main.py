import argparse
import random
import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import connected_components
import gudhi

# Arquitetura modular preservada idêntica ao projeto original SING:
from input import readPointCloud, generateCubePointCloud
from casing_geometry import computeCASINGDistances
from main import extractSINGEdges
from compare import compute_persistence_diagram
from draw import drawPointCloudClusters
from utils import prepare_casing_normals, add_gaussian_noise


def processPointCloud(points: np.ndarray, normals: np.ndarray, pcd: o3d.geometry.PointCloud,
                      epsilon: float = 1.8, density: float = 0.0,
                      threshold: float = 0.85, penalty: float = 10.0,
                      shouldDraw: bool = True, showNormals: bool = False):
    """
    Executa o pipeline CASING completo sobre uma nuvem de pontos 3D:
    1. Preparação/garantia de vetores normais
    2. Cálculo da matriz de distâncias curvature-aware (CASING)
    3. Extração das arestas estáveis pelo limiar epsilon
    4. Agrupamento em componentes conexas (clusters / faces geométricas)
    5. Visualização 3D interativa
    """
    # 1. Garante que os pontos possuem normais estimadas se não fornecidas
    if len(normals) == 0:
        pcd = prepare_casing_normals(pcd, k_neighbors=15)
        normals = np.asarray(pcd.normals)

    # 2. Matriz de distâncias CASING
    print(f"\n[CASING] Calculando matriz com {len(points)} pontos (density={density}, threshold={threshold})...")
    dist_mat, lower_tri = computeCASINGDistances(
        points, normals, density=density, threshold=threshold, penalty_val=penalty
    )

    # 3. Extração das arestas estáveis e clustering
    edges, adj_mat = extractSINGEdges(dist_mat, epsilon=epsilon)
    n_components, labels = connected_components(csgraph=adj_mat, directed=False, return_labels=True)

    print(f"-> Epsilon de corte: {epsilon}")
    print(f"-> Arestas mantidas: {len(edges)}")
    print(f"-> Componentes/Faces identificadas: {n_components}")

    # 4. Análise de Homologia Persistente (TDA via Gudhi)
    diag = compute_persistence_diagram(dist_mat)
    print(f"-> Homologia persistente calculada ({len(diag)} feições).")

    # 5. Visualização 3D
    if shouldDraw:
        drawPointCloudClusters(
            pcd, n_components, labels,
            window_name=f"CASING 3D - Clusters ({n_components} componentes, eps={epsilon})",
            show_normals=showNormals
        )

    return points, dist_mat, lower_tri, n_components, labels


def main():
    parser = argparse.ArgumentParser(description="CASING: Curvature-Aware Stability-Incorporated Neighborhood Graph for 3D Point Clouds.")
    parser.add_argument("--filename", type=str, default=None, help="Caminho para o arquivo da nuvem de pontos (.ply, .pcd, .xyz)")
    parser.add_argument("--model", type=str, default="cube", choices=["cube", "custom"], help="Tipo de modelo: 'cube' sintético ou 'custom' via arquivo")
    parser.add_argument("--pointsPerFace", type=int, default=150, help="Número de pontos por face para o modelo do cubo (padrão: 150 = 900 pontos)")
    parser.add_argument("--noise", type=float, default=0.0, help="Desvio padrão do ruído gaussiano a injetar (padrão: 0.0)")
    parser.add_argument("--epsilon", type=float, default=1.8, help="Limiar epsilon do grafo CASING (padrão: 1.8)")
    parser.add_argument("--density", type=float, default=0.0, help="Expoente de densidade herdado do SING (padrão: 0.0)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Threshold do produto escalar para descontinuidade normal (padrão: 0.85)")
    parser.add_argument("--penalty", type=float, default=10.0, help="Fator de penalidade para quinas/arestas vivas (padrão: 10.0)")
    parser.add_argument("--showNormals", action="store_true", help="Exibe os vetores normais na janela 3D do Open3D")
    parser.add_argument("--noDraw", action="store_true", help="Desativa a janela gráfica 3D")

    args = parser.parse_args()
    random.seed(1335)
    np.random.seed(1335)

    if args.filename:
        points, normals, pcd = readPointCloud(args.filename)
    else:
        points, normals, pcd = generateCubePointCloud(n_points_per_face=args.pointsPerFace, noise_std=args.noise)

    processPointCloud(
        points, normals, pcd,
        epsilon=args.epsilon,
        density=args.density,
        threshold=args.threshold,
        penalty=args.penalty,
        shouldDraw=not args.noDraw,
        showNormals=args.showNormals
    )


if __name__ == "__main__":
    main()
