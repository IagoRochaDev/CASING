import argparse
import random
import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import connected_components
import gudhi

# Arquitetura modular preservada idêntica ao projeto original SING:
from input import readPointCloud, generateCubePointCloud
from casing_geometry_experimental import computeCASINGDistances
from main import extractSINGEdges
from compare import compute_persistence_diagram
from draw import drawPointCloudClusters
from utils import prepare_casing_normals, add_gaussian_noise


# ==========================================================
# ETAPA 2 - ANÁLISE DE ROBUSTEZ E CONECTIVIDADE
# ==========================================================
#
# Objetivo:
# Investigar o comportamento do grafo CASING no cubo sintético
# e determinar um epsilon adequado para evitar fragmentações
# internas sem criar conexões entre faces diferentes.
#
# As funções abaixo foram utilizadas durante a investigação
# experimental da Etapa 2.
#
# Resultado principal:
# epsilon = 2.1
#
# Com epsilon = 2.1:
# - 6 componentes foram identificados;
# - cada face possui um único componente;
# - não foram observadas conexões entre faces diferentes;
# - a fragmentação observada na Face 3 foi eliminada;
# - o comportamento permaneceu estável com ruído gaussiano
#   de desvio padrão 0.01.
#
# Os parâmetros fundamentais do CASING permaneceram:
# density = 0.0
# threshold = 0.85
# penalty = 10.0
#
# ==========================================================


def analyze_casing_connectivity(
    points: np.ndarray,
    normals: np.ndarray,
    dist_mat: np.ndarray,
    epsilon: float,
    points_per_face: int = 150
):
    """
    Analisa as conexões do grafo CASING usando o cubo sintético
    como ground truth.

    Cada bloco de points_per_face pontos representa uma das
    6 faces geradas por generateCubePointCloud().
    """

    n_points = len(points)

    # Cada ponto recebe o rótulo da face verdadeira.
    true_faces = np.arange(n_points) // points_per_face

    # Considera somente a parte triangular inferior da matriz.
    rows, cols = np.tril_indices(n_points, k=-1)

    distances = dist_mat[rows, cols]

    # Arestas efetivamente mantidas pelo epsilon.
    edge_mask = distances <= epsilon

    edge_rows = rows[edge_mask]
    edge_cols = cols[edge_mask]

    # Classificação das arestas:
    # mesma face ou faces diferentes.
    same_face_mask = true_faces[edge_rows] == true_faces[edge_cols]
    different_face_mask = ~same_face_mask

    same_face_edges = np.sum(same_face_mask)
    different_face_edges = np.sum(different_face_mask)

    # Analisa as normais.
    normal_dots = np.sum(
        normals[edge_rows] * normals[edge_cols],
        axis=1
    )

    same_face_dot = normal_dots[same_face_mask]
    different_face_dot = normal_dots[different_face_mask]

    print("\n" + "=" * 60)
    print("ANÁLISE DE CONECTIVIDADE CASING")
    print("=" * 60)

    print(f"-> Total de pares analisados: {len(distances)}")
    print(f"-> Arestas mantidas pelo epsilon: {len(distances[edge_mask])}")

    print("\n[ARESTAS]")
    print(f"-> Arestas dentro da mesma face: {same_face_edges}")
    print(f"-> Arestas entre faces diferentes: {different_face_edges}")

    print("\n[NORMAIS - MESMA FACE]")
    if len(same_face_dot) > 0:
        print(f"-> Produto escalar mínimo: {same_face_dot.min():.4f}")
        print(f"-> Produto escalar médio:  {same_face_dot.mean():.4f}")
        print(f"-> Produto escalar máximo: {same_face_dot.max():.4f}")

    print("\n[NORMAIS - FACES DIFERENTES]")
    if len(different_face_dot) > 0:
        print(f"-> Produto escalar mínimo: {different_face_dot.min():.4f}")
        print(f"-> Produto escalar médio:  {different_face_dot.mean():.4f}")
        print(f"-> Produto escalar máximo: {different_face_dot.max():.4f}")

    print("\n[CONEXÕES POR FACE]")

    for face in range(6):
        face_mask = (
            (true_faces[edge_rows] == face) &
            (true_faces[edge_cols] == face)
        )

        print(
            f"-> Face {face}: "
            f"{np.sum(face_mask)} conexões internas"
        )

    print("=" * 60)


def analyze_components_per_face(
    dist_mat: np.ndarray,
    epsilon: float,
    points_per_face: int = 150
):
    """
    Verifica quantos componentes conexos existem dentro de cada
    uma das 6 faces verdadeiras do cubo sintético.
    """

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    print("\n" + "=" * 60)
    print("COMPONENTES POR FACE")
    print("=" * 60)

    for face in range(6):

        start = face * points_per_face
        end = start + points_per_face

        # Matriz de distâncias somente dos pontos desta face
        face_dist = dist_mat[start:end, start:end]

        # Mantém somente as conexões dentro do epsilon
        face_adj = face_dist <= epsilon

        # Remove a diagonal
        np.fill_diagonal(face_adj, False)

        # Grafo não direcionado
        face_graph = csr_matrix(face_adj)

        n_components, labels = connected_components(
            face_graph,
            directed=False,
            return_labels=True
        )

        sizes = np.bincount(labels)

        print(
            f"-> Face {face}: "
            f"{n_components} componente(s) | "
            f"tamanhos: {sorted(sizes, reverse=True)}"
        )

    print("=" * 60)


def analyze_small_components(
    points: np.ndarray,
    dist_mat: np.ndarray,
    epsilon: float,
    points_per_face: int = 150
):
    """
    Identifica componentes pequenos dentro de cada face.

    Mostra os índices e coordenadas dos pontos pertencentes
    a componentes com menos de 10 pontos.
    """

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    print("\n" + "=" * 60)
    print("ANÁLISE DOS COMPONENTES PEQUENOS")
    print("=" * 60)

    for face in range(6):

        start = face * points_per_face
        end = start + points_per_face

        face_dist = dist_mat[start:end, start:end].copy()

        face_adj = face_dist <= epsilon
        np.fill_diagonal(face_adj, False)

        face_graph = csr_matrix(face_adj)

        n_components, labels = connected_components(
            face_graph,
            directed=False,
            return_labels=True
        )

        sizes = np.bincount(labels)

        for component_id, size in enumerate(sizes):

            if size < 10:

                local_indices = np.where(
                    labels == component_id
                )[0]

                global_indices = start + local_indices

                print(
                    f"\n-> Face {face}: componente pequeno "
                    f"com {size} ponto(s)"
                )

                print(
                    f"   Índices globais: "
                    f"{global_indices.tolist()}"
                )

                for idx in global_indices:

                    print(
                        f"   Ponto {idx}: "
                        f"{points[idx]}"
                    )

    print("=" * 60)


def analyze_problematic_points(
    points: np.ndarray,
    dist_mat: np.ndarray,
    epsilon: float,
    problematic_indices: list
):
    """
    Analisa detalhadamente os pontos pertencentes
    ao componente pequeno da Face 3.

    Para cada ponto problemático são exibidos:

    - coordenadas;
    - 10 vizinhos CASING mais próximos;
    - distância CASING;
    - distância Euclidiana;
    - indicação de conexão pelo epsilon.
    """

    print("\n" + "=" * 60)
    print("ANÁLISE DETALHADA DOS PONTOS PROBLEMÁTICOS")
    print("=" * 60)

    for idx in problematic_indices:

        # Considera apenas os outros pontos da Face 3.
        face_start = 450
        face_end = 600

        face_indices = np.arange(
            face_start,
            face_end
        )

        # Remove o próprio ponto.
        face_indices = face_indices[
            face_indices != idx
        ]

        # Ordena pela distância CASING.
        face_distances = dist_mat[
            idx,
            face_indices
        ]

        order = np.argsort(face_distances)

        print(f"\n-> Ponto {idx}")
        print(f"   Coordenadas: {points[idx]}")

        print(
            "\n   10 vizinhos CASING mais próximos "
            "dentro da Face 3:"
        )

        for rank, position in enumerate(
            order[:10],
            start=1
        ):

            neighbor_idx = face_indices[position]

            casing_distance = dist_mat[
                idx,
                neighbor_idx
            ]

            euclidean_distance = np.linalg.norm(
                points[idx] -
                points[neighbor_idx]
            )

            connected = casing_distance <= epsilon

            print(
                f"   {rank:2d}. ponto {neighbor_idx}: "
                f"CASING={casing_distance:.6f} | "
                f"Euclidiana={euclidean_distance:.6f} | "
                f"conectado="
                f"{'SIM' if connected else 'NÃO'}"
            )

    print("=" * 60)


# ==========================================================
# FIM DA ANÁLISE DA ETAPA 2
# ==========================================================


def processPointCloud(
    points: np.ndarray,
    normals: np.ndarray,
    pcd: o3d.geometry.PointCloud,
    epsilon: float = 2.1,
    density: float = 0.0,
    threshold: float = 0.85,
    penalty: float = 10.0,
    shouldDraw: bool = True,
    showNormals: bool = False
):
    """
    Executa o pipeline CASING completo sobre uma nuvem de pontos 3D.

    Nesta versão experimental, além do pipeline normal, são
    executadas as análises utilizadas durante a Etapa 2.
    """

    # 1. Garante que os pontos possuem normais estimadas
    #    se não fornecidas.
    if len(normals) == 0:

        pcd = prepare_casing_normals(
            pcd,
            k_neighbors=15
        )

        normals = np.asarray(
            pcd.normals
        )

    # 2. Matriz de distâncias CASING
    print(
        f"\n[CASING] Calculando matriz com "
        f"{len(points)} pontos "
        f"(density={density}, "
        f"threshold={threshold})..."
    )

    dist_mat, lower_tri = computeCASINGDistances(
        points,
        normals,
        density=density,
        threshold=threshold,
        penalty_val=penalty
    )

    # 3. Extração das arestas estáveis e clustering
    edges, adj_mat = extractSINGEdges(
        dist_mat,
        epsilon=epsilon
    )

    n_components, labels = connected_components(
        csgraph=adj_mat,
        directed=False,
        return_labels=True
    )

    print(
        f"-> Epsilon de corte: {epsilon}"
    )

    print(
        f"-> Arestas mantidas: {len(edges)}"
    )

    print(
        f"-> Componentes/Faces identificadas: "
        f"{n_components}"
    )

    # ==========================================================
    # ETAPA 2 - ANÁLISES EXPERIMENTAIS
    # ==========================================================

    analyze_casing_connectivity(
        points,
        normals,
        dist_mat,
        epsilon=epsilon,
        points_per_face=150
    )

    analyze_components_per_face(
        dist_mat,
        epsilon=epsilon,
        points_per_face=150
    )

    analyze_small_components(
        points,
        dist_mat,
        epsilon=epsilon,
        points_per_face=150
    )

    analyze_problematic_points(
        points,
        dist_mat,
        epsilon=epsilon,
        problematic_indices=[451, 560]
    )

    # ==========================================================
    # FIM DA ETAPA 2
    # ==========================================================

    # 4. Análise de Homologia Persistente
    #    (TDA via Gudhi)
    diag = compute_persistence_diagram(
        dist_mat
    )

    print(
        f"-> Homologia persistente calculada "
        f"({len(diag)} feições)."
    )

    # 5. Visualização 3D
    if shouldDraw:

        drawPointCloudClusters(
            pcd,
            n_components,
            labels,
            window_name=(
                f"CASING 3D - Clusters "
                f"({n_components} componentes, "
                f"eps={epsilon})"
            ),
            show_normals=showNormals
        )

    return (
        points,
        dist_mat,
        lower_tri,
        n_components,
        labels
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "CASING: Curvature-Aware "
            "Stability-Incorporated "
            "Neighborhood Graph for "
            "3D Point Clouds."
        )
    )

    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help=(
            "Caminho para o arquivo da "
            "nuvem de pontos "
            "(.ply, .pcd, .xyz)"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="cube",
        choices=["cube", "custom"],
        help=(
            "Tipo de modelo: 'cube' sintético "
            "ou 'custom' via arquivo"
        )
    )

    parser.add_argument(
        "--pointsPerFace",
        type=int,
        default=150,
        help=(
            "Número de pontos por face para "
            "o modelo do cubo "
            "(padrão: 150 = 900 pontos)"
        )
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help=(
            "Desvio padrão do ruído gaussiano "
            "a injetar (padrão: 0.0)"
        )
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.1,
        help=(
            "Limiar epsilon do grafo CASING "
            "(padrão: 2.1)"
        )
    )

    parser.add_argument(
        "--density",
        type=float,
        default=0.0,
        help=(
            "Expoente de densidade herdado "
            "do SING (padrão: 0.0)"
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help=(
            "Threshold do produto escalar "
            "para descontinuidade normal "
            "(padrão: 0.85)"
        )
    )

    parser.add_argument(
        "--penalty",
        type=float,
        default=10.0,
        help=(
            "Fator de penalidade para "
            "quinas/arestas vivas "
            "(padrão: 10.0)"
        )
    )

    parser.add_argument(
        "--showNormals",
        action="store_true",
        help=(
            "Exibe os vetores normais "
            "na janela 3D do Open3D"
        )
    )

    parser.add_argument(
        "--noDraw",
        action="store_true",
        help=(
            "Desativa a janela gráfica 3D"
        )
    )

    args = parser.parse_args()

    random.seed(1335)
    np.random.seed(1335)

    if args.filename:

        points, normals, pcd = readPointCloud(
            args.filename
        )

    else:

        points, normals, pcd = generateCubePointCloud(
            n_points_per_face=args.pointsPerFace,
            noise_std=args.noise
        )

    processPointCloud(
        points,
        normals,
        pcd,
        epsilon=args.epsilon,
        density=args.density,
        threshold=args.threshold,
        penalty=args.penalty,
        shouldDraw=not args.noDraw,
        showNormals=args.showNormals
    )


if __name__ == "__main__":
    main()