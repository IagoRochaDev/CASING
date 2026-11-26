"""
====================================================================
CASING — GEOMETRY EXPERIMENTAL
ETAPA 2 — INVESTIGAÇÃO DA ROBUSTEZ E DETERMINAÇÃO DO EPSILON
====================================================================

OBJETIVO
--------

Este arquivo reúne a implementação da geometria CASING utilizada
durante os experimentos da Etapa 2.

Diferentemente de `casing_geometry.py`, este arquivo preserva também
a instrumentação utilizada durante a investigação experimental.

O objetivo principal da Etapa 2 foi analisar a conectividade do
grafo CASING no cubo sintético e determinar um valor adequado para
o parâmetro epsilon.

A investigação foi realizada sem alterar a fórmula principal da
métrica CASING.

--------------------------------------------------------------------
CONFIGURAÇÃO INICIAL
--------------------------------------------------------------------

Cubo sintético:
    - 6 faces
    - 150 pontos por face
    - 900 pontos no total

Parâmetros iniciais:

    density  = 0.0
    threshold = 0.85
    penalty   = 10.0
    epsilon   = 1.8

--------------------------------------------------------------------
RESULTADO INICIAL — EPSILON = 1.8
--------------------------------------------------------------------

Com epsilon = 1.8 foram obtidas:

    - 4402 arestas;
    - 7 componentes conexas;
    - 0 conexões entre faces diferentes.

Apesar de não ocorrerem conexões incorretas entre faces, uma das
faces verdadeiras estava fragmentada.

A Face 3 apresentou:

    [148, 2]

ou seja, 148 pontos permaneceram em um componente e 2 pontos
formaram um componente separado.

Os pontos identificados no pequeno componente foram:

    451
    560

--------------------------------------------------------------------
INVESTIGAÇÃO DA DISTRIBUIÇÃO NN
--------------------------------------------------------------------

A distância ao vizinho mais próximo (NN) foi analisada na Face 3
para investigar a origem da fragmentação.

Face 3:

    índices 450 até 599

Foram analisados:

    - mínimo;
    - máximo;
    - média;
    - mediana;
    - P25;
    - P75;
    - P90;
    - P95;
    - os 10 menores valores de NN;
    - pontos específicos envolvidos na fragmentação.

Pontos utilizados na análise detalhada:

    451
    560
    502

--------------------------------------------------------------------
RESULTADOS DA DISTRIBUIÇÃO NN
--------------------------------------------------------------------

Para o cubo sem ruído:

    Mínimo:    0.002204
    Máximo:    0.109989
    Média:     0.042066
    Mediana:   0.039230
    P25:       0.026116
    P75:       0.056129
    P90:       0.072651
    P95:       0.083248

Os menores valores identificados incluíram:

    Ponto 459: NN = 0.002204
    Ponto 527: NN = 0.002204
    Ponto 516: NN = 0.006754
    Ponto 467: NN = 0.006754
    Ponto 451: NN = 0.007495
    Ponto 560: NN = 0.007495
    Ponto 517: NN = 0.007630
    Ponto 482: NN = 0.007630
    Ponto 499: NN = 0.009040
    Ponto 597: NN = 0.009040

--------------------------------------------------------------------
ANÁLISE DOS PONTOS PROBLEMÁTICOS
--------------------------------------------------------------------

Para o ponto 451:

    Coordenadas:
        [1.0, 0.95682571, 0.90955271]

Algumas distâncias CASING observadas:

    451 -> 502
        CASING    = 2.069352
        Euclidiana = 0.077198

    451 -> 541
        CASING    = 2.306206
        Euclidiana = 0.144372

Para o ponto 560:

    Coordenadas:
        [1.0, 0.95290705, 0.91594141]

Algumas distâncias observadas:

    560 -> 541
        CASING    = 2.223840
        Euclidiana = 0.139215

    560 -> 502
        CASING    = 2.247076
        Euclidiana = 0.083828

Um caso importante foi:

    451 -> 560

    CASING     = 0.5
    Euclidiana = 0.007495

Esse comportamento mostrou que o problema não era simplesmente
a ausência de proximidade entre os pontos, mas a forma como as
distâncias CASING estavam sendo comparadas com o epsilon.

--------------------------------------------------------------------
TESTE DE EPSILON = 2.1
--------------------------------------------------------------------

Foi realizado o teste:

    epsilon = 2.1

Resultado:

    - 5867 arestas;
    - 6 componentes;
    - 5867 conexões dentro das faces;
    - 0 conexões entre faces diferentes.

Além disso, cada face passou a apresentar exatamente um componente:

    Face 0: 1 componente — 150 pontos
    Face 1: 1 componente — 150 pontos
    Face 2: 1 componente — 150 pontos
    Face 3: 1 componente — 150 pontos
    Face 4: 1 componente — 150 pontos
    Face 5: 1 componente — 150 pontos

O ponto 451 passou a se conectar ao ponto 502:

    CASING = 2.069352

como:

    2.069352 <= 2.1

Isso eliminou a fragmentação observada com epsilon = 1.8.

--------------------------------------------------------------------
TESTE DE EPSILON = 2.2
--------------------------------------------------------------------

Também foi testado:

    epsilon = 2.2

Resultado:

    - 6429 arestas;
    - 6 componentes;
    - 0 conexões entre faces diferentes.

Entretanto, o resultado de segmentação não apresentou melhoria
em relação a epsilon = 2.1.

Por isso, foi escolhido o menor valor que resolvia o problema:

    epsilon = 2.1

--------------------------------------------------------------------
VALIDAÇÃO COM RUÍDO
--------------------------------------------------------------------

Após a escolha de epsilon = 2.1, foi realizada uma validação com
ruído gaussiano:

    noise = 0.01

Configuração:

    epsilon = 2.1
    density = 0.0
    threshold = 0.85
    penalty = 10.0
    noise = 0.01

Resultado:

    - 6852 arestas;
    - 6 componentes;
    - 0 conexões entre faces diferentes.

Cada face permaneceu como um único componente de 150 pontos.

--------------------------------------------------------------------
CONCLUSÃO DA ETAPA 2
--------------------------------------------------------------------

A investigação mostrou que:

    epsilon = 1.8
        -> preservava a separação entre faces,
           mas produzia fragmentação interna.

    epsilon = 2.1
        -> eliminava a fragmentação;
        -> mantinha as seis faces separadas;
        -> não produzia conexões entre faces diferentes;
        -> permanecia funcional com ruído gaussiano de 0.01.

    epsilon = 2.2
        -> também produzia seis componentes;
        -> não apresentava vantagem sobre 2.1.

Portanto, o valor adotado para a implementação atual foi:

    epsilon = 2.1

A fórmula CASING não foi modificada durante essa investigação.
A alteração realizada na Etapa 2 foi a determinação experimental
de um epsilon mais adequado para o cenário avaliado.

====================================================================
"""

import numpy as np
from scipy.spatial import cKDTree


def computeCASINGDistances(points: np.ndarray, normals: np.ndarray, density: float = 0.0,
                           threshold: float = 0.85, penalty_val: float = 10.0,
                           write: bool = False, filename: str = ""):
    """
    Implementação da métrica CASING utilizada durante os
    experimentos da Etapa 2.

    Esta função é mantida equivalente à implementação final,
    permitindo reproduzir os experimentos documentados neste arquivo.
    """

    tree = cKDTree(points)
    n_points = len(points)
    lower_tri = []
    dist_mat = np.zeros((n_points, n_points))
    nn = np.empty(n_points)

    # ============================================================
    # 1. Distância ao vizinho mais próximo
    # ============================================================

    for i, point in enumerate(points):
        ds, _ = tree.query(point, 2)
        nn[i] = ds[1]

    # ============================================================
    # 2. INSTRUMENTAÇÃO — DISTRIBUIÇÃO NN DA FACE 3
    # ============================================================
    #
    # Este bloco foi utilizado para investigar a fragmentação
    # encontrada com epsilon = 1.8.
    #
    # Ele não altera a métrica.
    # ============================================================

    debug_indices = [451, 560, 502]

    print("\n[ANÁLISE DA DISTRIBUIÇÃO NN - FACE 3]")

    face3_nn = nn[450:600]

    print(f"-> Mínimo:   {np.min(face3_nn):.6f}")
    print(f"-> Máximo:   {np.max(face3_nn):.6f}")
    print(f"-> Média:    {np.mean(face3_nn):.6f}")
    print(f"-> Mediana:  {np.median(face3_nn):.6f}")
    print(f"-> P25:      {np.percentile(face3_nn, 25):.6f}")
    print(f"-> P75:      {np.percentile(face3_nn, 75):.6f}")
    print(f"-> P90:      {np.percentile(face3_nn, 90):.6f}")
    print(f"-> P95:      {np.percentile(face3_nn, 95):.6f}")

    small_nn = np.argsort(face3_nn)[:10]

    print("\n-> 10 menores NN da Face 3:")

    for local_idx in small_nn:
        global_idx = 450 + local_idx

        print(
            f"   Ponto {global_idx}: "
            f"NN={nn[global_idx]:.6f}"
        )

    print("\n[DEBUG NN]")

    for idx in debug_indices:
        print(
            f"-> Ponto {idx}: "
            f"NN = {nn[idx]:.6f}"
        )

    # ============================================================
    # 3. Construção da matriz CASING
    # ============================================================

    for i in range(n_points):
        dists = []

        for j in range(i):
            euclidean_dist = np.linalg.norm(points[i] - points[j])

            dot_product = np.clip(
                np.dot(normals[i], normals[j]),
                -1.0,
                1.0
            )

            # Penalidade geométrica baseada nas normais.
            penalty = 1.0 if dot_product > threshold else penalty_val

            # Fator de densidade herdado do SING.
            density_factor = 1.0

            if density > 0.0:
                min_nn = min(nn[i], nn[j])

                if min_nn > 1e-12:
                    density_factor = (
                        max(nn[i], nn[j]) / min_nn
                    ) ** density

            # Fórmula CASING.
            distance = (
                (euclidean_dist * penalty)
                / (nn[i] + nn[j])
            ) * density_factor

            dist_mat[i][j] = distance
            dist_mat[j][i] = distance
            dists.append(distance)

        lower_tri.append(dists)

    if write and filename:
        np.savetxt(filename, dist_mat, delimiter=",")

    return dist_mat, lower_tri


def verify_curvature_penalty():
    """
    Teste utilizado para validar experimentalmente a penalidade
    geométrica baseada nas orientações das normais.
    """

    print("=" * 60)
    print("TESTE UNITÁRIO: PENALIDADE GEOMÉTRICA CASING")
    print("=" * 60)

    # Pontos A e B na mesma face.
    pt_a = np.array([0.0, 0.0, 1.0])
    norm_a = np.array([0.0, 0.0, 1.0])

    pt_b = np.array([0.1, 0.0, 1.0])
    norm_b = np.array([0.0, 0.0, 1.0])

    # Ponto C na face adjacente.
    pt_c = np.array([0.1, 0.0, 0.9])
    norm_c = np.array([1.0, 0.0, 0.0])

    pts = np.array([
        pt_a,
        pt_b,
        pt_c
    ])

    norms = np.array([
        norm_a,
        norm_b,
        norm_c
    ])

    dist_mat, _ = computeCASINGDistances(
        pts,
        norms,
        density=0.0
    )

    dist_ab = dist_mat[0, 1]
    dist_ac = dist_mat[0, 2]

    print(f"Distância CASING mesma face (A-B): {dist_ab:.4f}")
    print(f"Distância CASING quina (A-C):      {dist_ac:.4f}")

    assert dist_ac > dist_ab * 2.0, (
        "Falha na penalização da quina!"
    )

    print(
        "-> SUCESSO: Aresta através da quina "
        "devidamente penalizada."
    )

    print("=" * 60)


if __name__ == "__main__":
    verify_curvature_penalty()