# CASING: Curvature-Aware Stability-Incorporated Neighborhood Graph for 3D Point Clouds

Extensão do algoritmo **SING** (*Stability-Incorporated Neighborhood Graph*, SIGGRAPH Asia 2024) para processamento, segmentação e análise topológica (TDA) de **nuvens de pontos 3D**, incorporando consciência geométrica e de curvatura (*curvature-aware*).

---

## 📌 Contexto e Motivação

O algoritmo SING original foi concebido para espaços 2D com métricas euclidianas e de círculos. Quando aplicado diretamente a nuvens de pontos 3D que representam superfícies com quinas vivas (como um cubo com 6 faces distintas amostrado uniformemente):
- A métrica euclidiana normalizada por vizinhos mais próximos $\frac{d(a, b)}{nn(a) + nn(b)}$ é **cega à orientação geométrica da superfície**.
- Em uma amostragem uniforme, a densidade local $nn$ é praticamente constante em todas as faces e bordas.
- O SING original conecta pontos através de quinas de 90°, fundindo todo o objeto em um único cluster ou gerando partições desprovidas de significado geométrico.

O **CASING** resolve essa limitação introduzindo uma **penalidade por descontinuidade normal/curvatura**:
$$d_{\text{CASING}}(a, b) = \left( \frac{\|a - b\| \cdot P(a, b)}{nn(a) + nn(b)} \right) \cdot \left( \frac{\max(nn(a), nn(b))}{\min(nn(a), nn(b))} \right)^{\text{density}}$$

Onde a penalidade $P(a, b)$ analisa o produto escalar das normais $\vec{n}_a \cdot \vec{n}_b$:
- **Mesma superfície suave** ($\vec{n}_a \cdot \vec{n}_b > \tau$): $P(a, b) = 1.0$ (preserva a conectividade intra-face).
- **Descontinuidade/Quina viva** ($\vec{n}_a \cdot \vec{n}_b \le \tau$): $P(a, b) = \text{penalty}$ (por exemplo, 10.0), cortando qualquer aresta no limiar $\epsilon$.

---

## 🏗️ Arquitetura Modular

O projeto segue estritamente a mesma arquitetura e separação de responsabilidades do repositório SING original:

```
CASING/
├── input.py            # Leitura e geração de dados (stipples, disks, nuvens de pontos 3D e cubos sintéticos)
├── utils.py            # Estimativa de normais, injeção de ruído e métricas auxiliares
├── casing_geometry.py  # Cálculo da matriz de distâncias CASING e teste de penalidade geométrica
├── compare.py          # Homologia persistente (TDA via RipsComplex do Gudhi)
├── draw.py             # Visualização 2D (matplotlib) e 3D interativa (Open3D)
├── main.py             # Ponto de entrada do SING original (2D)
├── casing_main.py      # Ponto de entrada do CASING (3D)
├── requirements.txt    # Dependências do projeto (Gudhi, Open3D, SciPy, Matplotlib, etc.)
└── README.md           # Esta documentação
```

---

## ⚙️ Dependências

Certifique-se de ter o ambiente virtual configurado e instale os pacotes:

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### 1. Execução do CASING no Cubo Sintético 3D (Padrão)

Gera um cubo 3D, calcula as normais, executa a matriz CASING, a persistência TDA via Gudhi e abre a visualização colorida no Open3D:

```bash
python casing_main.py
```

### 2. Parâmetros da Linha de Comando (`casing_main.py`)

| Argumento | Tipo | Padrão | Descrição |
| :--- | :---: | :---: | :--- |
| `--filename` | `str` | `None` | Caminho para arquivo 3D (`.ply`, `.pcd`, `.xyz`). Se omitido, usa o cubo sintético |
| `--pointsPerFace` | `int` | `150` | Quantidade de pontos amostrados por face do cubo sintético |
| `--noise` | `float` | `0.0` | Desvio padrão de ruído gaussiano adicionado à nuvem |
| `--epsilon` | `float` | `1.8` | Limiar de corte $\epsilon$ para formação de arestas no grafo |
| `--threshold` | `float` | `0.85` | Limiar do produto escalar entre normais (tolerância de ângulo ~31°) |
| `--penalty` | `float` | `10.0` | Fator de penalidade aplicado a conexões que cruzam quinas |
| `--density` | `float` | `0.0` | Expoente do fator de densidade do SING original |
| `--showNormals` | `flag` | `False` | Exibe os vetores normais na janela 3D do Open3D |
| `--noDraw` | `flag` | `False` | Executa o pipeline em modo headless sem abrir janela gráfica |

### 3. Exemplos de Uso

- **Com exibição de normais:**
  ```bash
  python casing_main.py --showNormals
  ```

- **Simulando scan com ruído gaussiano leve:**
  ```bash
  python casing_main.py --noise 0.005 --epsilon 1.8
  ```

- **Processando uma nuvem de pontos externa:**
  ```bash
  python casing_main.py --filename caminho/para/modelo.ply
  ```

- **Validando o teste unitário de penalidade geométrica:**
  ```bash
  python casing_geometry.py
  ```

---

## 📊 Resultados Esperados

- **SING Original**: Conecta as quinas indistintamente e funde o cubo em 1 único bloco ou em componentes dispersos.
- **CASING**: A penalidade associada às normais ortogonais quebra as conexões nas esquinas, permitindo que o grafo identifique e segmente as **faces geométricas** com cores distintas e preserve a estabilidade topológica.
