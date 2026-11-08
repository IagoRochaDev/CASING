import numpy as np

from matplotlib.pyplot import cm


def readStipplefile(file_name, index = 0):
    points = []
    xs = []
    ys = []
    i = 0
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip().split()
            if(len(line)):
                data = [eval(i) for i in line]
                xs.append(data[index])
                ys.append(data[index + 1])
                
                points.append(np.array([data[index], data[index + 1]]))

            i+=1

    print(f"Input length: {len(points)}")
    return points, xs, ys


def readBasicDiskFile(filename):

    file = open(filename)

    # ignore for now

    all_data = []
    xs = []
    ys = []
    radii = []
    points = []

    lines = file.readlines()

    for line in lines:
        line = line.strip().split()
        data = [eval(i) for i in line]

        xs.append(data[0])
        ys.append(data[1])
        point = np.array([data[0], data[1]])
        points.append(point)
        radii.append(data[2])
        all_data.append(data)

    print(f"Input length: {len(points)}")

    return all_data, xs, ys, radii, points

def readDiskFile(filename):

    file = open(filename)

    header = file.readline().strip()

    header_info = header.split(" ")

    species_labels = [eval(i) for i in header_info[1:]]

    global color
    color = cm.rainbow(np.linspace(0, 1, int(header_info[0])))


    # ignore for now
    simulation = file.readline().strip()

    all_data = []
    species = []
    xs = []
    ys = []
    radii = []
    colours = []
    points = []

    lines = file.readlines()

    for line in lines:
        line = line.strip().split()
        data = [eval(i) for i in line]

        species.append(data[0])
        xs.append(data[1])
        ys.append(data[2])
        point = np.array([data[1], data[2]])
        points.append(point)
        radii.append(data[3])
        colours.append(color[species_labels.index(data[0])])
        data.append(color[species_labels.index(data[0])])
        all_data.append(data)

    print(f"Input length: {len(points)}")

    return all_data, species, xs, ys, radii, colours, points, species_labels


##############################   OUT


def save_labels(filename, points, labels):
    with open(filename+"_clustered.txt", 'w') as file:
        for index, point in enumerate(points):
            data = [str(point[0]), str(point[1]), str(labels[index])]
            file.writelines(' '.join(data))
            file.write("\n")  


def save_per_cluster(filename, points, labels, no_labels):
    clustered_points = []

    for index_lb in range(no_labels):
        clustered_points.append([])
        for index, point in enumerate(points):
            if labels[index] == index_lb:
                data = [str(point[0]), str(point[1])]
                clustered_points[index_lb].append(data)
        
        
        
    for index_lb in range(no_labels):
        with open(filename+"_cluster_"+str(index_lb)+".txt", 'w') as file:
            for point in clustered_points[index_lb]:            
                file.writelines(' '.join(point))
                file.write("\n")  


##############################   CASING 3D INPUT


def readPointCloud(filename: str):
    """
    Carrega uma nuvem de pontos 3D usando Open3D a partir de um arquivo (.ply, .pcd, .xyz, etc.).
    Retorna os pontos (Nx3), normais (Nx3) e o objeto PointCloud.
    """
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(filename)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else np.empty((0, 3))
    print(f"PointCloud carregada de {filename}: {len(points)} pontos.")
    return points, normals, pcd


def generateCubePointCloud(n_points_per_face: int = 150, noise_std: float = 0.0):
    """
    Gera uma nuvem de pontos sintética para um cubo 3D [0, 1]^3 com 6 faces bem definidas
    e normais analíticas unitárias.
    Permite adicionar opcionalmente ruído gaussiano para simulação de dados de scanner.
    """
    import open3d as o3d
    faces_pts = []
    faces_normals = []

    face_configs = [
        (2, 0.0, [0.0, 0.0, -1.0]),  # Face Z = 0
        (2, 1.0, [0.0, 0.0, 1.0]),   # Face Z = 1
        (0, 0.0, [-1.0, 0.0, 0.0]),  # Face X = 0
        (0, 1.0, [1.0, 0.0, 0.0]),   # Face X = 1
        (1, 0.0, [0.0, -1.0, 0.0]),  # Face Y = 0
        (1, 1.0, [0.0, 1.0, 0.0]),   # Face Y = 1
    ]

    for axis, val, norm in face_configs:
        pts = np.random.uniform(0.02, 0.98, size=(n_points_per_face, 3))
        pts[:, axis] = val
        if noise_std > 0.0:
            noise = np.random.normal(0, noise_std, size=pts.shape)
            pts += noise
        faces_pts.append(pts)
        faces_normals.append(np.tile(norm, (len(pts), 1)))

    points = np.vstack(faces_pts)
    normals = np.vstack(faces_normals)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    print(f"Cubo 3D sintético gerado: {len(points)} pontos ({n_points_per_face} por face).")
    return points, normals, pcd
