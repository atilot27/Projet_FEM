# gmsh_utils.py
import numpy as np
import gmsh
from pathlib import Path

def gmsh_init(model_name="Brake_Disc_3D"):
    gmsh.initialize()
    gmsh.model.add(model_name)

def gmsh_finalize():
    gmsh.finalize()

def prepare_quadrature_and_basis(elemType, order):
    """Récupère les points d'intégration et les fonctions de forme."""
    rule = f"Gauss{2 * order}"
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)
    _, N, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN

def get_jacobians(elemType, xi, tag=-1):
    """Calcul des Jacobiens pour les éléments."""
    jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
    return jacobians, dets, coords

def getPhysicalEntities(name):
    """Récupère les entités géométriques d'un groupe physique par son nom."""
    entities = []
    try:
        for dim, physTag in gmsh.model.getPhysicalGroups():
            if gmsh.model.getPhysicalName(dim, physTag) == name:
                raw_entities = gmsh.model.getEntitiesForPhysicalGroup(dim, physTag)
                entities = [(dim, int(tag)) for tag in raw_entities]
                break
    except Exception:
        return []
    return entities


def build_brake_disk_3d_disk1(cl=10):
    step_filename="Frein_velo_1.STEP"
    print(f"Importing STEP geometry from {step_filename}")

    step_path = Path(__file__).resolve().parent / step_filename
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), cl)

    print("Génération du maillage 3D à partir du STEP...")
    gmsh.model.mesh.generate(3)

    nodeTags, _, _ = gmsh.model.mesh.getNodes()
    elemTypes, elemTags, _ = gmsh.model.mesh.getElements(dim=3)

    # Classification des surfaces : les deux faces planes du disque sont chauffées.
    contact_tags = []
    other_tags = []
    for dim, tag in gmsh.model.getEntities(2):
        nodeTagsSurf, coordsSurf, _ = gmsh.model.mesh.getNodes(dim, tag)
        coordsSurf = np.asarray(coordsSurf, dtype=float).reshape(-1, 3)

        normal = np.array([0.0, 0.0, 0.0])
        if coordsSurf.shape[0] >= 3:
            v1 = coordsSurf[1] - coordsSurf[0]
            v2 = coordsSurf[2] - coordsSurf[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm

        if abs(normal[2]) > 0.9:
            contact_tags.append(tag)
        else:
            other_tags.append(tag)

    if contact_tags:
        gmsh.model.addPhysicalGroup(2, contact_tags, tag=100, name="ContactSurface")
    if other_tags:
        gmsh.model.addPhysicalGroup(2, other_tags, tag=101, name="OtherSurfaces")

    volumes = [t[1] for t in gmsh.model.getEntities(3)]
    if volumes:
        gmsh.model.addPhysicalGroup(3, volumes, tag=1, name="DiscVolume")

    bnds = [("ContactSurface", 2), ("OtherSurfaces", 2)]
    bnds_tags = []
    for name, dim in bnds:
        actual_tag = -1
        for pt in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, pt[1]) == name:
                actual_tag = pt[1]
                break

        if actual_tag != -1:
            nodes = gmsh.model.mesh.getNodesForPhysicalGroup(dim, actual_tag)[0]
            bnds_tags.append(nodes)
        else:
            bnds_tags.append(np.array([], dtype=int))

    return nodeTags, elemTypes, elemTags, bnds, bnds_tags


def build_brake_disc_3d_basic(cl=10):
    print("Building a simple disk-shaped cylinder geometry")

    # Géométrie simple : disque plat, rayon proche d'un disque de vélo, épaisseur faible.
    radius = 80.0   # rayon en mm
    thickness = 5.0 # épaisseur en mm

    # Création d'un cylindre aligné avec l'axe Z
    gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, thickness, radius)
    gmsh.model.occ.synchronize()

    # Réglage de la taille de maille sur tous les points de la géométrie
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), cl)

    print("Génération du maillage 3D...")
    gmsh.model.mesh.generate(3)

    nodeTags, _, _ = gmsh.model.mesh.getNodes()
    elemTypes, elemTags, _ = gmsh.model.mesh.getElements(dim=3)

    # Classification des surfaces : les deux faces planes du disque sont chauffées.
    contact_tags = []
    other_tags = []
    for dim, tag in gmsh.model.getEntities(2):
        nodeTagsSurf, coordsSurf, _ = gmsh.model.mesh.getNodes(dim, tag)
        coordsSurf = np.asarray(coordsSurf, dtype=float).reshape(-1, 3)
        z_mean = coordsSurf[:, 2].mean() if coordsSurf.size else 0.0

        normal = np.array([0.0, 0.0, 0.0])
        if coordsSurf.shape[0] >= 3:
            v1 = coordsSurf[1] - coordsSurf[0]
            v2 = coordsSurf[2] - coordsSurf[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm

        # Les faces planes supérieures et inférieures ont une normale proche de l'axe Z.
        if abs(normal[2]) > 0.9:
            contact_tags.append(tag)
        else:
            other_tags.append(tag)

    if contact_tags:
        gmsh.model.addPhysicalGroup(2, contact_tags, tag=100, name="ContactSurface")
    if other_tags:
        gmsh.model.addPhysicalGroup(2, other_tags, tag=101, name="OtherSurfaces")

    volumes = [t[1] for t in gmsh.model.getEntities(3)]
    if volumes:
        gmsh.model.addPhysicalGroup(3, volumes, tag=1, name="DiscVolume")

    bnds = [("ContactSurface", 2), ("OtherSurfaces", 2)]
    bnds_tags = []
    for name, dim in bnds:
        actual_tag = -1
        for pt in gmsh.model.getPhysicalGroups(dim):
            if gmsh.model.getPhysicalName(dim, pt[1]) == name:
                actual_tag = pt[1]
                break

        if actual_tag != -1:
            nodes = gmsh.model.mesh.getNodesForPhysicalGroup(dim, actual_tag)[0]
            bnds_tags.append(nodes)
        else:
            bnds_tags.append(np.array([], dtype=int))

    return nodeTags, elemTypes, elemTags, bnds, bnds_tags