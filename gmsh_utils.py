# gmsh_utils.py
import numpy as np
import gmsh
from pathlib import Path

def gmsh_init(model_name="Brake_Disc_3D"):
    gmsh.initialize()
    gmsh.model.add(model_name)

def gmsh_finalize():
    gmsh.finalize()


def center_geometry_at_origin():
    """Recentre la géométrie importée sur l'origine avant le maillage."""
    entities = gmsh.model.getEntities()
    if not entities:
        return

    xs = []
    ys = []
    zs = []
    for dim, tag in entities:
        try:
            bb = gmsh.model.getBoundingBox(dim, tag)
        except Exception:
            continue
        if bb is None:
            continue
        xs.extend((bb[0], bb[3]))
        ys.extend((bb[1], bb[4]))
        zs.extend((bb[2], bb[5]))

    if not xs or not ys or not zs:
        return

    cx = 0.5 * (min(xs) + max(xs))
    cy = 0.5 * (min(ys) + max(ys))
    cz = 0.5 * (min(zs) + max(zs))

    gmsh.model.occ.translate(entities, -cx, -cy, -cz)
    gmsh.model.occ.synchronize()


def get_surface_z_stats(dim, tag):
    """Retourne la moyenne z, l'étendue en z et les coordonnées d'une surface."""
    try:
        _, coordsSurf_raw, _ = gmsh.model.mesh.getNodes(dim, tag)
        coordsSurf = np.asarray(coordsSurf_raw, dtype=float).reshape(-1, 3)
    except Exception:
        coordsSurf = np.empty((0, 3), dtype=float)

    if coordsSurf.size == 0:
        bb = gmsh.model.getBoundingBox(dim, tag)
        coordsSurf = np.array([[bb[0], bb[1], bb[2]], [bb[3], bb[4], bb[5]]], dtype=float)

    z_values = coordsSurf[:, 2]
    return float(np.mean(z_values)), float(np.ptp(z_values)), coordsSurf


def is_contact_surface(dim, tag, z_min, z_max, tol=None):
    z_mean, z_range, coordsSurf = get_surface_z_stats(dim, tag)
    thickness = abs(z_max - z_min)
    if tol is None:
        tol = max(1e-3, 0.01 * thickness)
    z_tol = max(tol, 0.01 * thickness)

    if z_range < z_tol and (abs(z_mean - z_min) < z_tol or abs(z_mean - z_max) < z_tol):
        return True

    if coordsSurf.shape[0] >= 3:
        v1 = coordsSurf[1] - coordsSurf[0]
        v2 = coordsSurf[2] - coordsSurf[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
            if abs(normal[2]) > 0.75 and (abs(z_mean - z_min) < z_tol or abs(z_mean - z_max) < z_tol):
                return True

    return False


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


def ensure_contact_surface_groups(contact_name="ContactSurface", other_name="OtherSurfaces"):
    """Crée les groupes physiques ContactSurface / OtherSurfaces si nécessaires."""
    if getPhysicalEntities(contact_name):
        return

    z_min = float('inf')
    z_max = -float('inf')
    for dim, tag in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_min = min(z_min, bb[2])
        z_max = max(z_max, bb[5])

    contact_tags = []
    other_tags = []
    for dim, tag in gmsh.model.getEntities(2):
        if is_contact_surface(dim, tag, z_min, z_max):
            contact_tags.append(tag)
        else:
            other_tags.append(tag)

    if contact_tags:
        gmsh.model.addPhysicalGroup(2, contact_tags, tag=100, name=contact_name)
    if other_tags:
        gmsh.model.addPhysicalGroup(2, other_tags, tag=101, name=other_name)


def build_brake_disk_3d_disk1(cl=10):
    step_filename="Frein_velo_1.step"
    print(f"Importing STEP geometry from {step_filename}")

    step_path = Path(__file__).resolve().parent / step_filename
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()

    center_geometry_at_origin()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), cl)

    print("Génération du maillage 3D à partir du STEP...")
    gmsh.model.mesh.generate(3)
    ensure_contact_surface_groups()

    nodeTags, _, _ = gmsh.model.mesh.getNodes()
    elemTypes, elemTags, _ = gmsh.model.mesh.getElements(dim=3)

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


def build_brake_disk_3d_WVA(cl=10):
    step_filename="Frein_velo_Van_Aert.step"
    print(f"Importing STEP geometry from {step_filename}")

    step_path = Path(__file__).resolve().parent / step_filename
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()

    center_geometry_at_origin()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), cl)

    print("Génération du maillage 3D à partir du STEP...")
    gmsh.model.mesh.generate(3)
    ensure_contact_surface_groups()

    nodeTags, _, _ = gmsh.model.mesh.getNodes()
    elemTypes, elemTags, _ = gmsh.model.mesh.getElements(dim=3)

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

def build_brake_disk_3d_Pogi(cl=10):
    step_filename="Frein_velo_Pogacar.step"
    print(f"Importing STEP geometry from {step_filename}")

    step_path = Path(__file__).resolve().parent / step_filename
    gmsh.merge(str(step_path))
    gmsh.model.occ.synchronize()

    center_geometry_at_origin()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), cl)

    print("Génération du maillage 3D à partir du STEP...")
    gmsh.model.mesh.generate(3)
    ensure_contact_surface_groups()

    nodeTags, _, _ = gmsh.model.mesh.getNodes()
    elemTypes, elemTags, _ = gmsh.model.mesh.getElements(dim=3)

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
    ensure_contact_surface_groups()

    nodeTags, _, _ = gmsh.model.mesh.getNodes()
    elemTypes, elemTags, _ = gmsh.model.mesh.getElements(dim=3)

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