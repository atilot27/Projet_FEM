# gmsh_utils.py
import numpy as np
import gmsh

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
    dimTags = gmsh.model.getEntitiesForPhysicalName(name)
    if not dimTags:
        return []
    return dimTags

def classify_surface_entities():
    """
    Identifie les surfaces :
    - ContactSurface : La piste de freinage (plat, entre R_min et R_max)
    - CoolingSurface : Le reste (ailettes, moyeu, bordures)
    """
    surfaces = gmsh.model.getEntities(2)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    nodeCoords = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)
    tag_to_index = {int(tag): i for i, tag in enumerate(nodeTags)}

    contact_tags = []
    other_tags = []

    # Paramètres typiques d'un disque (à ajuster selon ton fichier STEP si nécessaire)
    # On cherche les faces planes (normale Z) qui sont sur le disque de friction
    R_INNER_CONTACT = 60.0  # Rayon min de la zone de contact des plaquettes
    R_OUTER_CONTACT = 140.0 # Rayon max de la zone de contact

    for dim, tag in surfaces:
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=dim, tag=tag)
        if len(elemTypes) == 0: continue
        
        tags = np.asarray(elemNodeTags[0], dtype=int)
        unique_tags = np.unique(tags)
        coords = nodeCoords[[tag_to_index[t] for t in unique_tags]]
        
        if coords.size == 0: continue

        centroid = coords.mean(axis=0)
        radius = np.sqrt(centroid[0]**2 + centroid[1]**2)
        
        # Calcul de la normale moyenne de la surface
        normal = np.zeros(3)
        if coords.shape[0] >= 3:
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm > 0: normal /= norm

        # CRITÈRE : Si la surface est plane (normale Z dominante) 
        # et située dans la zone radiale des plaquettes
        if abs(normal[2]) > 0.9 and (R_INNER_CONTACT < radius < R_OUTER_CONTACT):
            contact_tags.append(tag)
        else:
            other_tags.append(tag)

    return contact_tags, other_tags

def classify_surface_by_color():
    """Identifie les surfaces selon leur couleur CAO (Vert = Contact)."""
    surfaces = gmsh.model.getEntities(2)
    contact_tags = []
    other_tags = []

    for dim, tag in surfaces:
        # Récupère la couleur (r, g, b, a)
        r, g, b, a = gmsh.model.getColor(dim, tag)
        
        # Détection du vert : 
        # On vérifie si le canal G est dominant ( > 200) et R, B sont faibles ( < 100)
        # Cela permet d'être robuste aux nuances de vert.
        if g > 200 and r < 100 and b < 100:
            contact_tags.append(tag)
        else:
            other_tags.append(tag)
            
    return contact_tags, other_tags

def build_brake_disc_3d_no_viz(step_file="Break Disc Practice.stp", cl=10):
    print(f"Importing STEP file: {step_file}")
    gmsh.model.occ.importShapes(step_file)
    
    gmsh.model.occ.synchronize()
    
    # --- CLASSIFICATION AVANT MAILLAGE ---
    # Il est préférable d'identifier les entités juste après la synchro
    contact_tags, other_tags = classify_surface_by_color()
    print(f" > Surfaces de contact (vertes) identifiées : {contact_tags}")

    # Réglage de la taille de maille
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), cl)
    

    print("Génération du maillage 3D...")
    gmsh.model.mesh.generate(3)
    
    nodeTags, _, _ = gmsh.model.mesh.getNodes()
    # Récupération de tous les types d'éléments de dimension 3 (tétraèdres)
    elemTypes, elemTags, _ = gmsh.model.mesh.getElements(dim=3)
    
    # --- CRÉATION DES GROUPES PHYSIQUES ---
    if contact_tags:
        gmsh.model.addPhysicalGroup(2, contact_tags, tag=100, name="ContactSurface")
    if other_tags:
        gmsh.model.addPhysicalGroup(2, other_tags, tag=101, name="OtherSurfaces")

    volumes = [t[1] for t in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, volumes, tag=1, name="DiscVolume")

    # Préparation des bnds_tags
    bnds = [("ContactSurface", 2), ("OtherSurfaces", 2)]
    bnds_tags = []
    for name, dim in bnds:
        try:
            p_tag = gmsh.model.getPhysicalGroups(dim)
            # On cherche le tag correspondant au nom
            actual_tag = -1
            for pt in p_tag:
                if gmsh.model.getPhysicalName(dim, pt[1]) == name:
                    actual_tag = pt[1]
                    break
            
            if actual_tag != -1:
                nodes = gmsh.model.mesh.getNodesForPhysicalGroup(dim, actual_tag)[0]
                bnds_tags.append(nodes)
            else:
                bnds_tags.append(np.array([], dtype=int))
        except:
            bnds_tags.append(np.array([], dtype=int))

    return nodeTags, elemTypes, elemTags, bnds, bnds_tags