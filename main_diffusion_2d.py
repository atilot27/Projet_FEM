# main_diffusion_3d.py
import argparse
from dataclasses import dataclass
import numpy as np
import gmsh
from scipy.sparse import lil_matrix

from gmsh_utils import (
    getPhysicalEntities, gmsh_init, gmsh_finalize, build_brake_disc_3d_basic, 
    prepare_quadrature_and_basis, get_jacobians, build_brake_disk_3d_disk1, build_brake_disk_3d_WVA,
    build_brake_disk_3d_Pogi
)


@dataclass
class SimulationParameters:
    "Paramètre liée à la simulation"
    order: int = 1
    theta: float = 1.0
    dt: float = 1.0e-02
    nsteps: int = 200

    "Vitesse de la roue en rad/s (rayon de 34cm et à 45km/h, ça fait environ 12.5 rad/s)"
    rayon = 0.34
    omega: float = 45.0 / 3.6 / rayon

    "Paramètres de convection pour moduler l'effet du vent sur la convection"
    #Left
    k00 = 1.896
    k10 = 0.01784
    k01 = 6.505
    K11 = -0.009343
    k02 = 0.04511 
    delta_T_moyen = 50.0
    bike_speed = 45.0 / 3.6
    h_conv = k00 + k10 * delta_T_moyen + k01 * bike_speed + K11 * bike_speed * delta_T_moyen + k02 * bike_speed**2


    "Paramètres du mesh"
    mesh_size: float = 50 #Attention, si tu met 1, ça va faire 600000 éléments

    "Paramètres de diffusion"
    kappa_value: float = 10e-5 #m^2/s, diffusivité thermique typique de l'acier
    initial_temperature: float = 15.0
    T_ext: float = 15.0

    "Paramètres des plaquettes de frein"
    # Le STEP importé est en mm, donc on utilise des valeurs en mm ici.
    pad_center_x: float = 61.5
    pad_half_width: float = 15.0
    pad_half_height: float = 9.0
    thickness: float = 2.0 # Epaisseur du disque de frein en mm
    contact_surface = 343.25 # mm² (Pogi:343.25 WVA: 260.25 )
    puissance_a_dissiper = 10000.0 # W
    pad_flux_value: float = puissance_a_dissiper / contact_surface  #W/mm²

    "Paramètres d'affichage dans Gmsh"
    colormap: int = 4 #Pour changer la couleur
    temperature_min: float = 0
    temperature_max: float = 250
    background_color: tuple = (255, 255, 255)
    view_light: int = 1
    hide_mesh_edges: bool = True #Cache le maillage pour que ça soit plus joli
    contact_surface_name: str = "ContactSurface"
    other_surface_name: str = "OtherSurfaces"
    model_name: str = "Brake_Disc_Heat_Flux"
    geometry_model: str = "Pogi" #Choix du modèle de disque de frein : "disk1", "Pogi" ou "WVA"
    use_gmsh_gui: bool = True

from stiffness import assemble_rhs_neumann, assemble_stiffness_and_rhs, assemble_boundary_convection
from mass import assemble_mass
from dirichlet import theta_step

def main():

    #-----------------------------------------------------------------------------------
    # Simulation parameters:
    #-----------------------------------------------------------------------------------

    print("Starting main simulation (Surface Heating)")
    params = SimulationParameters()

    parser = argparse.ArgumentParser(description="Diffusion 3D with surface heat flux")
    parser.add_argument("--no-gmsh-view", action="store_true",
                        help="Disable Gmsh GUI after the simulation")
    parser.add_argument("--model", choices=["disk1", "Pogi", "WVA"], default=params.geometry_model,
                        help="Choose the brake disk geometry to use: disk1, Pogi or WVA")
    args = parser.parse_args()
    if args.no_gmsh_view:
        params.use_gmsh_gui = False
    params.geometry_model = args.model

    dt = params.dt

    #-----------------------------------------------------------------------------------
    # Génération du maillage:
    #-----------------------------------------------------------------------------------

    gmsh_init(params.model_name)
    gmsh.option.setNumber("General.Terminal", 0)
    if params.geometry_model == "disk1":
        nodeTags, elemTypes, elemTags, bnds, bnds_tags = build_brake_disk_3d_disk1(cl=params.mesh_size)
    elif params.geometry_model == "Pogi":
        nodeTags, elemTypes, elemTags, bnds, bnds_tags = build_brake_disk_3d_Pogi(cl=params.mesh_size)
    else:
        nodeTags, elemTypes, elemTags, bnds, bnds_tags = build_brake_disk_3d_WVA(cl=params.mesh_size)
    gmsh.model.mesh.setOrder(params.order)

    #-----------------------------------------------------------------------------------
    # Extraction des données
    #-----------------------------------------------------------------------------------

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTypes3D, elemTags3D, elemNodeTags3D = gmsh.model.mesh.getElements(dim=3)
    if not elemTypes3D:
        raise RuntimeError("Aucun élément volumique 3D n'a été trouvé dans le maillage. Vérifiez la génération du maillage.")

    elemType = elemTypes3D[0]
    elemTags = elemTags3D[0]
    elemNodeTags = elemNodeTags3D[0]

    if len(elemNodeTags) == 0:
        raise RuntimeError("Aucun noeud d'élément n'a été trouvé pour le type d'élément 3D. Vérifiez la génération du maillage.")

    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))
    tag_to_index = {int(tag): i for i, tag in enumerate(nodeTags)}
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    
    all_coords = nodeCoords.reshape(-1, 3)
    dof_coords = np.zeros((num_dofs, 3))
    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[tag_to_index[int(tag)]]

    z_min = float(np.min(dof_coords[:, 2]))
    z_max = float(np.max(dof_coords[:, 2]))
    z_tol = 1e-3

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, params.order)
    jac, det, coords = get_jacobians(elemType, xi)

    #-----------------------------------------------------------------------------------
    # Paramètres physiques et fonctions de source:
    #-----------------------------------------------------------------------------------

    def kappa(x):
        return params.kappa_value

    def u0(x):
        return params.initial_temperature

    def f(x, t):
        return 0.0

    def rotate_point_about_z(x, theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return np.array([
            cos_t * x[0] - sin_t * x[1],
            sin_t * x[0] + cos_t * x[1],
            x[2]
        ])

    def flux(x, t):
        # Le disque tourne ; le patin de frein reste fixe dans le repère.
        theta = params.omega * t
        x_rot = rotate_point_about_z(x, theta)

        # 1. Vérification X/Y : Est-on sous les plaquettes de frein ?
        in_x = abs(x_rot[0] - params.pad_center_x) <= params.pad_half_width
        in_y = abs(x_rot[1]) <= params.pad_half_height

        # 2. Vérification Z : Est-on sur une face extérieure ?
        # On demande à ce que le point soit suffisamment éloigné du centre Z=0
        # (ex: au-delà de 40% de la demi-épaisseur) pour éviter de chauffer 
        # l'intérieur d'éventuels trous de ventilation.
        half_thick = params.thickness / 2.0
        is_external_face = abs(x_rot[2]) > (half_thick * 0.4)

        if in_x and in_y and is_external_face:
            return params.pad_flux_value
            
        return 0.0

    # Condition de Neumann : flux non nul uniquement sur les zones de contact des plaquettes
    neumann_data = {
        params.contact_surface_name: flux,
        params.other_surface_name: flux
    }

    #-----------------------------------------------------------------------------------
    # Assemblage des matrices et vecteurs
    #-----------------------------------------------------------------------------------

    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    K_lil, F0 = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, kappa, lambda x: f(x, 0), tag_to_dof)

    # Ici, on prépare les contributions de convection aux surfaces.
    K_conv_lil = lil_matrix((len(F0), len(F0)), dtype=float)
    F_conv = np.zeros(len(F0), dtype=float)

    if params.h_conv > 0.0:
        def h_effective(x):
            return params.h_conv

        for name in neumann_data:
            entities = getPhysicalEntities(name)
            for dim, entityTag in entities:
                elemTypesBnd, elemTagsBnd, elemNodeTagsBnd = gmsh.model.mesh.getElements(dim=dim, tag=entityTag)
                if not elemTypesBnd:
                    continue

                eTypeBnd = elemTypesBnd[0]
                xiBnd, wBnd, NBnd, gNBnd = prepare_quadrature_and_basis(eTypeBnd, params.order)
                jacBnd, detBnd, coordsBnd = get_jacobians(eTypeBnd, xiBnd, tag=entityTag)

                K_conv_lil, F_conv = assemble_boundary_convection(
                    K_conv_lil,
                    F_conv,
                    elemTagsBnd[0],
                    elemNodeTagsBnd[0],
                    jacBnd,
                    detBnd,
                    coordsBnd,
                    wBnd,
                    NBnd,
                    gNBnd,
                    h_effective,
                    params.T_ext,
                    tag_to_dof
                )
    #Ici on ajoute la contribution de la convection à la matrice de rigidité et au second membre.
    M = M_lil.tocsr()
    K = (K_lil + K_conv_lil).tocsr()
    F0 += F_conv

    U = np.array([u0(x) for x in dof_coords], dtype=float)
    dir_dofs = np.array([], dtype=int)

    def point_scalars_from_dofs(U):
        scalars = np.zeros(len(nodeTags), dtype=float)
        for tag in unique_dofs_tags:
            scalars[tag_to_index[int(tag)]] = U[tag_to_dof[int(tag)]]
        return scalars

    view_tag = gmsh.view.add("Temperature Evolution")

    #-----------------------------------------------------------------------------------
    # Boucle temporelle
    #-----------------------------------------------------------------------------------
    
    for step in range(params.nsteps):
        t = step * dt

        # Initialisation des RHS pour le schéma en thêta
        F = F0.copy()
        Fnp1 = F0.copy()

        # On applique le flux de surface sur les entités physiques
        for name, flux_func in neumann_data.items():
            entities = getPhysicalEntities(name)
            for dim, entityTag in entities:
                elemTypesBnd, elemTagsBnd, elemNodeTagsBnd = gmsh.model.mesh.getElements(dim=dim, tag=entityTag)
                if not elemTypesBnd: continue
                
                # Récupération des données d'intégration pour la surface (triangles)
                eTypeBnd = elemTypesBnd[0]
                xiBnd, wBnd, NBnd, gNBnd = prepare_quadrature_and_basis(eTypeBnd, params.order)
                jacBnd, detBnd, coordsBnd = get_jacobians(eTypeBnd, xiBnd, tag=entityTag)
                
                # Accumulation du flux dans le second membre
                F = assemble_rhs_neumann(F, elemTagsBnd[0], elemNodeTagsBnd[0], jacBnd, detBnd, coordsBnd, wBnd, NBnd, gNBnd, lambda x: flux_func(x, t), tag_to_dof)
                Fnp1 = assemble_rhs_neumann(Fnp1, elemTagsBnd[0], elemNodeTagsBnd[0], jacBnd, detBnd, coordsBnd, wBnd, NBnd, gNBnd, lambda x: flux_func(x, t+dt), tag_to_dof)

        # Résolution du pas de temps
        U = theta_step(M, K, F, Fnp1, U, dt=dt, theta=params.theta, dirichlet_dofs=dir_dofs, dir_vals_np1=np.array([]))

        if step % 10 == 0 or step == params.nsteps - 1:
            print(f"Step {step}/{params.nsteps}, t={t:.3f}s, Max T={np.max(U):.1f}°C")
            temperature = point_scalars_from_dofs(U)
            gmsh.view.addModelData(view_tag, step, gmsh.model.getCurrent(), "NodeData", 
                                  nodeTags.astype(int).tolist(), temperature.reshape(-1, 1).tolist(), 
                                  time=t, numComponents=1)

    #-----------------------------------------------------------------------------------
    # Configuration de l'affichage dans Gmsh
    #-----------------------------------------------------------------------------------

    view_index = gmsh.view.getIndex(view_tag)
    v_str = f"View[{view_index}]"
    
    gmsh.option.setNumber(f"{v_str}.RangeType", 2)
    
    gmsh.option.setNumber(f"{v_str}.CustomMin", params.temperature_min)
    gmsh.option.setNumber(f"{v_str}.CustomMax", params.temperature_max)
    gmsh.option.setNumber(f"{v_str}.ColormapNumber", params.colormap)
    gmsh.option.setNumber(f"{v_str}.Light", params.view_light)
    gmsh.option.setColor("General.Background", *params.background_color)
    if params.hide_mesh_edges:
        gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
        gmsh.option.setNumber("Mesh.VolumeEdges", 0)
    if params.use_gmsh_gui:
        gmsh.fltk.run()

    gmsh_finalize()

if __name__ == "__main__":
    main()