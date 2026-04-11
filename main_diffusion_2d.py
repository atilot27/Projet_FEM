# main_diffusion_3d.py
import argparse
import numpy as np
import gmsh

from gmsh_utils import (
    getPhysicalEntities, gmsh_init, gmsh_finalize, build_brake_disc_3d_no_viz, 
    prepare_quadrature_and_basis, get_jacobians
)
from stiffness import assemble_rhs_neumann, assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step

def main():
    print("Starting main simulation (Surface Heating)")
    parser = argparse.ArgumentParser(description="Diffusion 3D with surface heat flux")
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0e-02)
    parser.add_argument("--nsteps", type=int, default=500)
    parser.add_argument("--gmsh-view", action="store_true")
    args = parser.parse_args()

    dt = args.dt
    gmsh_init("Brake_Disc_Heat_Flux")

    # --- GÉNÉRATION DU MAILLAGE ---
    # On récupère les tags avec la nouvelle classification radiale
    nodeTags, elemTypes, elemTags, bnds, bnds_tags = build_brake_disc_3d_no_viz(step_file="Break Disc Practice.stp", cl=10)
    gmsh.model.mesh.setOrder(args.order)

    # --- EXTRACTION DES DONNÉES ---
    elemType = gmsh.model.mesh.getElementType("tetrahedron", args.order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

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

    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # --- PARAMÈTRES PHYSIQUES ET FLUX ---
    def kappa(x): return 0.5
    def u0(x): return 30.0
    
    # On coupe la source volumique (le chauffage vient de la surface désormais)
    def f_source_null(x, t): return 0.0 

    # Condition de Neumann : Flux de chaleur imposé sur la piste de freinage
    # Note : Le nom doit correspondre exactement à celui dans gmsh_utils.py
    neumann_data = {
        "ContactSurface": lambda x, t: 100.0, # Flux intense (W/m² ou unité cohérente)
        "OtherSurfaces": lambda x, t: 0.0      # Isolé
    }

    # --- ASSEMBLAGE DES MATRICES DE BASE ---
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    # K et F0 (F0 sera vide car f_source est nulle)
    K_lil, F0 = assemble_stiffness_and_rhs(elemTags, elemNodeTags, jac, det, coords, w, N, gN, 
    kappa, lambda x: f_source_null(x, 0), tag_to_dof)

    M = M_lil.tocsr()
    K = K_lil.tocsr()

    U = np.array([u0(x) for x in dof_coords], dtype=float)
    dir_dofs = np.array([], dtype=int)

    def point_scalars_from_dofs(U):
        scalars = np.zeros(len(nodeTags), dtype=float)
        for tag in unique_dofs_tags:
            scalars[tag_to_index[int(tag)]] = U[tag_to_dof[int(tag)]]
        return scalars

    view_tag = gmsh.view.add("Temperature Evolution")

    # --- BOUCLE TEMPORELLE ---
    for step in range(args.nsteps):
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
                xiBnd, wBnd, NBnd, gNBnd = prepare_quadrature_and_basis(eTypeBnd, args.order)
                jacBnd, detBnd, coordsBnd = get_jacobians(eTypeBnd, xiBnd, tag=entityTag)
                
                # Accumulation du flux dans le second membre
                F = assemble_rhs_neumann(F, elemTagsBnd[0], elemNodeTagsBnd[0], jacBnd, detBnd, coordsBnd, wBnd, NBnd, gNBnd, lambda x: flux_func(x, t), tag_to_dof)
                Fnp1 = assemble_rhs_neumann(Fnp1, elemTagsBnd[0], elemNodeTagsBnd[0], jacBnd, detBnd, coordsBnd, wBnd, NBnd, gNBnd, lambda x: flux_func(x, t+dt), tag_to_dof)

        # Résolution du pas de temps
        U = theta_step(M, K, F, Fnp1, U, dt=dt, theta=args.theta, dirichlet_dofs=dir_dofs, dir_vals_np1=np.array([]))

        if step % 10 == 0 or step == args.nsteps - 1:
            print(f"Step {step}/{args.nsteps}, t={t:.3f}s, Max T={np.max(U):.1f}°C")
            temperature = point_scalars_from_dofs(U)
            gmsh.view.addModelData(view_tag, step, gmsh.model.getCurrent(), "NodeData", 
                                  nodeTags.astype(int).tolist(), temperature.reshape(-1, 1).tolist(), 
                                  time=t, numComponents=1)
    
    # --- CONFIGURATION FINALE DE L'AFFICHAGE ---
    view_index = gmsh.view.getIndex(view_tag)
    v_str = f"View[{view_index}]"
    
    gmsh.option.setNumber(f"{v_str}.RangeType", 2)
    gmsh.option.setNumber(f"{v_str}.CustomMin", 20.0) 
    gmsh.option.setNumber(f"{v_str}.CustomMax", 1500.0)
    gmsh.option.setNumber(f"{v_str}.ColormapNumber", 7) # Palette "Hot"
    gmsh.option.setNumber(f"{v_str}.Light", 1)
    gmsh.option.setColor("General.Background", 255, 255, 255)

    if args.gmsh_view:
        gmsh.fltk.run()

    gmsh_finalize()

if __name__ == "__main__":
    main()