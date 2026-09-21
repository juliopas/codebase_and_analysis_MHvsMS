# pyanal.ARay.structure

def create_structure_from_flags(bool_type, int_type, float_type, vlen_float, flag_dict):
    dtype = []
    
    # Basic time step info
    dtype.append(('time_step', float_type))
    dtype.append(('seed_count',  int_type))

    # for anal SURFACE
    if flag_dict.get("do_ENERGIES", False):
        dtype.append(('E_bond', float_type))
        dtype.append(('E_dip', float_type))
        dtype.append(('E_Zee', float_type))

        dtype.append(('E_Zee_dipm', float_type))

    if flag_dict.get("do_MAGNETIZATION", False):
        dtype.append(('m_z', float_type))
        dtype.append(('m_z_dipm', float_type))
        dtype.append(('M_z', float_type))

        dtype.append(('E_chains', float_type))
    
    if flag_dict.get("do_HEIGHTS", False):
        dtype.extend([
            ('rms', float_type),
            ('h_avg', float_type)
            ])
        
        if flag_dict.get("do_USE_BOND_TO_PARTS", False):
            dtype.extend([
                ('rms_with_bonds', float_type),
                ('h_avg_with_bonds', float_type)
            ])

        if flag_dict.get("do_USE_SURFACE_SHAPE", False):
            dtype.extend([
                ('rms_with_surface', float_type),
                ('h_avg_with_surface', float_type),
                ('rms_interpolated', float_type),
                ('h_avg_interpolated', float_type)
            ])

            if flag_dict.get("do_USE_BOND_TO_PARTS", False):
                dtype.extend([
                    ('rms_with_bonds_with_surface', float_type),
                    ('h_avg_with_bonds_with_surface', float_type)
                ])

        if flag_dict.get("do_PEAK_STATISTICS", False):
            dtype.extend([
                ('n_peaks', float_type),
                ('peak_h_avg', float_type),
                ('peak_h_std', float_type),
                ('peak_distance_avg', float_type),
                ('peak_distance_std', float_type)
            ])

            if flag_dict.get("do_USE_BOND_TO_PARTS", False):
                dtype.extend([
                    ('n_peaks_with_bonds', float_type),
                    ('peak_h_avg_with_bonds', float_type),
                    ('peak_h_std_with_bonds', float_type),
                    ('peak_distance_avg_with_bonds', float_type),
                    ('peak_distance_std_with_bonds', float_type)
                ])
        
    if flag_dict.get("do_DENSITY_Z", False):
        dtype.append(('dens_z_bin_edges', vlen_float)) # numpy arrayof unkown size

        dtype.append(('dens_z', vlen_float))
        dtype.append(('thickness', float_type))

        if flag_dict.get("do_USE_BOND_TO_PARTS", False):
            dtype.append(('dens_z_with_bonds', vlen_float))
            dtype.append(('thickness_with_bonds', float_type))
    
    if flag_dict.get("do_VOLUME", False):
        dtype.append(('volume_avg', float_type))

        if flag_dict.get("do_USE_BOND_TO_PARTS", False):
            dtype.append(('volume_avg_with_bonds', float_type))

    # for anal CLUSTER
    if flag_dict.get("do_ENERGETIC_CLUSTER", False):
        dtype.append(("cutoff_E_cluster", float_type))
        dtype.append(("cutoff_r_cluster", float_type))

        dtype.append(("n_E_clusters_avg", float_type))
        dtype.append(("num_E_neighbors_list", vlen_float))
        dtype.append(("cluster_E_sizes_list", vlen_float))

    if flag_dict.get("do_CLUSTER", False):
        dtype.append(("cutoff_cluster", float_type))

        dtype.append(("n_clusters_avg", float_type))
        dtype.append(("num_neighbors_list", vlen_float))
        dtype.append(("distances_hist", vlen_float))
        dtype.append(("distances_bins", vlen_float))
        dtype.append(("cosines_list", vlen_float))
        dtype.append(("cosine_avg_per_neighbourhood", vlen_float))
        dtype.append(("cluster_sizes_list", vlen_float))

    if flag_dict.get("do_BOP", False):
        dtype.append(("cutoff_BOP", float_type))

        if isinstance(flag_dict.get("l_list"), list):
            dtype.append(("l_list", vlen_float))
            for l in flag_dict.get("l_list"):
                dtype.append((f"q_{l}_vals", vlen_float))
                dtype.append((f"q_{l}", float_type))
        else:
            l_list_val = flag_dict.get("l_list", None)
            raise TypeError(f"l_list should be a python list and not: {type(l_list_val)}")

    if flag_dict.get("do_ENERGETIC_BOP", False):
        dtype.append(("cutoff_E_BOP", float_type))
        dtype.append(("cutoff_r_BOP", float_type))

        if isinstance(flag_dict.get("l_list"), list):
            dtype.append(("l_list_E", vlen_float))
            for l in flag_dict.get("l_list"):
                dtype.append((f"q_{l}_E_vals", vlen_float))
                dtype.append((f"q_{l}_E", float_type))
        else:
            l_list_val = flag_dict.get("l_list", None)
            raise TypeError(f"l_list should be a python list and not: {type(l_list_val)}")

    if flag_dict.get("do_kink_ENERGIES", False):
        dtype.append(("bulk_E", float_type))
        dtype.append(("surface_E", float_type))

        dtype.append(("height_bins", vlen_float))
        dtype.append(("layer_Es", vlen_float))
        dtype.append(("layer_Es_in", vlen_float))
        dtype.append(("layer_Es_out", vlen_float))

    return dtype