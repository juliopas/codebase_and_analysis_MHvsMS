A_to_obj_dict = {'HM': "PointDipolePermanent", 'SM': "PointDipoleSuperpara", 'SMfl': "PointDipoleSuperpara", 'SMdumb': "PointDipoleSuperpara"}
A_help_dict = {"HM": "pdp", "SM": "pds", "SMfl": "pds", "SMdumb": "pds"}

K_legend_dict = {"hard": r"$K_{high}$", "soft": r"$K_{low}$"}
K_lims_dict = {'soft': (0.001, 0.01), 'hard': (0.01, 0.1)}

# Explicit export list
__all__ = [
    "A_to_obj_dict",
    "A_help_dict",
    "K_legend_dict",
    "K_lims_dict"
]