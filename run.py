from defect_gen_class import generate_defect

"""
This file is meant as a conventient method to generate many defects
All parameters are specified here, such as the size of the supercell, the defects, the distance between layers or its stacking
The user can also freely choose the name/directory of the input file and the directory of the output file

The defect can either be a single string in which case there will be only a single layer with that defect
In case the defect is an array with strings, each index denotes the layer in which the defect resides from the bottom up
E.g. ["defect1", "defect2"] gives a bilayer system with the bottom layer defect1 and in the second layer defect2
The defects array below (defs) should generate 2 single layer systems, then 2 bilayer systems and one trilayer system
"""
mat = "BN"
ext = ".in"

dims = [5]  # Number of unit cells in both x and y direction, so an NxN supercell in the xy plane
defs = ["pure", "V_N", ["pure", "C_B"], ["C_B", "pure"], ["naphtalene", "pure", "C_B_1_0!C_N_1_0"]]
z_dist = 4
stacking = "AA"

write = "FHI-aims"
path = "test"

system = generate_defect(material=mat, extension=ext)
for ii in dims:
    for jj in defs:
        system.createDefectStructure(dt=jj, Nx=ii, Ny=ii, Nz=len(jj), z_dist=z_dist, write=write, path=path, stacking=stacking)
