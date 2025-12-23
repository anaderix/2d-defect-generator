import os
import numpy as np


class generate_defect:
    def __init__(self, material: str, extension: str = ".in"):
        """
        Initiate class by obtaining lattice and atom information of a pure material from a material.extension file
        """
        F = open(material + extension, "r")
        lattice, atom_info_frac = [], []
        for line in F:
            if "lattice_vector" in line:  # Extract lattice vector
                lattice.append([float(ii) for ii in line.split()[1:]])
            if "atom_frac" in line:  # Extract fractional coords of atoms
                atom_info_frac.append([line.split()[-1]] + [float(ii) for ii in line.split()[1:-1]])

        self.material = material
        self.lattice = lattice
        self.atom_info_frac = atom_info_frac
        self.num_atoms = len(atom_info_frac)

    r"""
       _____                                 ____   _____      __
      / ___/__  ______  ___  _____________  / / /  / ___/___  / /___  ______
      \__ \/ / / / __ \/ _ \/ ___/ ___/ _ \/ / /   \__ \/ _ \/ __/ / / / __ \
     ___/ / /_/ / /_/ /  __/ /  / /__/  __/ / /   ___/ /  __/ /_/ /_/ / /_/ /
    /____/\__,_/ .___/\___/_/   \___/\___/_/_/   /____/\___/\__/\__,_/ .___/
              /_/                                                   /_/
    """

    def _adjustUnitVectorZ(self, z_dist: float):
        """
        Adjust the out-of-plane value of unit-vector z: [..., ..., z_dist]
        """
        if z_dist == None:
            return
        else:
            self.lattice[2][2] = z_dist

    def _initSupercell(self, Nx: int, Ny: int, Nz: int):
        """
        Initialise parameters necessary for supercell
        """
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.sup_lattice = np.array(np.multiply(self.lattice, [Nx, Ny, Nz]))

    def _createSupercell(self, zi: int):
        """
        Create the supercell in the xy plane of Nx by Ny unit cells at layer z_i
        """
        self.num_sup_atoms = self.num_atoms * self.Nx * self.Ny

        sup_atom_info_dtype = [
            ("index", "i4"),  # Index as stored in the array
            ("atom_type", "U2"),  # Atom type corresponding to other data, e.g. B or N
            ("coord_frac", ("f8", 3)),  # Fractional coords, always between 0 and 1
            ("coord_cart", ("f8", 3)),  # Cart coords as calculated by multiplying with lattice vector
            ("charges", "f8"),  # Add charges to specific atoms, currently not in use
        ]
        sup_atom_info = np.zeros(self.num_sup_atoms, dtype=sup_atom_info_dtype)

        index = 0
        for xi in range(self.Nx):
            for yi in range(self.Ny):
                for atom in self.atom_info_frac:
                    sup_atom_info[index]["index"] = index
                    sup_atom_info[index]["atom_type"] = atom[0]
                    sup_atom_info[index]["coord_frac"] = [
                        (atom[1] + xi) / self.Nx,
                        (atom[2] + yi) / self.Ny,
                        (atom[3] + zi) / self.Nz,
                    ]
                    sup_atom_info[index]["charges"] = 0
                    index += 1

        sup_atom_info["coord_cart"] = sup_atom_info["coord_frac"] @ self.sup_lattice
        self.sup_atom_info = sup_atom_info

    r"""
        ____       ____          __     ______                __  _
       / __ \___  / __/__  _____/ /_   / ____/_______  ____ _/ /_(_)___  ____
      / / / / _ \/ /_/ _ \/ ___/ __/  / /   / ___/ _ \/ __ `/ __/ / __ \/ __ \
     / /_/ /  __/ __/  __/ /__/ /_   / /___/ /  /  __/ /_/ / /_/ / /_/ / / / /
    /_____/\___/_/  \___/\___/\__/   \____/_/   \___/\__,_/\__/_/\____/_/ /_/
    """

    def _createSuper3x3(self):
        """
        Create a 3x3 supersupercell structured numpy array of the exisiting supercell, e.g. 2x2 becomes 6x6

        Returns
        -------
        sup_order: Structured numpy array
            Array containing the 3x3 supersupercell of the existing supercell as a structured numpy array
            Will be updated to contain necessary info to select atoms based on distance and relative angle with periodicity
        """
        sup_order_dtype = self.sup_atom_info.dtype.descr + [
            ("per_coord_cart", ("f8", 3)),  # Periodic Cartesian coordinates
            ("rel_coord_cart", ("f8", 3)),  # Relative Cartesian coordinates
            ("dist_cart", "f8"),  # Cartesian distance
            ("dist_group", "i4"),  # Graph distance
            ("abs_rel_angle", "f8"),  # Absolute relative angle
            ("big_index", "i4"),  # Index as stored in the 3x3 supersupercell array
        ]
        sup_order = np.zeros(9 * self.num_sup_atoms, dtype=sup_order_dtype)

        # Copy data and shift the supercells using fractional coords
        for ii in range(9):
            xi, yi = ii // 3 - 1, ii % 3 - 1
            indx_i, indx_f = ii * self.num_sup_atoms, (ii + 1) * self.num_sup_atoms
            for name in ["index", "atom_type"]:  # Copy index and atom type
                sup_order[name][indx_i:indx_f] = self.sup_atom_info[name]
            for jj in range(9 * self.num_sup_atoms):
                sup_order[jj]["big_index"] = jj  # Determine the index as stored in the 3x3 supersupercell array
            sup_order["coord_frac"][indx_i:indx_f] = self.sup_atom_info["coord_frac"] + [xi, yi, 0]  # Shift of fractional coords
        sup_order["coord_cart"] = sup_order["coord_frac"] @ self.sup_lattice
        sup_order["per_coord_cart"] = sup_order["coord_cart"]  # Periodic Cart coords are equal to Cart coordiantes, will be updated
        return sup_order

    def _getIndicesClosestTwoNeighbours(self, seed_indx: int, sup_order: np.ndarray):
        """
        From a specific seed_indx, find all first and second equidistant neighbours in real space
        Real space is used as this is equivalent to finding all vertices of distance 1 and 2 in a pure hexagonal lattice

        Arguments
        ---------
        seed_indx: int
            Index of atom whose neighbours you want to find
        sup_order: Structured numpy array

        Returns
        -------
        N1_indx: numpy array of integers
            First neighbours, there should be at most 3
        N2_indx: numpy array of integers
            Second neighbours, there should be at most 6

        """
        N1_indx, N2_indx = [], []
        indx_coord_frac = sup_order[seed_indx]["coord_frac"]
        rel_coord_cart = (sup_order["coord_frac"] - indx_coord_frac) @ self.sup_lattice
        dist_cart = np.round(np.linalg.norm(rel_coord_cart, axis=1), decimals=5)
        unique_vals, group_indices = np.unique(dist_cart, return_inverse=True)

        N1_indx = np.where(group_indices == 1)[0]
        N2_indx = np.where(group_indices == 2)[0]
        return N1_indx, N2_indx

    def _getDistanceGroups(self, seed_indx: int):
        """
        Algorithm to determine the periodic distances in the supercell from a seed index

        Arguments
        ---------
        seed_indx: int
            Index of atom whose distances you want to find

        Returns
        -------
        sup_order: Structured numpy array
            Updated with correct periodic graph distances relative to a seed index
        """
        sup_order = self._createSuper3x3()
        visit, visit_this_round, count = np.arange(9 * self.num_sup_atoms), [seed_indx + ii * self.num_sup_atoms for ii in range(9)], 0
        dist_group = np.zeros((9 * self.num_sup_atoms, 2))
        dist_group.T[0] = visit.copy()

        visit = np.delete(visit, np.where(np.isin(visit, visit_this_round)))  # Keep track of which indices have not yet been visited
        while len(visit) != 0:
            N1_indx_round, N2_indx_round = [], []  # Keep track of the indices
            for seed_indx in visit_this_round:
                N1_indx, N2_indx = self._getIndicesClosestTwoNeighbours(seed_indx, sup_order)

                N1_indx_round.append(N1_indx)
                N2_indx_round.append(N2_indx)
            N1_indx_round = np.intersect1d(np.concatenate(N1_indx_round), visit)
            N2_indx_round = np.intersect1d(np.concatenate(N2_indx_round), visit)

            # Save distance for index if it has not been determined before or is smaller than the previous distance
            for ii in N2_indx_round:
                distance = 2 + 2 * count
                if dist_group[ii][1] == 0 or distance < dist_group[ii][1]:
                    dist_group[ii][1] = distance
            for ii in N1_indx_round:
                distance = 1 + 2 * count
                if dist_group[ii][1] == 0 or distance < dist_group[ii][1]:
                    dist_group[ii][1] = distance

            # Remove visited indices
            visit = np.delete(visit, np.where(np.isin(visit, N1_indx_round))[0])
            visit = np.delete(visit, np.where(np.isin(visit, N2_indx_round))[0])

            # Visit all second neighbours to check their neighbours
            visit_this_round = N2_indx_round

            count += 1
        sup_order["dist_group"] = dist_group.T[1]
        return sup_order

    def _getRelPerCartCoords(self, seed_index: int, sup_order: np.ndarray):
        """
        Get the Cartesian coords accounting for periodicity
        This only affects atoms further removed within the supercell than when using the supersupercell

        Arguments
        ---------
        seed_indx: int
            Index of atom whose distances you want to find
        sup_order: Structured numpy array

        Returns
        -------
        sup_order: Structured numpy array
            Updated with periodic Cartesian coordinates
        """
        for ii in range(self.num_sup_atoms):
            indx_i = 4 * self.num_sup_atoms  # Start after the 4th supercell in the supersupercell
            seed_coord = sup_order["coord_cart"][indx_i + seed_index]
            indx_ii = np.where(sup_order["index"] == ii)[0]
            coord_ii = sup_order["coord_cart"][indx_ii]

            # Calculate the distance
            per_dist = np.linalg.norm(coord_ii - seed_coord, axis=1)
            per_dist_min = np.round(np.min(per_dist), decimals=5)
            cell_dist = np.linalg.norm(sup_order["coord_cart"][indx_i + ii] - seed_coord)

            # If the periodic distance is smaller than the distance within the cell, use the periodic distance
            if per_dist_min < cell_dist:
                sup_order["per_coord_cart"][indx_i + ii] = sup_order["coord_cart"][np.argmin(per_dist) * self.num_sup_atoms + ii]

        return sup_order

    def _getAbsRelAngle(self, rel_coord: np.ndarray, p_angle: float):
        """
        Get the absolute relative angle based on a relative coordinate

        Arguments
        ---------
        rel_coord: numpy array
            Relative coordinates [x, y, z]
        p_angle: float
            Desired angle as defined in the unit circle, i.e. going right is 0 degrees, up is 90 degrees

        Returns
        -------
        Absolute relative angle, e.g. if p_angle = 30 and there are atoms at 0, 30, 50, it would return 30, 0, 20, respectively
        """
        return np.round(np.abs((np.rad2deg(np.arctan2(*rel_coord.T[0:2][[1, 0]])) - p_angle + 180) % 360 - 180), decimals=4)

    def _getIndexNeighbour(self, seed_index: int, dist_group: int = 1, p_angle: int = 0, num_atoms: int = 1, atom_type: str = None):
        """
        Sort neighbours of a seed index (or coordinate) based on graph distance and relative angle. Can optionally filter for atom type.

        Arguments
        ---------
        seed_index: int
            Index of atom whose distances you want to find
        dist_group: int
            Desired graph distance
        p_angle: int
            Desired angle as defined in the unit circle, i.e. going right is 0 degrees, up is 90 degrees
        num_atoms: int
            Number desired atoms, however if set larger than the amount of possible atoms, it will crash
        atom_type: str
            Desired atom type, however if atom type is not compatible with the distance group, it will crash

        Returns
        -------
        List of indices to be replaced
        """
        # Closest distance from seed index is trivial, thus skip index 0
        first_index = 1
        seed_coord_frac = self.sup_atom_info[seed_index]["coord_frac"]
        seed_atom_type = self.sup_atom_info[seed_index]["atom_type"]

        # Create and update the supersupercell with distance groups and relative periodic coordinates
        sup_order = self._getDistanceGroups(seed_index)
        sup_order = self._getRelPerCartCoords(seed_index, sup_order)
        sup_order = sup_order[4 * self.num_sup_atoms : 5 * self.num_sup_atoms]  # Select the middle supercell from the supersupercell

        # Calculate the spatial distance and absolute relative angle
        sup_order["rel_coord_cart"] = sup_order["per_coord_cart"] - (seed_coord_frac @ self.sup_lattice)
        sup_order["dist_cart"] = np.linalg.norm(sup_order["rel_coord_cart"], axis=1)
        sup_order["abs_rel_angle"] = np.round(
            np.abs((np.rad2deg(np.arctan2(*sup_order["rel_coord_cart"].T[0:2][[1, 0]])) - p_angle + 180) % 360 - 180), decimals=4
        )

        # Sort and select the correct elements
        sup_order = np.sort(sup_order, order=["dist_group", "abs_rel_angle"])
        sup_order = sup_order[first_index:]
        sup_order = sup_order[np.where(sup_order["dist_group"] == dist_group)]

        if atom_type != None:
            sup_order = sup_order[np.where(sup_order["atom_type"] == atom_type)]
        if num_atoms == "all":
            return sup_order["index"]
        else:
            return sup_order["index"][0:num_atoms]

    def _getIndexCentremostAtom(self, atom_type: str = "B"):
        """
        Get the index of the centremost atom

        Arguments
        ---------
        atom_type: str
            Desired atom type

        Returns
        -------
        Index of centremost atom
        """
        if atom_type == "B":
            p_angle = -30
        elif atom_type == "N":
            p_angle = 120

        rel_coord_cart = (self.sup_atom_info["coord_frac"] - 0.5) @ self.sup_lattice
        distances = np.round(np.linalg.norm(rel_coord_cart, axis=1), decimals=5)

        min_dist_indx = np.where(distances == distances.min())[0]
        min_dist = self.sup_atom_info[min_dist_indx]
        rel_angle = self._getAbsRelAngle(rel_coord_cart, p_angle)[min_dist_indx]

        atom_type_indx = np.where(min_dist["atom_type"] == atom_type)[0]
        return min_dist["index"][atom_type_indx][np.where(rel_angle[atom_type_indx] == rel_angle[atom_type_indx].min())][0]

    def _getDefectInfo(self, dt: str):
        """
        Obtain indices of atoms te be replaced/removed based on defect type (dt)
        This code is quite flexible and customisable, however since there are many different types of defects, one has to come up with smart ways
            to select these properly based on the desired system

        To do: All defects that can be described as a chain can be automated by writing the defect as A1_B1_X1_Y1-A2_B2_X2_Y2-...
            with A the defect atom (e.g. C, O, V), B the replaced atom (B or N), X the distance and Y the angle
            This can then be automatically parsed, the length determined and the defect array can be calculated very quickly

        Arguments
        ---------
        dt: str
            Name of defect type

        Returns
        -------
        defect_info: Structured numpy array
            Indx contains the indices to be replaced and atom contains the atoms with which they are replaced
        """
        defect_info_dtype = [("indx", "i4"), ("atom", "U2")]

        if dt == "pure":
            print("Oops, this should not be called, goodbye!")
            return

        elif "!" in dt:
            """
            Example of defects in the shape of a1_b1_c1_d1!a2_b2_c2_d2!...

            The first defect a1 will be at atom b1 of the specified type in the middle of the lattice (c1 and d1 are not used)
            Then defect a2 will be at atom b2 at distance c2 with preferred angle nearest to d1 in degrees.

            So C_N_1_0!C_B_1_0 will replace the centremost N atom with a carbon and also the B atom that is only one position removed and near an angle of 0 degrees
            Note that C_B_1_0!C_B_1_0 will not result in unintended behaviour since there is no B atom one position removed from another B atom
            In this example, the user is advised to ensure that all defect structure are indeed valid with this method
            """
            replacement, to_be_replaced, angle, dist_group = [], [], [], []
            for count, ii in enumerate(dt.split("!")):
                re, to_be = np.array(ii.split("_"))[0:2]
                replacement.append(re)
                to_be_replaced.append(to_be)
                if count != len(dt.split("!")) - 1:
                    dist_group.append(int(ii.split("_")[2]))
                    angle.append(float(ii.split("_")[3]))

            defect_info = np.zeros(len(dt.split("!")), dtype=defect_info_dtype)

            ind_seed = self._getIndexCentremostAtom(atom_type=to_be_replaced[0])
            indices = [ind_seed]
            for ii in range(1, len(dt.split("!"))):
                indices.append(self._getIndexNeighbour(indices[-1], dist_group=dist_group[ii - 1], num_atoms=1, p_angle=angle[ii - 1])[0])

            defect_info["indx"] = indices
            defect_info["atom"] = replacement
            return defect_info

        elif dt in ["C_B", "C_N", "V_B", "V_N"]:
            # Hard coded example of a simple defect
            defect_info = np.zeros(1, dtype=defect_info_dtype)

            # Get index centremost B/N
            replacement, to_be_replaced = dt.split("_")
            ind_seed = self._getIndexCentremostAtom(atom_type=to_be_replaced)

            defect_info["indx"] = ind_seed
            defect_info["atom"] = replacement
            return defect_info

        elif dt in ["C_N_2_V_N", "C_B_2_V_B", "C_B_1_C_N", "C_N_1_C_B"]:
            # Hard coded example of a simple defect
            replacement = dt.split("_")[0::3]
            to_be_replaced = dt.split("_")[1::3]
            dist_group = int(dt.split("_")[2])

            defect_info = np.zeros(2, dtype=defect_info_dtype)

            # Get index centremost B/N
            ind_seed = self._getIndexCentremostAtom(atom_type=to_be_replaced[0])
            ind_2 = self._getIndexNeighbour(ind_seed, dist_group=dist_group, num_atoms=1, atom_type=to_be_replaced[1], p_angle=0)[0]

            defect_info["indx"] = [ind_seed, ind_2]
            defect_info["atom"] = replacement

            return defect_info

        elif dt in ["C_BC_N_3", "C_NC_B_3"]:
            # Hard coded example of a simple defect
            replacement = "C"
            to_be_replaced = dt.split("C_")[1]

            defect_info = np.zeros(4, dtype=defect_info_dtype)

            # Get index centremost B/N
            ind_seed = self._getIndexCentremostAtom(atom_type=to_be_replaced)
            indices = self._getIndexNeighbour(ind_seed, dist_group=1, num_atoms=3)

            defect_info["indx"] = np.append(ind_seed, indices)
            defect_info["atom"] = 4 * ["C"]

            return defect_info

        elif dt in ["naphtalene"]:
            # Hard coded example of a simple defect
            replacement = "C"

            defect_info = np.zeros(10, dtype=defect_info_dtype)

            # Get index centremost N
            ind_seed = self._getIndexCentremostAtom(atom_type="N")
            indices = [ind_seed]
            for ii in [150, -150, -90, -30, 30, -30, 30, 90, 150]:
                indices.append(self._getIndexNeighbour(indices[-1], dist_group=1, num_atoms=1, p_angle=ii)[0])

            defect_info["indx"] = indices
            defect_info["atom"] = 10 * ["C"]

            return defect_info

        else:
            print("The method you provided has not been found, goodbye!")
            exit()

    def _applyDefect(self, dt: str):
        """
        Adjust lattice according to defect type (dt) by overwriting the intended atoms with their replacements (or removing the in case of vacancies)

        Arguments
        ---------
        dt: str
            Name of defect type
        """
        self.defect_atom_info = self.sup_atom_info.copy()

        if dt == "pure":
            None
        else:
            # Get information about replacing/removing atoms
            defect_info = self._getDefectInfo(dt)
            index_replace = np.where(defect_info["atom"] != "V")[0]
            index_vacancy = np.where(defect_info["atom"] == "V")[0]

            # Important to first replace the atoms and then remove them as otherwise the indices will not be correct!!

            # Replace the atoms
            for indx in index_replace:
                list_indx = defect_info["indx"][indx]
                self.defect_atom_info["atom_type"][list_indx] = defect_info["atom"][indx]

            # Remove the atoms
            self.vac_info = self.defect_atom_info[defect_info["indx"][index_vacancy]]
            self.defect_atom_info = np.delete(self.defect_atom_info, defect_info["indx"][index_vacancy])
        self.num_defect_atoms = len(self.defect_atom_info)

    def _switchBandN(self):
        """
        Switch all atom types from B to N and vice versa
        """
        ind_B = np.where(self.defect_atom_info["atom_type"] == "B")
        ind_N = np.where(self.defect_atom_info["atom_type"] == "N")
        self.defect_atom_info["atom_type"][ind_B] = "N"
        self.defect_atom_info["atom_type"][ind_N] = "B"

    def _applyStacking(self, zi, stacking):
        """
        When stacking multiple layers, layer zi can have a certain shift/operation applied to it depending on the type of stacking
        The stacking follows the convention by Li, et al. https://doi.org/10.1038/s41699-024-00455-y, with p denoting prime, so ABpp is AB''
        Other uses are free and encouraged to implement other types of stacking suitable for their system

        Arguments
        ---------
        zi: int
            Number of layer with the initial layer being 0
        """
        # Value for specific lattice parameters to shift B on N
        shift_N_to_B_vert = np.array([0, -1.444666, 0])
        shift_N_to_B_right = np.array([-1.250988, 0.722258, 0])

        if stacking == "AA":
            shift = 0

        elif stacking == "AAp":
            if zi % 2 == 1:  # Odd
                self._switchBandN()
            shift = 0

        elif stacking == "AB":
            shift = np.power(0, 1 - zi % 2) * shift_N_to_B_vert

        elif stacking == "ABp":
            if zi % 2 == 1:  # Odd
                self._switchBandN()
            shift = np.power(0, 1 - zi % 2) * shift_N_to_B_right

        elif stacking == "ABpp":
            if zi % 2 == 1:  # Odd
                self._switchBandN()
            shift = np.power(0, 1 - zi % 2) * -shift_N_to_B_right

        elif stacking == "ABC":
            abc = zi % 3
            if abc == 0:
                shift = 0
            elif abc == 1:
                shift = shift_N_to_B_vert
            elif abc == 2:
                shift = (shift_N_to_B_vert + np.array(self.lattice[0])) / 2

        elif stacking == "AAp":  # For AA' stacking
            shift = np.power(0, 1 - zi % 2) * np.array([])

        self.defect_atom_info["coord_cart"] += shift
        self.defect_atom_info["coord_frac"] = self.defect_atom_info["coord_cart"] @ np.linalg.inv(self.sup_lattice)

    r"""
       ____        __              __     ______                __  _
      / __ \__  __/ /_____  __  __/ /_   / ____/_______  ____ _/ /_(_)___  ____
     / / / / / / / __/ __ \/ / / / __/  / /   / ___/ _ \/ __ `/ __/ / __ \/ __ \
    / /_/ / /_/ / /_/ /_/ / /_/ / /_   / /___/ /  /  __/ /_/ / /_/ / /_/ / / / /
    \____/\__,_/\__/ .___/\__,_/\__/   \____/_/   \___/\__,_/\__/_/\____/_/ /_/
                  /_/
    """

    def _writeFHIaims(self, cwd, full_path, name):
        """
        Write input for FHI-aims including the geometry.in file, the submission file and control.in file
        In case the the submission file/control.in files are not necessary for the user's case, they can remove those pieces
        """
        # Create/overwrite geometry.in file
        F = open("geometry.in", "w")
        F.write("# FHiaims input file generated by defect generator\n")
        for l in self.sup_lattice:
            F.write(f"lattice_vector {l[0]:15.10f} {l[1]:15.10f} {l[2]:15.10f}\n")
        for atom_info in self.defect_atom_info:
            coord = atom_info["coord_frac"]
            atom_type = atom_info["atom_type"]
            F.write(f"atom_frac {float(coord[0]):15.10f} {float(coord[1]):15.10f} {float(coord[2]):15.10f} {atom_type[0]}\n")
        F.close()

        # Create/overwrite .sub file
        F = open(name + ".sub", "w")
        F.write(
            "User can write their own submision file if the generated structure is intended as input for some sort of code. See an example below \n"
        )

        F.write("#!/bin/sh \n")
        F.write(f"#PBS -N {name} \n")
        F.write("#PBS -l ncpus=32 \n")
        F.write("#PBS -l mem=128GB \n")
        F.write("#PBS -l walltime=48:00:00 \n")
        F.write(" \n")
        F.write("module purge \n")
        F.write("module load PrgEnv-intel \n")
        F.write("module swap intel intel-classic/2023.1.0 \n")
        F.write("module load craype \n")
        F.write("module load cray-mpich/8.1.15 \n")
        F.write("module load mkl/2023.1.0 \n")
        F.write(" \n")
        F.write("# Environment variables to prevent oversubscription and ensure numerical stability \n")
        F.write("export OMP_NUM_THREADS=1 \n")
        F.write("export MKL_NUM_THREADS=1 \n")
        F.write("export MKL_DYNAMICS=FALSE \n")
        F.write("ulimit -s unlimited \n")
        F.write(" \n")
        F.write(f"cd {full_path} \n")
        print(f"Watch out you currently set {full_path} as your path")
        F.write("mpirun -np 32 /dir/to/executable > aims.out 2> aims.err")
        F.close()

        # Copy control file to folder
        os.system(f"cp {cwd}/control.in .")

    def _writeOutput(self, path, write):
        """
        Function to creaate the output to a certain path and for which code, e.g. FHI-aims
        """
        cwd = os.getcwd()
        if self.Nz > 1:
            dt_name = "[" + "!".join([f"{ii}" for ii in self.dt]) + "]" + "-" + self.stacking
        else:
            dt_name = self.dt

        name = f"{self.material}-{self.Nx}-{self.Ny}-{self.Nz}-{dt_name}"
        full_path = cwd + "/" + path + "/" + name

        # Check if path exists, if not, create it first
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        os.chdir(full_path)

        if write == "FHI-aims":
            self._writeFHIaims(cwd, full_path, name)

        os.chdir(cwd)

    def createDefectStructure(
        self,
        dt: str | list[str],
        Nx: int,
        Ny: int,
        Nz: int,
        z_dist: float = None,
        write: str = "FHI-aims",
        path: str = "",
        stacking: str = "AA",
    ):

        self.dt = dt
        self.stacking = stacking
        self._adjustUnitVectorZ(z_dist)
        if isinstance(dt, str):
            Nz = 1
            self._initSupercell(Nx, Ny, Nz)
            self._createSupercell(0)
            self._applyDefect(dt)

        elif isinstance(dt, list):
            Nz = len(dt)
            data_stack = []
            self._initSupercell(Nx, Ny, Nz)
            for zi, dt_zi in enumerate(dt):
                self._createSupercell(zi)
                self._applyDefect(dt_zi)
                self._applyStacking(zi, stacking)
                data_stack.append(self.defect_atom_info)

            self.defect_atom_info = np.concatenate(data_stack)
        self._writeOutput(path, write)
