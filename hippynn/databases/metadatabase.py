"""
MetaDatabase

Parses a `Database` object to extract species, positions, forces, and other relevant data,
organizing them into structured metadata. Calculates metrics such as force magnitudes,
pairwise atomic distances, and simulation box densities to facilitate data searching
and visualization.

Designed for easy extension with additional metadata calculations and visualization methods.
"""
from .database import Database

from ase.data import atomic_masses, chemical_symbols
from ase.units import _amu
import json
import numpy as np
from collections import defaultdict, Counter
from itertools import islice 
import re
import matplotlib.pyplot as plt  

class MetaDatabase(Database):
    """
    MetaDatabase

    A class to parse a database object and generate a metadata representation.
    This metadata facilitates searching, filtering, and visualization of the underlying representations.

    Attributes
    ----------
    db : Database
        The original database object to be parsed.
    metadata : dict
        A dictionary containing extracted metadata from the database.
    densities : list of float
        Calculated density values for each entry in the dataset.
    atom_counts : dict of {int: int}
        A mapping from atomic numbers to their respective atom counts in the dataset.

    Methods
    -------
    calculate_atom_counts()
        Computes the number of atoms for each unique atomic number present in the dataset.
    
    calculate_densities()
        Calculates the density for each entry based on mass and volume.
    
    search(criteria)
        Searches the metadata based on specified criteria and returns matching entries.
                search_entries_by_distance_range(distance_range):
                search_entries_by_max_force(force_range):
                search_entries_by_species(target_species):
    
    plot_density_distribution()
        Generates a histogram plot of the density distribution across the dataset.
    
    plot_atom_counts()
        Creates a bar chart representing the count of each atomic number in the dataset.
    
    plot_coordinates(entry_id)
        Plots the spatial coordinates of atoms for a given entry in 3D space.
    
    Examples
    --------
    >>> # Initialize MetaDatabase with an existing database object
    >>> meta_db = MetaDatabase(
    >>> arr_dict = db.arr_dict,
    >>> inputs=inputs,
    >>> targets=targets,
    >>> seed=12345,
    >>> num_workers=1,
    >>> pin_memory=True,
    >>> allow_unfound=True,
    >>> quiet=True,        
    >>> species_key='species',
    >>> coordinates_key='coordinates',
    >>> energies_key='energy',
    >>> forces_key='forces',
    >>> metadata={ 
    >>>    "Energy_unit" : 'eV',
    >>>    "Mass_unit" : 'grams/mol', 
    >>>    "Distance_unit" : 'Angstroms',
    >>>    "Electronic_Structure_Package" : 'VASP',
    >>>    "Electronic_Structure_Package_Version" : '6.4.3',
    >>>    "Computer_System" : 'LANL',
    >>>    "Input_Procedure" : { 
    >>>        "ENCUT": '',
    >>>        "ALGO": '',
    >>>        "SIGMA": '',
    >>>        "KPAR": ''
    >>>     }
    >>> },
    >>> populate_metadata=True,
    >>> write_metadata_to_json=True,
    >>> json_filename='metadata.json',
    >>> )
    
    >>> # Calculate atom counts and densities
    >>> meta_db.calculate_atom_counts()
    >>> meta_db.calculate_densities()
    
    >>> # Search for entries with density between 1.0 and 5.0
    >>> results = meta_db.search({'density': {'min': 1.0, 'max': 5.0}})
    
    >>> # Plot the Force Magnitude Distribution , Density Distribution and Pairwise Distance Distribution
    >>> meta_db.plot_distributions()
    
    >>> # Update metadata with a single "Comments" key
    >>> meta_db.update_metadata({"Comments": '' })

    >>> # Remove "Input_Proceedure" key from metadata
    >>> meta_db.remove_metadata("Input_Proceedure")

    >>> # Print metadata
    >>> meta_db.print_metadata()

    >>> # Search for indicies out of all entries containing atleast Carbon 
    >>> meta_db.search_entries_by_species(['C'], exact_match=False)
    
    >>> # Search for indicies out of all databaseentries containing exactly Hydrogen, Carbon and Oxygen
    >>> meta_db.search_entries_by_species(['CHO'], exact_match=True)
    
    >>> # Search for indicies out of all database entries with a calculated maximum atomic force in the range of [0,0.1]
    >>> meta_db.search_entries_by_max_force([0.0,0.1])

    >>> # Search for indicies out of all database entries with a calculated maximum pairwise atomic distance in the range of [0,0.9]
    >>> meta_db.search_entries_by_distance_range([0.0,0.9])


    **Key Functionalities:**

    1. **Parsing and Metadata Extraction:**
       - Extracts species and coordinate information from the database.
       - Computes unique atomic numbers and counts of each atom type.
       - Calculates physical properties like mass and density based on extracted data.

    2. **Searching Capabilities:**
       - Enables complex queries based on multiple criteria (e.g., density ranges,
         specific atomic compositions).
       - Supports logical operations (AND, OR, NOT) to refine search results.
       - Returns entries that match the specified search parameters.

    3. **Plotting and Visualization:**
       - Provides methods to visualize database distributions (e.g., pairwise atomic distances, density, force, histograms).
       - Generates plots for atom counts to understand elemental compositions of database.


    """
    def __init__(
        self,
        arr_dict,
        inputs,
        targets,
        species_key='species',
        coordinates_key='coordinates',
        energies_key='energies',
        forces_key='forces',
        metadata: dict[str, object] = None,
        entry_metadata: dict[int, dict[str, object]] = None,  
        populate_metadata=True,
        write_metadata_to_json=True,
        json_filename='metadata.json',
        **kwargs
    ):
        self.metadata = metadata.copy() if metadata else {}
        self.entry_metadata = entry_metadata.copy() if entry_metadata else {}

        super().__init__(
            arr_dict=arr_dict,
            inputs=inputs,
            targets=targets,
            **kwargs
        )
        self.species_key = species_key
        self.coordinates_key = coordinates_key
        self.energies_key = energies_key
        self.forces_key = forces_key
        self.write_metadata_to_json = write_metadata_to_json
        self.json_filename = json_filename

        self.atomic_numbers_in_dataset = None
        self.element_combinations = None
        self.atom_counts = None
        self.entry_species_index = None
        self.densities = None
        self.max_force = None
        self.min_force = None
        self.max_distance = None
        self.min_distance = None
      
        # Populate metadata if specified, otherwise call after instantiation
        if populate_metadata:
            self.populate_metadata(update=True, quiet=False)
            
    #
    #  Entry level Metadata
    #


    def set_entry_metadata(self, index: int, metadata: dict[str, object]):
        """Set metadata for a specific entry."""
        self.entry_metadata[index] = metadata

    def get_entry_metadata(self, index: int) -> dict[str, object]:
        """Get metadata for a specific entry."""
        return self.entry_metadata.get(index, {})

    def update_entry_metadata(self, index: int, metadata: dict[str, object]):
        """Update metadata for a specific entry."""
        if index not in self.entry_metadata:
            self.entry_metadata[index] = {}
        self.entry_metadata[index].update(metadata)

    def remove_entry_metadata(self, index: int):
        """Remove metadata for a specific entry."""
        if index in self.entry_metadata:
            del self.entry_metadata[index]
            
    def print_all_entry_metadata(self):
        """Print metadata for all entries."""
        print("Entry Metadata:")
        for index, metadata in self.entry_metadata.items():
            print(f"Entry {index}: {metadata}")

    def metadata_generator(self):
        """
        Generator that yields individual data entries from the database.
        """
        species_array = self.arr_dict[self.species_key]
        coordinates_array = self.arr_dict[self.coordinates_key]
        num_entries = len(species_array)
        for i in range(num_entries):
            yield {
                self.species_key: species_array[i],
                'coordinates': coordinates_array[i],
            }

    #
    #  Global Metadata
    #
    
    def set_metadata(self, key: str, value: object):
        """Set a global metadata field."""
        self.metadata[key] = value

    def get_metadata(self, key: str):
        """Get a global metadata field."""
        return self.metadata.get(key, None)

    def update_metadata(self, new_metadata: dict[str, object]):
        """Update the global metadata dictionary."""
        self.metadata.update(new_metadata)

    def remove_metadata(self, key: str):
        """Remove global metadata for a specific key."""
        if key in self.metadata:
            del self.metadata[key]

    def print_metadata(self):
        """Print global metadata in a readable format."""
        print("Metadata:")
        for key, value in self.metadata.items():
            print(f"  {key}: {value}")
            
    #
    #  Atom type and mass Mapping Functions
    #

    def convert_atomic_number_to_symbol(self):
        """
        Returns a dictionary mapping atomic numbers to element symbols using ASE's chemical_symbols array.
        """
        return {i: symbol for i, symbol in enumerate(chemical_symbols) if symbol}

    def convert_symbol_to_atomic_number(self):
        """
        Returns a dictionary mapping element symbols to their atomic numbers using ASE's chemical_symbols array.
        """
        return {symbol: i for i, symbol in enumerate(chemical_symbols) if symbol}


    def atomic_masses(self):
        """
        Returns a dictionary of atomic masses based on the specified unit.
        Supported units: 'grams/mol', 'amu', 'kg'.
        """
        mass_unit = self.metadata.get("Mass_unit", "grams/mol")  # Default to grams/mol
        
        # Define conversion factors
        conversion_factors = {
            "grams/mol": 1.0,  # In other words, no conversion needed
            "amu": 1.0 / 1.66053906660e-24,  # Convert to Atomic mass units
            "kg": _amu  # Convert grams/mol to kg using ASE's atomic mass unit
        }
        
        if mass_unit not in conversion_factors:
            raise ValueError(f"Unsupported mass unit: {mass_unit}. Supported units: {list(conversion_factors.keys())}")
        
        factor = conversion_factors[mass_unit]
        
        # Dynamically map atomic masses from ASE
        return {symbol: atomic_masses[i] * factor for i, symbol in enumerate(chemical_symbols) if symbol}

    def get_mass_from_species(self, species):
        if species == 0:
            return 0.0  # Assuming atomic number 0 represents padding in hippynn database
        masses = self.atomic_masses()
        number_to_symbol = self.convert_atomic_number_to_symbol()
        symbol = number_to_symbol.get(species)
        if symbol and symbol in masses:
            return masses[symbol]
        return 0.0  # Return 0.0 if species is invalid or not found
  

    #
    #  Data parsing Functions
    #

    def extract_element_combinations_large(self, chunk_size=100000):
        """
        Parse the database in chunks to extract and count element combinations.

        Args:
            chunk_size: The max number of entries to process in one batch (to avoid loading too much into memory).

        Returns:
            A dictionary with element combinations (given by atomic number) and their counts.
        """
        element_combinations = Counter()
        data_gen = self.metadata_generator()

        while True:
            chunk = list(islice(data_gen, chunk_size))
            if not chunk:
                break  # Stop when the generator is exhausted

            for data in chunk:
                # Count unique combinations of species
                species = data[self.species_key]
                unique_species = tuple(sorted(set(species[species != 0])))
                element_combinations[unique_species] += 1

        self.element_combinations = dict(element_combinations)
        return self.element_combinations

    def extract_unique_numbers_large(self):
        """
        Parse the database in chunks to extract unique atomic numbers.

        Returns:
            A sorted list of unique atomic numbers in the dataset.
        """
        self.extract_element_combinations_large(chunk_size=100000)
        unique_numbers = set(num for combo in self.element_combinations.keys() for num in combo)
        self.atomic_numbers_in_dataset = sorted(unique_numbers)
        return self.atomic_numbers_in_dataset


    #
    #  Data calculation Functions
    #

    def calculate_volume(self,coordinates):
        """
        For a given database entry, Compute the bounding box volume for given set of coordinates, NOT equal to the simulation box volume
        
        Args: coordinates: the cartesian positions of each atom in a single database entry
        
        Returns:
            A float representing the volume of the bounding box
        """
        if len(coordinates) > 0:
            x_min, y_min, z_min = coordinates.min(axis=0)
            x_max, y_max, z_max = coordinates.max(axis=0)
            return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        return None # empty database entry



    def calculate_max_distance(self):
        """
        For a given database entry, compute the maximum pairwise atomic distance in each structure (i.e. entry in the database).

        Returns:
            max_distances: A list containing the maximum pairwise atomic distance for each entry in the database.
        """
        max_distances = [] 
        
        for i, structure_coords in enumerate(self.arr_dict[self.coordinates_key]):
            valid_indices = np.where(self.arr_dict[self.species_key][i] != 0)[0] # Filters out non-atoms (species == 0)
            valid_coords = structure_coords[valid_indices]
            
            # Compute pairwise distances
            num_atoms = valid_coords.shape[0]
            if num_atoms < 2:
                max_distances.append(np.inf)  # No valid distances in single-atom or empty database entry
                continue
            
            # Calculate distance matrix
            diff = valid_coords[:, np.newaxis, :] - valid_coords[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            
            # Mask diagonal (self-distances)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Find the maximum distance
            max_distances.append(np.max(dist_matrix))
        
        self.max_distance = max_distances
        return max_distances
        
        
    def calculate_min_distance(self):
        """
        For a given database entry, compute the minimum pairwise distance of atoms in each structure.

        Returns:
            min_distances: A list containing the minimum pairwise atomic distance for each entry in the database.
        """
        min_distances = []  
        
        for i, structure_coords in enumerate(self.arr_dict[self.coordinates_key]):
            # Filter out invalid atoms (species == 0)
            valid_indices = np.where(self.arr_dict[self.species_key][i] != 0)[0]
            valid_coords = structure_coords[valid_indices]
            
            # Compute pairwise distances
            num_atoms = valid_coords.shape[0]
            if num_atoms < 2:
                min_distances.append(np.inf)  # No valid distances in single-atom or empty structures
                continue
            
            # Calculate distance matrix
            diff = valid_coords[:, np.newaxis, :] - valid_coords[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            
            # Mask diagonal (self-distances)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Find the minimum distance
            min_distances.append(np.min(dist_matrix))
        
        self.min_distance = min_distances
        return min_distances

    def calculate_atom_counts(self):
        """
        Calculate the count of atoms for each unique atomic number in the entire dataset.
        
        Returns:
        atom_counts: A dictionary mapping each atomic number to its corresponding atom count.
  
      """
        self.extract_unique_numbers_large()
      
        self.atom_counts = {}
        for atomic_number in self.atomic_numbers_in_dataset:
            count = self.count_atoms_by_type(atomic_number)
            self.atom_counts[atomic_number] = count
        return self.atom_counts

    def calculate_max_force(self):
        """
        Calceulates the maximum force magnitude for each entry and stores it in self.max_force.
        """
        forces = self.arr_dict[self.forces_key] 
        force_magnitudes = np.linalg.norm(forces, axis=2) 
        max_forces = np.max(force_magnitudes, axis=1) 
      
        self.max_force = max_forces


    def calculate_min_force(self):
        """
        Calculates the minimum force magnitude for each entry and stores it in self.min_force.
        """
        forces = self.arr_dict[self.forces_key]  
        force_magnitudes = np.linalg.norm(forces, axis=2)  
        min_forces = np.min(force_magnitudes, axis=1)  

        self.min_force = min_forces
    
    def calculate_densities(self): 
        """
        Calculates the density for each entry in the dataset.
      
        Notes:
            - Species lists may contain padding represented by zeros, which are excluded
                from calculations.
            - If the mass is zero, volume is invalid, or volume is zero, the density is set
                to `None` (or could be set to `np.nan` to indicate an invalid density).
        """
      
        densities = []
        # Precompute masses for species
        species_array = self.arr_dict[self.species_key]
        unique_species = np.unique(species_array.flatten())
        species_masses = {species: self.get_mass_from_species(species) for species in unique_species}

        for species_list, coordinates in zip(species_array, self.arr_dict[self.coordinates_key]):
            # Remove zeros (padding)
            species_list = species_list[species_list != 0]
            # Compute total mass of entry
            mass = sum(species_masses.get(species, 0.0) for species in species_list)
            volume = self.calculate_volume(coordinates) 

            if mass > 0 and volume is not None and volume > 0:
                density = mass / volume
            else:
                density = None  

            densities.append(density)

        self.densities = densities

    #
    #   Helper Functions for Search Functions
    #
    
    def count_atoms_by_type(self, atomic_number):
        """
        Counts the total number of atoms with a given atomic_number in the database.

        Args:
            atomic_number: The atomic number to count.

        Returns:
            The total count of atoms with the specified atomic number.
        """
        species_array = self.arr_dict[self.species_key]
        # Count occurrences of atomic number
        atom_type_count = np.sum(species_array == atomic_number)
        return atom_type_count



    def build_entry_species_index(self):
        """
        Builds an index that maps species combinations to the indices of entries containing them.
        """
        self.entry_species_index = defaultdict(list)
        species_array = self.arr_dict[self.species_key]
        num_entries = len(species_array)
        for i in range(num_entries):
            species = species_array[i]
            # Remove zeros (padding) and create a sorted tuple of unique species
            unique_species = tuple(sorted(set(species[species != 0])))
            self.entry_species_index[unique_species].append(i)

    
    def generate_custom_priority(self):
        """
        Generates a custom priority list of element symbols based on their atomic numbers.
    
        Returns:
            List of element symbols ordsered by their atomic numbers.
        """
        atomic_number_to_symbol = self.convert_atomic_number_to_symbol()
        # Extract and sort atomic numbers
        sorted_atomic_numbers = sorted(atomic_number_to_symbol.keys())
        # Map sorted atomic numbers to their symbols
        return [atomic_number_to_symbol[num] for num in sorted_atomic_numbers]


    def split_combined_species(self, combined_species, valid_symbols):
        """
        Splits a string of combined species (e.g., 'HCO' or 'ClZn') into valid symbols.

        Args:
            combined_species (str): The string to split.
            valid_symbols (set): A set of valid element symbols (e.g., {'H', 'C', 'O', 'Cl', 'Zn'}).

        Returns:
            list: A list of valid symbols if the split is successful.

        Raises:
            ValueError: If any part of the string is not a valid element symbol.
        """
        # Regular expression to match element symbols (1 uppercase letter, optionally 1 lowercase letter)
        element_pattern = r'[A-Z][a-z]?'
        matches = re.findall(element_pattern, combined_species)

        # Ensure all matched symbols are valid
        for match in matches:
            if match not in valid_symbols:
                raise ValueError(
                    f"Invalid element symbol: '{match}'. Valid elements in this dataset include: {', '.join(valid_symbols)}"
                )

        return matches

    #
    #   Search Functions
    #

    def search_entries_by_species(self, target_species, exact_match=True, use_symbols=True):
        """
        Searches for database entries that contain the specified atomic species.

        Args:
            target_species (list or set): The atomic species to search for (symbols or atomic numbers).
            exact_match (bool): If True, finds entries containing exactly the target species.
                                If False, finds entries containing at least the target species.
            use_symbols (bool): If True, interprets `target_species` as element symbols.

        Returns:
            List of indices of matching entries.
        """
        if self.entry_species_index is None:
            self.build_entry_species_index()

        if use_symbols:
            # Convert element symbols to atomic numbers
            symbol_to_number = self.convert_symbol_to_atomic_number()
            # Retrieve valid symbols dynamically from the dataset
            valid_atomic_numbers = np.unique(self.arr_dict[self.species_key])  # Unique atomic numbers in the dataset
            valid_symbols = {symbol for symbol, num in symbol_to_number.items() if num in valid_atomic_numbers}

            # Split combined strings into individual symbols
            expanded_target_species = []
            for item in target_species:
                if isinstance(item, str):  # Handle combined strings like 'HCO' or 'ClZn'
                    expanded_target_species.extend(self.split_combined_species(item, valid_symbols))
                else:
                    expanded_target_species.append(item)

            # Ensure all symbols in the combination are valid
            invalid_symbols = [symbol for symbol in expanded_target_species if symbol not in valid_symbols]
            if invalid_symbols:
                raise ValueError(
                    f"The element symbol(s): {', '.join(invalid_symbols)} are not found in this dataset. "
                    f"Found elements in this dataaset include: {', '.join(valid_symbols)}"
                )

            # Convert to atomic numbers
            target_species = [symbol_to_number[symbol] for symbol in expanded_target_species]

        target_species_set = set(target_species)
        matching_indices = []

        for species_combo, indices in self.entry_species_index.items():
            species_set = set(species_combo)
            if exact_match:
                if species_set == target_species_set:
                    matching_indices.extend(indices)
            else:
                if target_species_set.issubset(species_set):
                    matching_indices.extend(indices)

        return matching_indices


    
    def search_entries_by_max_force(self, force_range):
        """
        Searches for entries where the maximum force is within the specified range.

        Args:
            force_range (list or tuple): A two-element list or tuple specifying the min and max force valueas [min_force, max_force].

        Returns:
            List of indices of matching entries.
        """
        if self.max_force is None:
            self.calculate_max_force()

        min_force, max_force = force_range

        # Validate the force_range input
        if min_force > max_force:
            raise ValueError("min_force should be less than or equal to max_force.")

        matching_indices = np.where((self.max_force >= min_force) & (self.max_force <= max_force))[0]
        return matching_indices.tolist()

    def search_entries_by_distance_range(self, distance_range):
        """
        Searches for entries where the minimum or maximum atomic distance is within the specified range.
    
        Args:
            distance_range (list or tuple): A two-element list or tuple specifying the min and max distance values [min_distance, max_distance].
    
        Returns:
            List of indices of matching entries.
        """
        if self.min_distance is None or self.max_distance is None:
            # Compute distances if they haven't been calculated yet
            self.calculate_min_distance()
            self.calculate_max_distance()
    
        if not isinstance(distance_range, (list, tuple)) or len(distance_range) != 2:
            raise ValueError("distance_range must be a list or tuple with exactly two elements [min_distance, max_distance].")
    
        min_distance, max_distance = distance_range
    
        # Validate the distance_range input
        if min_distance > max_distance:
            raise ValueError("min_distance should be less than or equal to max_distance.")
    
        # Ensure min_distance and max_distance are NumPy arrays
        min_distance_array = np.array(self.min_distance)
        max_distance_array = np.array(self.max_distance)
    
        # Find indices of entries where either min_distance or max_distance falls within the range
        matching_indices = np.where(
            (min_distance_array >= min_distance) & (min_distance_array <= max_distance) |
            (max_distance_array >= min_distance) & (max_distance_array <= max_distance)
        )[0]
    
        return matching_indices.tolist()

    #
    #   Get Functions
    #

    def get_element_combinations(self):
        """
        Returns extracted element combinations with atomic symbols and their frequency.
        """
        if self.element_combinations is None:
            raise ValueError("Element combinations not extracted yet. Run 'extract_element_combinations_large()' first.")
        
        # Map atomic numbers to element symbols
        atomic_number_to_symbol = self.convert_atomic_number_to_symbol()
        result = {}
        
        for combination, count in self.element_combinations.items():
            # Convert atomic numbers to symbols
            symbols = [atomic_number_to_symbol[num] for num in combination]
            # Join symbols into a readable string
            symbol_str = "".join(symbols)
            result[symbol_str] = count
    
        return result

    def get_density_range(self):
        """
        Returns the range of denasities (min and max) across the dataset.
        """
        if self.densities is None:
            self.calculate_densities()
        
        valid_densities = [d for d in self.densities if d is not None]
    
        if valid_densities:
            min_density = min(valid_densities)
            max_density = max(valid_densities)
            return [min_density, max_density]
        else:
            return [None, None]  # No valid densities found

    

    def get_atom_counts_by_symbol(self):
        """
        Returns a dictionary smapping element symbols to their counts in the dataset.
        """
        if self.atom_counts is None:
            raise ValueError("Atom counts not computed yet. Run 'calculate_atom_counts()' first.")
        
        # Map atomic numbers to element symbols
        atomic_number_to_symbol = self.convert_atomic_number_to_symbol()
        
        counts_by_symbol = {}
        for atomic_number, count in self.atom_counts.items():
            symbol = atomic_number_to_symbol.get(atomic_number, f"Unknown({atomic_number})")
            counts_by_symbol[symbol] = count
        
        return counts_by_symbol


    def get_min_distance_range(self):
        """
        Compute and print the range of minimum distances (min and max) across the dataset.
    
        Returns:
            min_distance_range: A tuple containing the minimum and maximum of the minimum distances.
        """
        if self.min_distance is None:
            raise ValueError("Minimum atomic distances not calculated yet. Run 'calculate_min_distance()' first.")
        
        # Calculate the min and max of the minimum distances
        min_dist = min(self.min_distance)
        max_dist = max(self.min_distance)
    
        return [min_dist, max_dist]

    def get_max_force_range(self):
        """
        Gets the range of force magnitude (min and max) of the max force across the dataset.

        Returns:
            (min_of_max_force, max_force): A tuple containing the minimum and maximum of the maximum atomic forces.
        """
        if self.max_force is None:
            raise ValueError("Minimum atomic distances not calculated yet. Run 'calculate_max_force()' first.")
        if self.min_force is None:
            raise ValueError("Minimum atomic distances not calculated yet. Run 'calculate_min_force()' first.")
        
        min_of_max_force = min(self.max_force) #self.min_force[self.min_force > 0.0].min() if (self.min_force > 0.0).any() else None
        max_force = max(self.max_force)
    
        return [min_of_max_force, max_force]


    def populate_metadata(self, update=True, quiet=False):
        """
        Runs the Calculates for various entry-level properties and optionally updates global metadata.
    
        Args:
            update (bool): Whether to update the global metadata with the calculated values.
            quiet (bool): If True, suppresses printing of metadata.
        """
        # Calculate entry)level properties
        self.calculate_densities()
        self.calculate_max_force()
        self.calculate_min_force()
        self.calculate_min_distance()
        self.calculate_atom_counts()
        self.extract_unique_numbers_large()
    
        if update:
            # Update global metadata with calculated properties
            self.update_metadata(
                {
                    "number_of_entries_for_each_combination_of_atom_types": self.get_element_combinations(),
                    "range_min_atomic_distance": self.get_min_distance_range(),
                    "range_max_force_magnitude": self.get_max_force_range(),
                    "range_density": self.get_density_range(),
                    "atom_counts_by_symbol": self.get_atom_counts_by_symbol(),
                }
            )
    
        if not quiet:
            # Print metadata if not in quiet mode
            self.print_metadata()

        # Save metadata to JSON if enabled
        if self.write_metadata_to_json:
            self.save_metadata_to_json()
    
    def make_json_serializable(self):
        """
        Converts the metadata into a JSON-compatible format.
        """
        def convert_value(value):
            if isinstance(value, (np.float32, np.float64)):
                return float(value)  # Convert NumPy floats to Python floats
            elif isinstance(value, (np.int32, np.int64)):
                return int(value)  # Convert NumPy integers to Python integers
            elif isinstance(value, np.ndarray):
                return value.tolist()  # Convert NumPy arrays to lists
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}  # Recursively convert dicts
            elif isinstance(value, list):
                return [convert_value(v) for v in value]  # Recursively convert lists
            else:
                return value  # Leave other types unchanged

        return {key: convert_value(val) for key, val in self.metadata.items()}

    def save_metadata_to_json(self):
        """
        Saves the metadata to a JSON file.
        Args:
            filename (str): The name of the JSON file to save.
        """
        json_compatible_metadata = self.make_json_serializable()
        with open(self.json_filename, "w") as f:
            json.dump(json_compatible_metadata, f, indent=4)

    def OLD_plot_distributions(self):
        import matplotlib.pyplot as plt
        """
        Plots the distributions of densities, force magnitudes, and atomic distances.
        """
        # Ensure densities are calculated
        if self.densities is None:
            self.calculate_densities()

        # Ensure max force is calculated
        if self.max_force is None:
            self.calculate_max_force()

        # Ensure min and max distances are calculated
        if self.min_distance is None or self.max_distance is None:
            self.calculate_min_distance()
            self.calculate_max_distance()

        # Filter valid densities
        valid_densities = [d for d in self.densities if d is not None and not np.isnan(d)]

        # Create subplots for better visualization
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Density Distribution
        axs[0].hist(valid_densities, bins=50, alpha=0.7, color='blue')
        axs[0].set_title("Density Distribution")
        axs[0].set_xlabel("Density ")
        axs[0].set_ylabel("Frequency")

        # Force Magnitude Distribution
        axs[1].hist(self.max_force, bins=50, alpha=0.7, color='orange')
        axs[1].set_title("Maximum Force Magnitude Distribution")
        axs[1].set_xlabel("Force Magnitude ")
        axs[1].set_ylabel("Frequency")

        # Distance Distribution
        axs[2].hist(self.min_distance, bins=50, alpha=0.7, color='green', label="Min Distance")
        axs[2].hist(self.max_distance, bins=50, alpha=0.7, color='purple', label="Max Distance")
        axs[2].set_title("Atomic Distance Distribution")
        axs[2].set_xlabel("Atomic Distance")
        axs[2].set_ylabel("Frequency")
        axs[2].legend()

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()


    def plot_distributions(self):
        import matplotlib.pyplot as plt
        import numpy as np
    
        # Ensure densities are calculated
        if self.densities is None:
            self.calculate_densities()
    
        # Ensure max force is calculated
        if self.max_force is None:
            self.calculate_max_force()
    
        # Ensure min and max distances are calculated
        if self.min_distance is None or self.max_distance is None:
            self.calculate_min_distance()
            self.calculate_max_distance()
    
        # Filter valid values for plotting
        valid_densities = [d for d in self.densities if d is not None and np.isfinite(d)]
        valid_min_distance = [d for d in self.min_distance if np.isfinite(d)]
        valid_max_distance = [d for d in self.max_distance if np.isfinite(d)]
        valid_max_force = [f for f in self.max_force if np.isfinite(f)]
    
        # Calculate ±3 standard deviations for the range
        def calculate_range(data):
            if len(data) > 0:
                mean = np.mean(data)
                std = np.std(data)
                return mean - 3 * std, mean + 3 * std
            else:
                return None, None
    
        density_range = calculate_range(valid_densities)
        max_force_range = calculate_range(valid_max_force)
        min_distance_range = calculate_range(valid_min_distance)
        max_distance_range = calculate_range(valid_max_distance)
    
        fig, axs = plt.subplots(2, 2, figsize=(18, 12)) 
    
        # Density Distribution
        axs[0, 0].hist(valid_densities, bins=50, alpha=0.7, color='blue')
        axs[0, 0].set_title("Density Distribution")
        axs[0, 0].set_xlabel("Density")
        axs[0, 0].set_ylabel("Frequency")
        if density_range[0] is not None:
            axs[0, 0].set_xlim(density_range)
    
        # Force Magnitude Distribution
        axs[0, 1].hist(valid_max_force, bins=50, alpha=0.7, color='orange')
        axs[0, 1].set_title("Maximum Force Magnitude Distribution")
        axs[0, 1].set_xlabel("Force Magnitude")
        axs[0, 1].set_ylabel("Frequency")
        if max_force_range[0] is not None:
            axs[0, 1].set_xlim(max_force_range)
    
        # Distance Distribution
        axs[1, 0].hist(valid_min_distance, bins=50, alpha=0.7, color='green', label="Min Distance")
        axs[1, 0].hist(valid_max_distance, bins=50, alpha=0.7, color='purple', label="Max Distance")
        axs[1, 0].set_title("Atomic Distance Distribution")
        axs[1, 0].set_xlabel("Atomic Distance")
        axs[1, 0].set_ylabel("Frequency")
        axs[1, 0].legend()
        if min_distance_range[0] is not None and max_distance_range[0] is not None:
            axs[1, 0].set_xlim(
                min(min_distance_range[0], max_distance_range[0]),
                max(min_distance_range[1], max_distance_range[1])
            )
    
        # Atom Counts by Symbol
        atom_counts_by_symbol = self.get_atom_counts_by_symbol()  
        symbols = list(atom_counts_by_symbol.keys())
        counts = list(atom_counts_by_symbol.values())
    
        axs[1, 1].bar(symbols, counts, color='skyblue', alpha=0.8)
        axs[1, 1].set_title("Atom Counts by Symbol")
        axs[1, 1].set_xlabel("Element Symbol")
        axs[1, 1].set_ylabel("Atom Count")
        for i, count in enumerate(counts):
            axs[1, 1].text(i, count, f"{count:,}", ha='center', va='bottom', fontsize=10)
    
        plt.tight_layout()
        plt.savefig("database_distribution.png") 
        plt.show()
    
