import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import os
import tempfile
import subprocess
from tqdm import tqdm
import random


class StructureRefiner:
    """Class for refining and diversifying RNA 3D structures"""

    def __init__(self, energy_weight=1.0, secondary_structure_weight=1.0,
                 clash_weight=1.0, smooth_weight=0.5, diversity_weight=0.2):
        """
        Initialize the structure refiner.

        Args:
            energy_weight: Weight for energy term in refinement
            secondary_structure_weight: Weight for secondary structure constraints
            clash_weight: Weight for clash avoidance
            smooth_weight: Weight for chain smoothness term
            diversity_weight: Weight for diversity term in ensemble generation
        """
        self.energy_weight = energy_weight
        self.secondary_structure_weight = secondary_structure_weight
        self.clash_weight = clash_weight
        self.smooth_weight = smooth_weight
        self.diversity_weight = diversity_weight

        # Constants for RNA
        self.min_dist = 3.0  # Minimum distance between consecutive residues (Å)
        self.max_dist = 8.0  # Maximum distance between consecutive residues (Å)
        self.bp_dist = 6.0  # Target distance for base pairs (Å)
        self.clash_dist = 4.0  # Minimum distance between non-consecutive residues (Å)

    def refine_structure(self, coords, sequence, base_pairs=None, max_iter=100, step_size=0.05):
        """
        Refine a structure using energy minimization.

        Args:
            coords: Initial 3D coordinates (N x 3)
            sequence: Sequence string
            base_pairs: List of base-paired positions (i, j)
            max_iter: Maximum iterations for refinement
            step_size: Size of update steps

        Returns:
            Refined 3D coordinates
        """
        coords = coords.copy()
        N = len(coords)

        # If no base pairs provided, use an empty list
        if base_pairs is None:
            base_pairs = []

        print(f"Refining structure with {N} residues and {len(base_pairs)} base pairs...")

        # Iterate for refinement
        for iter_idx in tqdm(range(max_iter)):
            # Calculate pairwise distances
            distances = squareform(pdist(coords))

            # Initialize gradients
            grads = np.zeros_like(coords)

            # Consecutive residue distance constraint
            for i in range(N - 1):
                dist = distances[i, i + 1]

                # Enforce min/max distance between consecutive residues
                if dist < self.min_dist:
                    # Push apart if too close
                    direction = coords[i + 1] - coords[i]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)

                    grads[i] -= direction * self.smooth_weight
                    grads[i + 1] += direction * self.smooth_weight

                elif dist > self.max_dist:
                    # Pull together if too far
                    direction = coords[i + 1] - coords[i]
                    direction = direction / (np.linalg.norm(direction) + 1e-10)

                    grads[i] += direction * self.smooth_weight
                    grads[i + 1] -= direction * self.smooth_weight

            # Base pair constraints
            for i, j in base_pairs:
                dist = distances[i, j]

                # Enforce base pair distance
                direction = coords[j] - coords[i]
                direction = direction / (np.linalg.norm(direction) + 1e-10)

                # Pull together or push apart to reach target distance
                if dist < self.bp_dist:
                    # Push apart slightly
                    force = (self.bp_dist - dist) * self.secondary_structure_weight
                    grads[i] -= direction * force
                    grads[j] += direction * force
                else:
                    # Pull together
                    force = (dist - self.bp_dist) * self.secondary_structure_weight
                    grads[i] += direction * force
                    grads[j] -= direction * force

            # Clash avoidance for non-consecutive, non-base-paired residues
            base_pair_indices = set([(i, j) for i, j in base_pairs] + [(j, i) for i, j in base_pairs])

            for i in range(N):
                for j in range(N):
                    if (abs(i - j) > 1 and (i, j) not in base_pair_indices
                            and distances[i, j] < self.clash_dist):
                        # Push apart if too close
                        direction = coords[j] - coords[i]
                        direction = direction / (np.linalg.norm(direction) + 1e-10)

                        grads[i] -= direction * self.clash_weight
                        grads[j] += direction * self.clash_weight

            # Update coordinates
            coords += step_size * grads

            # Optional: reduce step size over time
            if iter_idx > max_iter // 2:
                step_size *= 0.99

        print("Refinement complete")
        return coords

    def generate_diverse_structures(self, base_coords, sequence, base_pairs=None,
                                    num_structures=5, perturb_scale=2.0):
        """
        Generate diverse structures by perturbing and refining.

        Args:
            base_coords: Base 3D coordinates
            sequence: Sequence string
            base_pairs: List of base-paired positions
            num_structures: Number of structures to generate
            perturb_scale: Scale of perturbation

        Returns:
            List of diverse structures
        """
        N = len(base_coords)
        structures = [base_coords.copy()]

        print(f"Generating {num_structures - 1} additional diverse structures...")

        # Generate additional structures
        for i in range(num_structures - 1):
            print(f"Generating structure {i + 2}/{num_structures}")

            # Perturb the base structure
            perturbed = base_coords.copy()

            # Add random noise, more to the middle of the chain than the ends
            for j in range(N):
                # Add less perturbation to the ends
                distance_from_end = min(j, N - 1 - j)
                scaling = (distance_from_end / (N / 2)) * perturb_scale

                # Add random noise
                perturbed[j] += np.random.normal(0, scaling, 3)

            # Refine the perturbed structure
            refined = self.refine_structure(
                perturbed, sequence, base_pairs,
                max_iter=50  # Use fewer iterations for diversity
            )

            # Add to list of structures
            structures.append(refined)

        # Ensure we have diverse structures by clustering
        if num_structures > 5:
            print("Clustering structures to ensure diversity...")
            return self._cluster_structures(structures, 5)

        return structures

    def _cluster_structures(self, structures, num_clusters):
        """
        Cluster structures and select representatives.

        Args:
            structures: List of structure coordinates
            num_clusters: Number of clusters to form

        Returns:
            List of representative structures
        """
        # Flatten structures for clustering
        flat_structures = [s.flatten() for s in structures]

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(flat_structures)

        # Select the structure closest to each cluster center
        representatives = []
        for i in range(num_clusters):
            # Get indices of structures in this cluster
            cluster_indices = np.where(cluster_labels == i)[0]

            if len(cluster_indices) == 0:
                continue

            # Find the structure closest to the center
            cluster_center = kmeans.cluster_centers_[i]
            distances = [np.linalg.norm(flat_structures[idx] - cluster_center)
                         for idx in cluster_indices]
            closest_idx = cluster_indices[np.argmin(distances)]

            representatives.append(structures[closest_idx])

        # If we have fewer than num_clusters representatives, add random structures
        while len(representatives) < num_clusters:
            # Add a random structure not already in representatives
            for s in structures:
                if not any(np.array_equal(s, r) for r in representatives):
                    representatives.append(s)
                    break

            # If we've exhausted all structures, duplicate the first one with noise
            if len(representatives) < num_clusters:
                noisy = representatives[0].copy()
                noisy += np.random.normal(0, 0.5, noisy.shape)
                representatives.append(noisy)

        return representatives

    def calculate_rmsd(self, coords1, coords2):
        """
        Calculate RMSD between two structures.

        Args:
            coords1, coords2: 3D coordinates to compare

        Returns:
            RMSD value
        """
        if coords1.shape != coords2.shape:
            raise ValueError("Coordinate shapes must match")

        return np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))

    def align_structures(self, target_coords, mobile_coords):
        """
        Align mobile structure to target using Kabsch algorithm.

        Args:
            target_coords: Target 3D coordinates
            mobile_coords: Mobile 3D coordinates to align

        Returns:
            Aligned mobile coordinates
        """
        if target_coords.shape != mobile_coords.shape:
            raise ValueError("Coordinate shapes must match")

        # Center the structures
        target_center = np.mean(target_coords, axis=0)
        mobile_center = np.mean(mobile_coords, axis=0)

        target_centered = target_coords - target_center
        mobile_centered = mobile_coords - mobile_center

        # Calculate the correlation matrix
        correlation_matrix = np.dot(mobile_centered.T, target_centered)

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(correlation_matrix)

        # Calculate the rotation matrix
        rotation_matrix = np.dot(Vt.T, U.T)

        # Check for reflection case
        if np.linalg.det(rotation_matrix) < 0:
            Vt[-1, :] *= -1
            rotation_matrix = np.dot(Vt.T, U.T)

        # Apply rotation and translation
        aligned_coords = np.dot(mobile_centered, rotation_matrix) + target_center

        return aligned_coords

    def evaluate_ensemble(self, structures, reference=None):
        """
        Evaluate the quality and diversity of a structure ensemble.

        Args:
            structures: List of structure coordinates
            reference: Reference structure if available

        Returns:
            Dictionary with evaluation metrics
        """
        if len(structures) < 2:
            return {"average_rmsd": 0, "diversity": 0}

        # Calculate pairwise RMSDs
        n_structures = len(structures)
        rmsd_matrix = np.zeros((n_structures, n_structures))

        for i in range(n_structures):
            for j in range(i + 1, n_structures):
                # Align structures first
                aligned = self.align_structures(structures[i], structures[j])
                rmsd = self.calculate_rmsd(structures[i], aligned)
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd

        # Calculate average RMSD (diversity measure)
        diversity = np.sum(rmsd_matrix) / (n_structures * (n_structures - 1))

        # Calculate RMSD to reference if provided
        average_rmsd_to_ref = None
        if reference is not None:
            rmsd_to_ref = []
            for s in structures:
                aligned = self.align_structures(reference, s)
                rmsd_to_ref.append(self.calculate_rmsd(reference, aligned))
            average_rmsd_to_ref = np.mean(rmsd_to_ref)

        return {
            "diversity": diversity,
            "rmsd_matrix": rmsd_matrix,
            "average_rmsd_to_ref": average_rmsd_to_ref
        }

    def visualize_ensemble(self, structures, title=None):
        """
        Visualize an ensemble of structures.

        Args:
            structures: List of structure coordinates
            title: Plot title
        """
        # Create a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Colors for different structures
        colors = plt.cm.rainbow(np.linspace(0, 1, len(structures)))

        # Plot each structure with a different color
        for i, coords in enumerate(structures):
            # Plot coordinates as points
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                       s=20, c=[colors[i]], label=f'Structure {i + 1}')

            # Connect consecutive residues with lines
            for j in range(len(coords) - 1):
                ax.plot(coords[j:j + 2, 0], coords[j:j + 2, 1], coords[j:j + 2, 2],
                        color=colors[i], alpha=0.5)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if title:
            ax.set_title(title)
        else:
            ax.set_title('Ensemble of Structures')

        ax.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()

        # Plot RMSD matrix if we have multiple structures
        if len(structures) > 1:
            # Calculate RMSD matrix
            eval_results = self.evaluate_ensemble(structures)
            rmsd_matrix = eval_results["rmsd_matrix"]

            plt.figure(figsize=(8, 6))
            plt.imshow(rmsd_matrix, cmap='viridis')
            plt.colorbar(label='RMSD (Å)')
            plt.title(f'Pairwise RMSD Matrix (Diversity: {eval_results["diversity"]:.2f} Å)')
            plt.xlabel('Structure Index')
            plt.ylabel('Structure Index')

            # Add text annotations
            for i in range(rmsd_matrix.shape[0]):
                for j in range(rmsd_matrix.shape[1]):
                    if i != j:
                        plt.text(j, i, f'{rmsd_matrix[i, j]:.1f}',
                                 ha='center', va='center',
                                 color='white' if rmsd_matrix[i, j] > np.mean(rmsd_matrix) else 'black')

            plt.tight_layout()
            plt.show()


class StructureEnsembleGenerator:
    """Class for generating and selecting ensembles of RNA 3D structures"""

    def __init__(self, refiner, predictor_3d, num_candidates=10, num_final=5):
        """
        Initialize the ensemble generator.

        Args:
            refiner: StructureRefiner object
            predictor_3d: RNA3DPredictor object
            num_candidates: Number of candidate structures to generate
            num_final: Number of structures to select for final ensemble
        """
        self.refiner = refiner
        self.predictor_3d = predictor_3d
        self.num_candidates = num_candidates
        self.num_final = num_final

    def generate_ensemble(self, sequence, target_id, secondary_structure_predictor=None, msa_data=None):
        """
        Generate an ensemble of structures for a sequence.

        Args:
            sequence: RNA sequence string
            target_id: Target identifier
            secondary_structure_predictor: SecondaryStructurePredictor object (optional)
            msa_data: MSA data for the sequence (optional)

        Returns:
            List of selected structures
        """
        print(f"Generating ensemble for {target_id}...")

        # Step 1: Get secondary structure information if predictor is provided
        base_pairs = []
        if secondary_structure_predictor is not None:
            print("Predicting secondary structure...")
            ss_results = secondary_structure_predictor.predict(sequence, msa_data)

            # Extract base pairs from secondary structure
            mfe_structure = ss_results['mfe_structure']

            # Parse parentheses notation to get base pairs
            stack = []
            for i, char in enumerate(mfe_structure):
                if char == '(':
                    stack.append(i)
                elif char == ')':
                    if stack:
                        base_pairs.append((stack.pop(), i))

            print(f"Found {len(base_pairs)} base pairs from secondary structure")

        # Step 2: Generate initial 3D structure predictions
        print("Generating initial 3D predictions...")
        # Assume the predictor_3d has a method to predict a single sequence
        initial_structures = self._predict_single_sequence(sequence, target_id, msa_data)

        # Step 3: Refine and diversify structures
        print("Refining and diversifying structures...")
        refined_structures = []

        for i, coords in enumerate(initial_structures):
            print(f"Refining structure {i + 1}/{len(initial_structures)}...")

            # Refine the structure
            refined = self.refiner.refine_structure(
                coords, sequence, base_pairs, max_iter=100
            )

            # Add to list of refined structures
            refined_structures.append(refined)

        # Generate additional diverse structures if needed
        if len(refined_structures) < self.num_candidates:
            # Use the first refined structure as a base
            base_coords = refined_structures[0] if refined_structures else initial_structures[0]

            print(f"Generating {self.num_candidates - len(refined_structures)} additional structures...")
            additional_structures = self.refiner.generate_diverse_structures(
                base_coords, sequence, base_pairs,
                num_structures=self.num_candidates - len(refined_structures)
            )

            refined_structures.extend(additional_structures)

        # Step 4: Select the final ensemble
        print("Selecting final ensemble...")
        final_ensemble = self._select_diverse_ensemble(refined_structures)

        print(f"Selected {len(final_ensemble)} structures for final ensemble")
        return final_ensemble

    def _predict_single_sequence(self, sequence, target_id, msa_data=None):
        """
        Predict structures for a single sequence using the 3D predictor.

        Args:
            sequence: RNA sequence string
            target_id: Target identifier
            msa_data: MSA data for the sequence (optional)

        Returns:
            List of predicted structures
        """
        # This is a placeholder. In a real implementation, we would use
        # the 3D predictor model to generate initial predictions.
        # For now, we'll generate a simple helix as an example.

        N = len(sequence)

        # Create a simple helix-like structure as a placeholder
        coords = np.zeros((N, 3))

        # Parameters for a simple helix
        radius = 10.0
        pitch = 1.5

        for i in range(N):
            angle = i * 2 * np.pi / 10  # 10 residues per turn

            coords[i, 0] = radius * np.cos(angle)
            coords[i, 1] = radius * np.sin(angle)
            coords[i, 2] = i * pitch

        # Create several variations
        structures = [coords.copy()]

        # Add variations with slight perturbations
        for _ in range(min(4, self.num_candidates - 1)):
            variant = coords.copy()
            variant += np.random.normal(0, 1.0, (N, 3))
            structures.append(variant)

        return structures

    def _select_diverse_ensemble(self, structures, num_select=None):
        """
        Select a diverse subset of structures from candidates.

        Args:
            structures: List of candidate structures
            num_select: Number of structures to select

        Returns:
            List of selected structures
        """
        if num_select is None:
            num_select = self.num_final

        if len(structures) <= num_select:
            return structures

        # Use clustering to select diverse structures
        return self.refiner._cluster_structures(structures, num_select)

    def format_for_submission(self, target_id, sequence, ensemble, test_sequences):
        """
        Format an ensemble for submission.

        Args:
            target_id: Target identifier
            sequence: RNA sequence string
            ensemble: List of structure coordinates
            test_sequences: DataFrame with test sequences

        Returns:
            List of submission rows
        """
        submission_rows = []

        # Ensure we have exactly 5 structures
        if len(ensemble) < 5:
            # Duplicate the last structure if needed
            while len(ensemble) < 5:
                ensemble.append(ensemble[-1])
        elif len(ensemble) > 5:
            # Select only the first 5
            ensemble = ensemble[:5]

        # For each residue
        for i, nt in enumerate(sequence):
            submission_row = {
                'ID': f"{target_id}_{i + 1}",
                'resname': nt,
                'resid': i + 1
            }

            # Add coordinates for each structure
            for j, structure in enumerate(ensemble):
                submission_row[f'x_{j + 1}'] = structure[i][0]
                submission_row[f'y_{j + 1}'] = structure[i][1]
                submission_row[f'z_{j + 1}'] = structure[i][2]

            submission_rows.append(submission_row)

        return submission_rows


# Example of how to use these classes with the other modules
if __name__ == "__main__":
    from rna_structure_predictor import RNAStructurePredictor
    from secondary_structure_predictor import SecondaryStructurePredictor
    from tertiary_structure_predictor import RNA3DPredictor

    # Initialize the base predictor
    rna_predictor = RNAStructurePredictor(temporal_cutoff="2022-05-27")
    rna_predictor.load_data()

    # Load MSAs for a subset of targets for demonstration
    sample_targets = rna_predictor.train_sequences['target_id'].iloc[:2].values
    rna_predictor.load_msas(sample_targets)

    # Get a sample sequence
    sample_target_id = sample_targets[0]
    sample_seq_row = rna_predictor.train_sequences[
        rna_predictor.train_sequences['target_id'] == sample_target_id
        ].iloc[0]
    sample_sequence = sample_seq_row['sequence']
    msa_data = rna_predictor.msa_data.get(sample_target_id)

    # Initialize predictors
    ss_predictor = SecondaryStructurePredictor(use_msa=True, use_ml=False)
    predictor_3d = RNA3DPredictor(hidden_dim=64, num_layers=2)
    refiner = StructureRefiner()

    # Initialize ensemble generator
    ensemble_generator = StructureEnsembleGenerator(
        refiner, predictor_3d, num_candidates=7, num_final=5
    )

    # Generate ensemble
    ensemble = ensemble_generator.generate_ensemble(
        sample_sequence, sample_target_id, ss_predictor, msa_data
    )

    # Visualize the ensemble
    refiner.visualize_ensemble(ensemble, title=f"Structure Ensemble for {sample_target_id}")

    # Format for submission
    submission_rows = ensemble_generator.format_for_submission(
        sample_target_id, sample_sequence, ensemble, rna_predictor.test_sequences
    )

    print(f"Generated {len(submission_rows)} submission rows for {sample_target_id}")