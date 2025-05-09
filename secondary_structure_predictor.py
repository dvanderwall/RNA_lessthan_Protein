import numpy as np
import os
import subprocess
import tempfile
import re
from Bio import SeqIO, AlignIO
import matplotlib.pyplot as plt
import RNA  # ViennaRNA package Python bindings
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ThreadPoolExecutor


class SecondaryStructurePredictor:
    """Class for predicting RNA secondary structure"""

    def __init__(self, use_msa=True, use_ml=True):
        """
        Initialize the secondary structure predictor.

        Args:
            use_msa: Whether to incorporate MSA information
            use_ml: Whether to use machine learning to enhance predictions
        """
        self.use_msa = use_msa
        self.use_ml = use_ml
        self.ml_model = None

    def predict_with_viennarna(self, sequence):
        """
        Predict secondary structure using ViennaRNA's RNAfold.

        Args:
            sequence: RNA sequence string

        Returns:
            Dictionary with structure prediction results
        """
        # Use ViennaRNA Python bindings for RNAfold
        fc = RNA.fold_compound(sequence)
        (ss, mfe) = fc.mfe()

        # Calculate base pairing probabilities
        fc.pf()
        bp_probabilities = []

        # Extract pair probabilities for each position
        # Get the entire base pair probability matrix
        bpp_matrix = fc.bpp()

        # Extract individual probabilities
        for i in range(1, len(sequence) + 1):
            for j in range(i + 1, len(sequence) + 1):
                # Adjust indices based on the matrix implementation
                # In some versions, bpp_matrix might be a flat array or a 2D matrix
                if isinstance(bpp_matrix, list) or isinstance(bpp_matrix, np.ndarray):
                    # Try to access as a 2D matrix if possible
                    try:
                        prob = bpp_matrix[i][j]
                    except (IndexError, TypeError):
                        # If not a 2D matrix, it might be a flat representation or have a different access method
                        # Skip this pair for now
                        continue
                else:
                    # For newer ViennaRNA versions that return a different object
                    try:
                        # Try to get probability through a method or attribute
                        prob = bpp_matrix.get(i, j) if hasattr(bpp_matrix, 'get') else 0
                    except (AttributeError, TypeError):
                        # If not accessible this way, skip
                        continue

                if prob > 0.01:  # Filter for significant probabilities
                    bp_probabilities.append((i - 1, j - 1, prob))  # Convert to 0-indexed

        # Predict MEA (Maximum Expected Accuracy) structure
        mea_structure = fc.MEA()

        # Get ensemble diversity
        ensemble_diversity = fc.mean_bp_distance()

        # Collect results
        results = {
            'sequence': sequence,
            'mfe_structure': ss,
            'mfe': mfe,
            'mea_structure': mea_structure[0],
            'mea_energy': mea_structure[1],
            'ensemble_diversity': ensemble_diversity,
            'bp_probabilities': bp_probabilities,
            'bp_matrix': self._create_bp_matrix(bp_probabilities, len(sequence))
        }

        return results

    def _create_bp_matrix(self, bp_probabilities, seq_len):
        """Create a base pairing probability matrix from list of probabilities"""
        matrix = np.zeros((seq_len, seq_len))
        for i, j, prob in bp_probabilities:
            matrix[i, j] = prob
            matrix[j, i] = prob  # Make it symmetric
        return matrix

    def predict_with_msa(self, sequence, msa_data):
        """
        Predict secondary structure using covariation information from MSA.

        Args:
            sequence: RNA sequence string
            msa_data: List of sequence records from MSA

        Returns:
            Dictionary with covariation-based prediction results
        """
        if not msa_data or len(msa_data) < 5:
            print("Not enough sequences in MSA for covariation analysis")
            return None

        seq_len = len(sequence)

        # Create covariation matrix
        covariation_matrix = np.zeros((seq_len, seq_len))

        # Extract aligned sequences
        aligned_seqs = []
        for seq_record in msa_data:
            seq = str(seq_record.seq)
            if len(seq) == seq_len:  # Ensure sequence is aligned correctly
                aligned_seqs.append(seq)

        if len(aligned_seqs) < 5:
            print("Not enough properly aligned sequences in MSA")
            return None

        # Create nucleotide frequency matrices for each position
        position_frequencies = []
        for i in range(seq_len):
            freq = {'A': 0, 'C': 0, 'G': 0, 'U': 0, '-': 0}
            for seq in aligned_seqs:
                nt = seq[i]
                if nt in freq:
                    freq[nt] += 1
                else:
                    freq['-'] += 1  # Count any unknown character as gap

            # Normalize frequencies
            total = sum(freq.values())
            if total > 0:
                for nt in freq:
                    freq[nt] /= total

            position_frequencies.append(freq)

        # Calculate mutual information between pairs of positions
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if i == j:
                    continue

                # Skip if either position is too conserved or has too many gaps
                if (position_frequencies[i]['-'] > 0.5 or
                        position_frequencies[j]['-'] > 0.5):
                    continue

                # Calculate joint frequencies
                joint_freq = {}
                for seq in aligned_seqs:
                    pair = (seq[i], seq[j])
                    if pair not in joint_freq:
                        joint_freq[pair] = 0
                    joint_freq[pair] += 1

                # Normalize joint frequencies
                total_pairs = len(aligned_seqs)
                for pair in joint_freq:
                    joint_freq[pair] /= total_pairs

                # Calculate mutual information
                mi = 0
                for nt_i in 'ACGU':
                    for nt_j in 'ACGU':
                        pair = (nt_i, nt_j)
                        if pair in joint_freq and joint_freq[pair] > 0:
                            p_ij = joint_freq[pair]
                            p_i = position_frequencies[i][nt_i]
                            p_j = position_frequencies[j][nt_j]

                            if p_i > 0 and p_j > 0:
                                mi += p_ij * np.log2(p_ij / (p_i * p_j))

                # Store mutual information as covariation score
                covariation_matrix[i, j] = mi
                covariation_matrix[j, i] = mi  # Make it symmetric

        # Identify potential base pairs based on covariation
        covariation_pairs = []
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if covariation_matrix[i, j] > 0.5:  # Threshold can be tuned
                    covariation_pairs.append((i, j))

        # Calculate covariation-based structure (simple dot-bracket notation)
        cov_structure = ['.' for _ in range(seq_len)]
        for i, j in covariation_pairs:
            cov_structure[i] = '('
            cov_structure[j] = ')'

        cov_structure = ''.join(cov_structure)

        return {
            'covariation_matrix': covariation_matrix,
            'covariation_pairs': covariation_pairs,
            'covariation_structure': cov_structure
        }

    def train_ml_predictor(self, sequences, structures):
        """
        Train a machine learning model to enhance structure predictions.

        Args:
            sequences: List of processed sequence data
            structures: Dictionary of known structures

        Returns:
            Trained ML model
        """
        if not self.use_ml:
            return None

        print("Training ML predictor for secondary structure...")

        # Prepare training data
        X = []
        y = []

        for seq_data in sequences:
            target_id = seq_data['target_id']
            sequence = seq_data['sequence']

            # Get ViennaRNA predictions
            vienna_results = self.predict_with_viennarna(sequence)

            print("Brief Vienna Results")
            for row in vienna_results['bp_matrix'][:10]:  # first 10 rows
                print(row[:10])  # first 10 columns
            # Extract features for each position
            for i in range(len(sequence)):
                # Find this residue in the known structures
                res_key = f"{target_id}_{i + 1}"  # 1-based indexing for residue IDs

                # Skip if no structure data for this residue
                if res_key not in structures or not structures[res_key]:
                    continue

                # Features: nucleotide identity, base pair probability, etc.
                features = [
                    ord(sequence[i]) - ord('A'),  # Nucleotide as numeric value
                    np.max(vienna_results['bp_matrix'][i]),  # Max base pair probability
                    np.sum(vienna_results['bp_matrix'][i]),  # Sum of pair probabilities
                    1 if vienna_results['mfe_structure'][i] in '()' else 0,  # In a pair in MFE?
                    1 if vienna_results['mea_structure'][i] in '()' else 0,  # In a pair in MEA?
                ]

                # Get MSA features if available
                if self.use_msa and 'msa_features' in seq_data and seq_data['msa_features']:
                    features.extend([
                        seq_data['msa_features']['conservation'][i],
                        seq_data['msa_features']['gaps'][i]
                    ])

                X.append(features)

                # Label: is this position paired in the known structure?
                # For simplicity, we'll just use a binary paired/unpaired label
                is_paired = 0
                if vienna_results['mfe_structure'][i] in '()':
                    is_paired = 1

                y.append(is_paired)

        if not X or not y:
            print("No training data available for ML predictor")
            return None

        # Train a simple RandomForest classifier
        print(f"Training on {len(X)} examples")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        self.ml_model = model
        return model

    def enhance_prediction_with_ml(self, sequence, vienna_results, msa_features=None):
        """
        Enhance structure prediction using the ML model.

        Args:
            sequence: RNA sequence string
            vienna_results: Results from ViennaRNA prediction
            msa_features: Features extracted from MSA

        Returns:
            Enhanced structure prediction
        """
        if not self.ml_model:
            return vienna_results

        # Prepare features for each position
        X_pred = []
        for i in range(len(sequence)):
            features = [
                ord(sequence[i]) - ord('A'),  # Nucleotide as numeric value
                np.max(vienna_results['bp_matrix'][i]),  # Max base pair probability
                np.sum(vienna_results['bp_matrix'][i]),  # Sum of pair probabilities
                1 if vienna_results['mfe_structure'][i] in '()' else 0,  # In a pair in MFE?
                1 if vienna_results['mea_structure'][i] in '()' else 0,  # In a pair in MEA?
            ]

            # Add MSA features if available
            if self.use_msa and msa_features:
                features.extend([
                    msa_features['conservation'][i],
                    msa_features['gaps'][i]
                ])

            X_pred.append(features)

        # Make predictions
        y_pred = self.ml_model.predict(X_pred)

        # Adjust structure based on ML predictions
        # This is a simple implementation; a more sophisticated approach would ensure
        # the structure remains valid (matched parentheses, etc.)
        ml_structure = list(vienna_results['mfe_structure'])

        # For now, we'll just mark positions as paired/unpaired based on ML
        # In a real implementation, we would need to ensure paired positions match
        for i, paired in enumerate(y_pred):
            if paired:
                if ml_structure[i] == '.':
                    ml_structure[i] = '?'  # Mark as potentially paired
            else:
                if ml_structure[i] in '()':
                    ml_structure[i] = '.'  # Mark as unpaired

        vienna_results['ml_enhanced_structure'] = ''.join(ml_structure)
        return vienna_results

    def predict(self, sequence, msa_data=None):
        """
        Predict secondary structure using all available methods.

        Args:
            sequence: RNA sequence string
            msa_data: MSA data for this sequence

        Returns:
            Dictionary with structure prediction results
        """
        # Get initial predictions from ViennaRNA
        vienna_results = self.predict_with_viennarna(sequence)

        # Extract MSA features if available
        msa_features = None
        if self.use_msa and msa_data:
            covariation_results = self.predict_with_msa(sequence, msa_data)
            if covariation_results:
                vienna_results.update(covariation_results)

                # Extract conservation and gap features from MSA
                msa_features = {
                    'conservation': np.zeros(len(sequence)),
                    'gaps': np.zeros(len(sequence))
                }

                for i in range(len(sequence)):
                    column = [str(seq_record.seq)[i] for seq_record in msa_data
                              if len(str(seq_record.seq)) == len(sequence)]
                    if column:
                        msa_features['gaps'][i] = column.count('-') / len(column)
                        msa_features['conservation'][i] = column.count(sequence[i]) / len(column)

        # Enhance prediction with ML model if available
        if self.use_ml and self.ml_model:
            vienna_results = self.enhance_prediction_with_ml(
                sequence, vienna_results, msa_features
            )

        return vienna_results

    def visualize_structure(self, results, title=None):
        """
        Visualize structure prediction results.

        Args:
            results: Structure prediction results
            title: Plot title
        """
        sequence = results['sequence']
        seq_len = len(sequence)

        # Create a figure with base pair probability matrix
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Plot base pair probability matrix
        im = axes[0].imshow(results['bp_matrix'], cmap='viridis', origin='lower')
        axes[0].set_title('Base Pair Probability Matrix')
        axes[0].set_xlabel('Sequence Position')
        axes[0].set_ylabel('Sequence Position')
        plt.colorbar(im, ax=axes[0])

        # Plot MFE and MEA structures as arc diagrams
        axes[1].set_xlim(0, seq_len)
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Secondary Structure')
        axes[1].set_xlabel('Sequence Position')
        axes[1].set_yticks([])

        # Mark MFE structure with arcs
        for i in range(seq_len):
            if results['mfe_structure'][i] == '(':
                # Find matching closing parenthesis
                depth = 1
                for j in range(i + 1, seq_len):
                    if results['mfe_structure'][j] == '(':
                        depth += 1
                    elif results['mfe_structure'][j] == ')':
                        depth -= 1
                        if depth == 0:
                            # Draw arc from i to j
                            radius = (j - i) / 2
                            center = (i + j) / 2
                            theta = np.linspace(0, np.pi, 100)
                            x = center + radius * np.cos(theta)
                            y = 0.5 + 0.4 * np.sin(theta)
                            axes[1].plot(x, y, 'b-', alpha=0.5, lw=1)
                            break

        # Add sequence along the x-axis
        for i, nt in enumerate(sequence):
            axes[1].text(i, 0.1, nt, ha='center', va='center', fontsize=8)

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        plt.show()


# Example of how to use this class with the RNAStructurePredictor
if __name__ == "__main__":
    from rna_structure_predictor import RNAStructurePredictor

    # Initialize the predictor
    rna_predictor = RNAStructurePredictor(temporal_cutoff="2022-05-27")
    rna_predictor.load_data()

    # Load MSAs for a subset of targets for demonstration
    sample_targets = rna_predictor.train_sequences['target_id'].iloc[:10].values
    rna_predictor.load_msas(sample_targets)

    # Initialize secondary structure predictor
    ss_predictor = SecondaryStructurePredictor(use_msa=True, use_ml=True)

    # Train the ML component (if needed)
    datasets = rna_predictor.prepare_datasets()
    ss_predictor.train_ml_predictor(
        datasets['train_data'][:100],  # Use a subset for quick demonstration
        datasets['train_structures']
    )

    # Predict and visualize a sample sequence
    sample_target_id = sample_targets[0]
    sample_seq_data = next(data for data in datasets['train_data']
                           if data['target_id'] == sample_target_id)

    msa_data = rna_predictor.msa_data.get(sample_target_id)

    # Make prediction
    ss_results = ss_predictor.predict(
        sample_seq_data['sequence'],
        msa_data
    )

    # Visualize results
    ss_predictor.visualize_structure(
        ss_results,
        title=f"Secondary Structure for {sample_target_id}"
    )