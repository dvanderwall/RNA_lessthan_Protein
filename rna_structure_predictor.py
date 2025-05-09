import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import glob
from sklearn.model_selection import train_test_split
from datetime import datetime
import torch
from tqdm import tqdm

# Base directories
DATA_DIR = "data"
MSA_DIR = os.path.join(DATA_DIR, "MSA")
OUTPUT_DIR = "output"
MODELS_DIR = "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


class RNAStructurePredictor:
    def __init__(self, temporal_cutoff=None, use_v2=True):
        """
        Initialize the RNA structure predictor with configuration options.

        Args:
            temporal_cutoff: Only use training data before this date (YYYY-MM-DD)
            use_v2: Whether to use the v2 training files
        """
        self.temporal_cutoff = temporal_cutoff
        self.use_v2 = use_v2

        # Data containers
        self.train_sequences = None
        self.train_labels = None
        self.validation_sequences = None
        self.validation_labels = None
        self.test_sequences = None
        self.msa_data = {}

    def load_data(self):
        """Load all necessary data files."""
        print("Loading sequence data...")

        # Load train sequences
        train_file = "train_sequences.v2.csv" if self.use_v2 else "train_sequences.csv"
        self.train_sequences = pd.read_csv(os.path.join(DATA_DIR, train_file))
        print("Train Sequences Data")
        print(self.train_sequences.head(10))

        # Load train labels
        train_labels_file = "train_labels.v2.csv" if self.use_v2 else "train_labels.csv"
        self.train_labels = pd.read_csv(os.path.join(DATA_DIR, train_labels_file))
        print("Train Labels Data")
        print(self.train_labels.head(10))
        # Load validation sequences
        self.validation_sequences = pd.read_csv(os.path.join(DATA_DIR, "validation_sequences.csv"))
        print("Validation Sequences Data")
        print(self.validation_sequences.head(10))

        # Load validation labels
        self.validation_labels = pd.read_csv(os.path.join(DATA_DIR, "validation_labels.csv"))
        print("Validation Labels Data")
        print(self.validation_labels.head(10))

        # Load test sequences
        self.test_sequences = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))
        print("Test Sequences Data")
        print(self.test_sequences.head(10))

        # Apply temporal cutoff if specified
        if self.temporal_cutoff:
            print("APPLYING TEMPORTAL CUTOFF")
            cutoff_date = datetime.strptime(self.temporal_cutoff, "%Y-%m-%d")

            # Debug: print counts before filtering
            print(f"Before temporal cutoff: {len(self.train_sequences)} sequences")

            self.train_sequences = self.train_sequences[
                pd.to_datetime(self.train_sequences['temporal_cutoff']) < cutoff_date
                ]

            # Debug: print counts after filtering
            print(f"After temporal cutoff: {len(self.train_sequences)} sequences")
            print(f"Sample target_ids: {self.train_sequences['target_id'].head(5).tolist()}")

            # Get valid target IDs
            valid_ids = self.train_sequences['target_id'].values

            # Debug: check ID formats
            if len(self.train_labels) > 0:
                print("Sample train_labels IDs:")
                print(self.train_labels['ID'].head(5).tolist())

                # Extract PDB IDs from train_labels for comparison
                # Modified to handle ID format 'PDB_CHAIN_RESID'
                label_ids = self.train_labels['ID'].str.split('_').str[:2].str.join('_').unique()
                print(f"Extracted {len(label_ids)} unique PDB_CHAIN combinations from labels")
                print(f"Sample extracted: {list(label_ids)[:5]}")

                # Check overlap between extracted IDs and valid_ids
                overlap = set(label_ids).intersection(set(valid_ids))
                print(f"Found {len(overlap)} matching IDs between sequences and labels")

                # Filter based on PDB_CHAIN instead of just PDB
                mask = self.train_labels['ID'].str.split('_').str[:2].str.join('_').isin(valid_ids)
                self.train_labels = self.train_labels[mask]

            print(f"After filtering: {len(self.train_labels)} label rows remain")

        print(f"Loaded {len(self.train_sequences)} training sequences")
        print(f"Loaded {len(self.validation_sequences)} validation sequences")
        print(f"Loaded {len(self.test_sequences)} test sequences")

    def load_msas(self, target_ids=None):
        """
        Load MSA files for specified target IDs or all training sequences.

        Args:
            target_ids: List of target IDs to load. If None, load all.
        """
        if target_ids is None:
            # Combine all target IDs from training, validation, and test
            target_ids = list(self.train_sequences['target_id'].unique())
            target_ids.extend(list(self.validation_sequences['target_id'].unique()))
            target_ids.extend(list(self.test_sequences['target_id'].unique()))
            target_ids = list(set(target_ids))  # Remove duplicates

        print(f"Loading MSA files for {len(target_ids)} targets...")

        for target_id in tqdm(target_ids):
            msa_file = os.path.join(MSA_DIR, f"{target_id}.MSA.fasta")
            if os.path.exists(msa_file):
                # Use Biopython to parse the MSA file
                msa_sequences = list(SeqIO.parse(msa_file, "fasta"))
                self.msa_data[target_id] = msa_sequences


        print(f"Loaded MSA data for {len(self.msa_data)} targets")

    def analyze_msas(self, target_id):
        """
        Analyze the MSA for a specific target ID to extract evolutionary information.

        Args:
            target_id: The target ID to analyze.

        Returns:
            Dictionary with MSA analysis results.
        """
        if target_id not in self.msa_data:
            print(f"MSA data for {target_id} not found. Loading...")
            self.load_msas([target_id])

            if target_id not in self.msa_data:
                print(f"Could not load MSA for {target_id}")
                return None

        msa_sequences = self.msa_data[target_id]
        query_seq = str(msa_sequences[0].seq)
        query_len = len(query_seq)

        # Initialize analysis results
        results = {
            'conservation': np.zeros(query_len),
            'gaps': np.zeros(query_len),
            'sequence_count': len(msa_sequences),
            'effective_sequence_count': 0
        }

        # Extract aligned sequences
        aligned_seqs = []
        for seq_record in msa_sequences:
            seq = str(seq_record.seq)
            if len(seq) == query_len:  # Ensure sequence is aligned correctly
                aligned_seqs.append(seq)

        # Calculate conservation and gap frequency
        for i in range(query_len):
            column = [seq[i] for seq in aligned_seqs]
            results['gaps'][i] = column.count('-') / len(aligned_seqs)

            # Simple conservation score: fraction of sequences matching query
            if query_seq[i] != '-':
                results['conservation'][i] = column.count(query_seq[i]) / len(aligned_seqs)

        # Estimate effective sequence count (simple approach)
        # More sophisticated approaches like sequence weighting could be implemented
        sequence_identity_matrix = np.zeros((len(aligned_seqs), len(aligned_seqs)))
        for i in range(len(aligned_seqs)):
            for j in range(i + 1, len(aligned_seqs)):
                identity = sum(1 for a, b in zip(aligned_seqs[i], aligned_seqs[j])
                               if a == b and a != '-' and b != '-') / query_len
                sequence_identity_matrix[i, j] = identity
                sequence_identity_matrix[j, i] = identity

        # Set diagonal to 1.0
        np.fill_diagonal(sequence_identity_matrix, 1.0)

        # Estimate effective sequence count as the sum of inverse average identity
        avg_identity = np.mean(sequence_identity_matrix, axis=1)
        results['effective_sequence_count'] = np.sum(1.0 / avg_identity)

        return results

    def preprocess_sequence_data(self, sequence_df):
        """
        Preprocess sequence data for model input.

        Args:
            sequence_df: DataFrame containing sequence data

        Returns:
            Preprocessed data ready for the model
        """
        # Convert sequences to one-hot encoding
        nucleotides = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

        processed_data = []
        for _, row in sequence_df.iterrows():
            target_id = row['target_id']
            sequence = row['sequence']

            # One-hot encode the sequence
            seq_length = len(sequence)
            one_hot = np.zeros((seq_length, 4))

            for i, nt in enumerate(sequence):
                if nt in nucleotides:
                    one_hot[i, nucleotides[nt]] = 1

            # Get MSA features if available
            msa_features = None
            if target_id in self.msa_data:
                msa_analysis = self.analyze_msas(target_id)
                if msa_analysis:
                    msa_features = {
                        'conservation': msa_analysis['conservation'],
                        'gaps': msa_analysis['gaps'],
                    }

            processed_data.append({
                'target_id': target_id,
                'sequence': sequence,
                'length': seq_length,
                'one_hot': one_hot,
                'msa_features': msa_features
            })

        return processed_data

    def preprocess_structure_data(self, labels_df):
        """
        Preprocess structure data (labels) for model training.

        Args:
            labels_df: DataFrame containing structure coordinates

        Returns:
            Dictionary mapping target_id_resid to 3D coordinates
        """
        structures = {}

        # Identify how many structures we have per residue
        columns = labels_df.columns
        coord_columns = [col for col in columns if col.startswith(('x_', 'y_', 'z_'))]
        num_structures = len([col for col in coord_columns if col.startswith('x_')])

        print(f"Found {num_structures} structures per residue")

        for _, row in labels_df.iterrows():
            # Parse the ID format which appears to be pdb_id_chain_id_resid
            id_parts = row['ID'].split('_')

            # Handle different ID formats
            if len(id_parts) == 3:  # Format: pdb_id_chain_id_resid
                target_id = id_parts[0]  # Just use the PDB ID as target_id
                resid = int(id_parts[2])  # Use the third part as resid
            elif len(id_parts) == 2:  # Format: target_id_resid
                target_id = id_parts[0]
                resid = int(id_parts[1])
            else:
                print(f"Warning: Unexpected ID format: {row['ID']}")
                continue

            # Create a list of coordinates for each structure
            coords_list = []
            for i in range(1, num_structures + 1):
                x_col = f'x_{i}'
                y_col = f'y_{i}'
                z_col = f'z_{i}'

                # Check if these columns exist and contain valid values
                if x_col in row and y_col in row and z_col in row and not (
                        pd.isna(row[x_col]) or pd.isna(row[y_col]) or pd.isna(row[z_col])
                ):
                    coords = np.array([row[x_col], row[y_col], row[z_col]])
                    coords_list.append(coords)

            if coords_list:
                structures[f"{target_id}_{resid}"] = coords_list

        print(f"Processed structure data: {len(structures)} residues found")
        return structures

    def prepare_datasets(self):
        """Prepare datasets for model training and evaluation."""
        print("Preparing datasets...")

        # Preprocess sequence data
        print('Processing Sequences')
        train_data = self.preprocess_sequence_data(self.train_sequences)
        val_data = self.preprocess_sequence_data(self.validation_sequences)
        test_data = self.preprocess_sequence_data(self.test_sequences)

        # Preprocess structure data
        print("Processing Structures")

        print("Here are Train Labels:")
        # Check for missing values in coordinate columns
        missing_coords = self.train_labels[['x_1', 'y_1', 'z_1']].isna().any(axis=1).sum()
        print(f"Rows with missing coordinates: {missing_coords} out of {len(self.train_labels)}")

        # Check ID format
        id_examples = self.train_labels['ID'].head(5).tolist()
        print(f"ID format examples: {id_examples}")

        print("Train Lavels Raw")
        print(self.train_labels)



        train_structures = self.preprocess_structure_data(self.train_labels)
        print("Here are the Training Structures:")
        print(train_structures)
        val_structures = self.preprocess_structure_data(self.validation_labels)

        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_structures': train_structures,
            'val_structures': val_structures
        }

    def visualize_sequence(self, target_id):
        """
        Visualize sequence and MSA data for a specific target.

        Args:
            target_id: Target ID to visualize
        """
        # Find sequence in data
        target_row = None
        for df in [self.train_sequences, self.validation_sequences, self.test_sequences]:
            if df is not None:
                matches = df[df['target_id'] == target_id]
                if not matches.empty:
                    target_row = matches.iloc[0]
                    break

        if target_row is None:
            print(f"Target {target_id} not found in any dataset")
            return

        sequence = target_row['sequence']
        print(f"Target: {target_id}")
        print(f"Sequence length: {len(sequence)}")
        print(f"Sequence: {sequence}")

        # Analyze MSA if available
        if target_id in self.msa_data:
            msa_analysis = self.analyze_msas(target_id)

            if msa_analysis:
                # Plot conservation and gap frequency
                fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

                # Plot conservation
                ax[0].plot(msa_analysis['conservation'], 'b-')
                ax[0].set_ylabel('Conservation')
                ax[0].set_title(f'MSA Analysis for {target_id}')

                # Plot gap frequency
                ax[1].plot(msa_analysis['gaps'], 'r-')
                ax[1].set_ylabel('Gap frequency')
                ax[1].set_xlabel('Sequence position')

                plt.tight_layout()
                plt.show()

                print(f"MSA contains {msa_analysis['sequence_count']} sequences")
                print(f"Effective sequence count: {msa_analysis['effective_sequence_count']:.2f}")
        else:
            print(f"No MSA data available for {target_id}")


# Example of how to use the class
if __name__ == "__main__":
    predictor = RNAStructurePredictor(temporal_cutoff="2022-05-27")
    predictor.load_data()
    predictor.load_msas()

    # Process a sample target
    sample_target_id = predictor.train_sequences['target_id'].iloc[0]
    #predictor.visualize_sequence(sample_target_id)

    # Prepare datasets
    datasets = predictor.prepare_datasets()
    print(f"Prepared {len(datasets['train_data'])} training samples")
    print(f"Prepared {len(datasets['val_data'])} validation samples")
    print(f"Prepared {len(datasets['test_data'])} test samples")