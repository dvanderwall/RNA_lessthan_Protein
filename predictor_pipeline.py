import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import datetime
import json
import time
import random
from concurrent.futures import ProcessPoolExecutor

# Import our modules
from rna_structure_predictor import RNAStructurePredictor
from secondary_structure_predictor import SecondaryStructurePredictor
from tertiary_structure_predictor import RNA3DPredictor, RNAStructure3DDataset
from structure_refinement import StructureRefiner, StructureEnsembleGenerator

# Set up directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, SUBMISSIONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


class RNAPredictionPipeline:
    """End-to-end pipeline for RNA structure prediction"""

    def __init__(self, config=None):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Dictionary with configuration parameters
        """
        # Set default configuration
        self.config = {
            'temporal_cutoff': "2022-05-27",  # Safe cutoff for CASP15
            'use_v2_data': True,  # Use v2 training files
            'use_msa': True,  # Use MSA information
            'batch_size': 8,  # Batch size for training
            'hidden_dim': 128,  # Hidden dimension for models
            'num_layers': 3,  # Number of layers in models
            'num_epochs': 50,  # Number of training epochs
            'learning_rate': 1e-3,  # Learning rate
            'weight_decay': 1e-5,  # Weight decay
            'dropout': 0.2,  # Dropout probability
            'patience': 10,  # Early stopping patience
            'num_candidates': 10,  # Candidate structures per sequence
            'num_final': 5,  # Structures in final ensemble
            'seed': 42,  # Random seed
            'device': 'cuda',  # Device to use ('cuda' or 'cpu')
            'num_workers': 4,  # Number of workers for data loading
            'max_sequence_length': 500,  # Maximum sequence length
            'subset_train': None,  # Subset of training data to use (for debugging)
            'subset_val': None,  # Subset of validation data to use
            'load_model': False,  # Whether to load existing model
            'model_path': None,  # Path to model checkpoint
            'parallel': True,  # Use parallel processing
            'num_processes': 4  # Number of parallel processes
        }

        # Update with provided configuration
        if config:
            self.config.update(config)

        print("Pipeline configuration:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")

        # Set random seeds
        self._set_seeds(self.config['seed'])

        # Set device
        if self.config['device'] == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.config['device'] = 'cpu'

        self.device = torch.device(self.config['device'])
        print(f"Using device: {self.device}")

        # Initialize components
        self.base_predictor = None
        self.ss_predictor = None
        self.predictor_3d = None
        self.refiner = None
        self.ensemble_generator = None

        # Data containers
        self.datasets = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def _set_seeds(self, seed):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def initialize_components(self):
        """Initialize all pipeline components"""
        print("Initializing pipeline components...")

        # Initialize base predictor
        self.base_predictor = RNAStructurePredictor(
            temporal_cutoff=self.config['temporal_cutoff'],
            use_v2=self.config['use_v2_data']
        )

        # Initialize secondary structure predictor
        self.ss_predictor = SecondaryStructurePredictor(
            use_msa=self.config['use_msa'],
            use_ml=True
        )

        # Initialize 3D structure predictor
        self.predictor_3d = RNA3DPredictor(
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            device=self.device
        )

        # Initialize structure refiner
        self.refiner = StructureRefiner(
            energy_weight=1.0,
            secondary_structure_weight=1.0,
            clash_weight=1.0,
            smooth_weight=0.5,
            diversity_weight=0.2
        )

        # Initialize ensemble generator
        self.ensemble_generator = StructureEnsembleGenerator(
            self.refiner,
            self.predictor_3d,
            num_candidates=self.config['num_candidates'],
            num_final=self.config['num_final']
        )

        print("Components initialized")

    def load_data(self):
        """Load and preprocess all data"""
        print("Loading data...")

        # Load base data
        self.base_predictor.load_data()

        # Load MSAs
        test_targets = self.base_predictor.test_sequences['target_id'].values
        val_targets = self.base_predictor.validation_sequences['target_id'].values

        # Load MSAs for test and validation targets
        target_ids = list(test_targets) + list(val_targets)

        # If we're using a subset of training data, add those targets
        if self.config['subset_train'] is not None:
            train_subset = self.base_predictor.train_sequences.iloc[:self.config['subset_train']]
            train_targets = train_subset['target_id'].values
            target_ids.extend(train_targets)

        # Remove duplicates
        target_ids = list(set(target_ids))

        print(f"Loading MSAs for {len(target_ids)} targets...")
        self.base_predictor.load_msas(target_ids)

        # Prepare datasets
        print("Preparing datasets...")
        self.datasets = self.base_predictor.prepare_datasets()

        # Limit to subsets if specified
        if self.config['subset_train'] is not None:
            self.datasets['train_data'] = self.datasets['train_data'][:self.config['subset_train']]

        if self.config['subset_val'] is not None:
            self.datasets['val_data'] = self.datasets['val_data'][:self.config['subset_val']]

        print(f"Prepared {len(self.datasets['train_data'])} training samples")
        print(f"Prepared {len(self.datasets['val_data'])} validation samples")
        print(f"Prepared {len(self.datasets['test_data'])} test samples")

        # Create datasets
        train_dataset = RNAStructure3DDataset(
            self.datasets['train_data'],
            self.datasets['train_structures'],
            max_length=self.config['max_sequence_length']
        )

        val_dataset = RNAStructure3DDataset(
            self.datasets['val_data'],
            self.datasets['val_structures'],
            max_length=self.config['max_sequence_length']
        )

        test_dataset = RNAStructure3DDataset(
            self.datasets['test_data'],
            max_length=self.config['max_sequence_length']
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            collate_fn=RNAStructure3DDataset.collate_fn
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=RNAStructure3DDataset.collate_fn
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one test sequence at a time
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=RNAStructure3DDataset.collate_fn
        )

        print("Data loading complete")

    def train_predictors(self):
        """Train the predictors"""
        print("Training predictors...")

        # Train secondary structure ML model if enabled
        if self.ss_predictor.use_ml:
            #print("Training secondary structure ML model...")
            #print("Input Train Data:")
            #print(self.datasets['train_data'])
            print("Input Training Structures:")
            print(self.datasets['train_structures'])
            self.ss_predictor.train_ml_predictor(
                self.datasets['train_data'],
                self.datasets['train_structures']
            )

        # Train 3D structure predictor
        if not self.config['load_model'] or not self.config['model_path']:
            print("Training 3D structure predictor...")
            start_time = time.time()

            train_losses, val_losses = self.predictor_3d.train(
                self.train_loader,
                self.val_loader,
                num_epochs=self.config['num_epochs'],
                patience=self.config['patience']
            )

            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")

            # Save the model
            model_path = os.path.join(MODELS_DIR, "model_3d.pt")
            self.predictor_3d.save_model(model_path)
            print(f"Model saved to {model_path}")
        else:
            # Load existing model
            print(f"Loading 3D structure model from {self.config['model_path']}...")
            self.predictor_3d.load_model(self.config['model_path'])

    def predict_structures(self):
        """Predict structures for test sequences"""
        print("Predicting structures for test sequences...")

        # Get test sequences
        test_sequences = self.base_predictor.test_sequences
        test_targets = test_sequences['target_id'].values

        # Results container
        all_predictions = {}
        submission_rows = []

        # Iterate over test sequences
        for target_id in tqdm(test_targets, desc="Predicting structures"):
            # Get sequence data
            seq_row = test_sequences[test_sequences['target_id'] == target_id].iloc[0]
            sequence = seq_row['sequence']
            msa_data = self.base_predictor.msa_data.get(target_id)

            # Process the sequence
            if self.config['parallel']:
                # Process in parallel (in a real implementation)
                # For now, just call the function directly
                result = self._process_single_sequence(target_id, sequence, msa_data)
            else:
                # Process sequentially
                result = self._process_single_sequence(target_id, sequence, msa_data)

            # Store results
            all_predictions[target_id] = result['ensemble']
            submission_rows.extend(result['submission_rows'])

        # Create submission file
        submission_df = pd.DataFrame(submission_rows)
        submission_path = os.path.join(
            SUBMISSIONS_DIR,
            f"submission_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        submission_df.to_csv(submission_path, index=False)

        print(f"Predictions complete. Submission saved to {submission_path}")
        return all_predictions, submission_df

    def _process_single_sequence(self, target_id, sequence, msa_data):
        """
        Process a single sequence to predict its structure.

        Args:
            target_id: Target identifier
            sequence: RNA sequence string
            msa_data: MSA data for the sequence

        Returns:
            Dictionary with prediction results
        """
        print(f"Processing {target_id}...")

        # Step 1: Predict secondary structure
        ss_results = self.ss_predictor.predict(sequence, msa_data)

        # Extract base pairs from secondary structure
        mfe_structure = ss_results['mfe_structure']

        # Parse parentheses notation to get base pairs
        base_pairs = []
        stack = []
        for i, char in enumerate(mfe_structure):
            if char == '(':
                stack.append(i)
            elif char == ')':
                if stack:
                    base_pairs.append((stack.pop(), i))

        # Step 2: Generate initial 3D structure predictions
        # In a full implementation, this would use the trained model
        # For now, we'll use a placeholder implementation

        # Generate ensemble
        ensemble = self.ensemble_generator.generate_ensemble(
            sequence, target_id, self.ss_predictor, msa_data
        )

        # Format for submission
        submission_rows = self.ensemble_generator.format_for_submission(
            target_id, sequence, ensemble, self.base_predictor.test_sequences
        )

        return {
            'target_id': target_id,
            'ensemble': ensemble,
            'submission_rows': submission_rows
        }

    def run_pipeline(self):
        """Run the complete pipeline"""
        # Initialize components
        self.initialize_components()

        # Load data
        self.load_data()

        # Train predictors
        self.train_predictors()

        # Predict structures
        predictions, submission_df = self.predict_structures()

        return {
            'predictions': predictions,
            'submission_df': submission_df
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='RNA Structure Prediction Pipeline')

    # Add arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--temporal_cutoff', type=str, help='Temporal cutoff date (YYYY-MM-DD)')
    parser.add_argument('--use_v2_data', action='store_true', help='Use v2 training files')
    parser.add_argument('--use_msa', action='store_true', help='Use MSA information')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--device', type=str, help='Device to use (cuda or cpu)')
    parser.add_argument('--load_model', action='store_true', help='Load existing model')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--subset_train', type=int, help='Use a subset of training data')
    parser.add_argument('--subset_val', type=int, help='Use a subset of validation data')
    parser.add_argument('--seed', type=int, help='Random seed')

    # Parse arguments
    args = parser.parse_args()

    # Load configuration from file if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Update with command line arguments
    for arg in vars(args):
        if getattr(args, arg) is not None and arg != 'config':
            config[arg] = getattr(args, arg)

    # Create and run pipeline
    pipeline = RNAPredictionPipeline(config)
    results = pipeline.run_pipeline()

    print("Pipeline execution complete")
    return results


if __name__ == "__main__":
    main()