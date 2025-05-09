import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import subprocess
import tempfile
import os
from tqdm import tqdm
import random
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform


class RNAStructure3DDataset(Dataset):
    """Dataset for RNA 3D structure prediction"""

    def __init__(self, sequences, structures=None, max_length=500):
        """
        Initialize the dataset.

        Args:
            sequences: List of processed sequence data
            structures: Dictionary of known structures (for training)
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.structures = structures
        self.max_length = max_length

        # Filter sequences that are too long
        self.valid_indices = []
        for i, seq_data in enumerate(sequences):
            if seq_data['length'] <= max_length:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        seq_data = self.sequences[self.valid_indices[idx]]
        target_id = seq_data['target_id']
        sequence = seq_data['sequence']
        one_hot = seq_data['one_hot']

        # Get 3D coordinates if available
        coords = None
        if self.structures:
            # Collect coordinates for this sequence
            seq_coords = []
            for i in range(len(sequence)):
                res_key = f"{target_id}_{i + 1}"  # 1-based indexing for residue IDs
                if res_key in self.structures and self.structures[res_key]:
                    # Use the first available structure
                    seq_coords.append(self.structures[res_key][0])
                else:
                    # Use NaN for missing coordinates
                    seq_coords.append(np.array([np.nan, np.nan, np.nan]))

            coords = np.stack(seq_coords) if seq_coords else None

        sample = {
            'target_id': target_id,
            'sequence': sequence,
            'one_hot': one_hot,
            'length': len(sequence)
        }

        if coords is not None:
            sample['coordinates'] = coords

        # Add MSA features if available
        if 'msa_features' in seq_data and seq_data['msa_features']:
            sample['msa_features'] = seq_data['msa_features']

        return sample

    @staticmethod
    def collate_fn(batch):
        """Custom collate function to handle variable-length sequences"""
        # Find max length in this batch
        max_len = max(sample['length'] for sample in batch)

        # Initialize batch tensors
        batch_size = len(batch)
        one_hot_batch = torch.zeros(batch_size, max_len, 4)
        mask = torch.zeros(batch_size, max_len)

        # Collect other sample information
        target_ids = []
        sequences = []
        lengths = []
        coords_batch = None
        msa_features_batch = None

        # If coordinates are available, create a tensor for them
        has_coords = 'coordinates' in batch[0]
        if has_coords:
            coords_batch = torch.full((batch_size, max_len, 3), float('nan'))

        # If MSA features are available, create tensors for them
        has_msa = 'msa_features' in batch[0]
        if has_msa:
            conservation_batch = torch.zeros(batch_size, max_len)
            gaps_batch = torch.zeros(batch_size, max_len)

        # Fill in the batch tensors
        for i, sample in enumerate(batch):
            seq_len = sample['length']
            one_hot_batch[i, :seq_len] = torch.tensor(sample['one_hot'])
            mask[i, :seq_len] = 1

            target_ids.append(sample['target_id'])
            sequences.append(sample['sequence'])
            lengths.append(seq_len)

            if has_coords and 'coordinates' in sample:
                coords = sample['coordinates']
                coords_batch[i, :seq_len] = torch.tensor(coords)

            if has_msa and 'msa_features' in sample:
                conservation = sample['msa_features']['conservation']
                gaps = sample['msa_features']['gaps']
                conservation_batch[i, :seq_len] = torch.tensor(conservation)
                gaps_batch[i, :seq_len] = torch.tensor(gaps)

        # Prepare the batch dictionary
        batch_dict = {
            'target_ids': target_ids,
            'sequences': sequences,
            'one_hot': one_hot_batch,
            'mask': mask,
            'lengths': torch.tensor(lengths)
        }

        if has_coords:
            batch_dict['coordinates'] = coords_batch

        if has_msa:
            batch_dict['msa_features'] = {
                'conservation': conservation_batch,
                'gaps': gaps_batch
            }

        return batch_dict


class RNAEncoder(nn.Module):
    """Encoder for RNA sequences"""

    def __init__(self, input_dim=4, hidden_dim=128, num_layers=3, bidirectional=True, dropout=0.2, use_msa=True):
        super(RNAEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_msa = use_msa  # Can be turned on/off

        # Create separate embedding layers for with/without MSA
        self.base_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if self.use_msa:
            self.msa_embedding = nn.Sequential(
                nn.Linear(input_dim + 2, hidden_dim),  # +2 for conservation and gaps
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # Bidirectional LSTM for sequence processing
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output dimension of LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Transformer encoder layer for capturing global dependencies
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=lstm_output_dim,
            nhead=8,
            dim_feedforward=lstm_output_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim)
        )

    def forward(self, x, msa_features=None, mask=None, lengths=None):
        batch_size, seq_len, _ = x.size()

        # Choose appropriate embedding based on MSA feature availability
        if self.use_msa and msa_features is not None:
            # Check if we have the expected MSA features
            if 'conservation' in msa_features and 'gaps' in msa_features:
                conservation = msa_features['conservation'].unsqueeze(-1)
                gaps = msa_features['gaps'].unsqueeze(-1)
                x_with_msa = torch.cat([x, conservation, gaps], dim=-1)
                x = self.msa_embedding(x_with_msa)
            else:
                # If expected MSA features are missing, use base embedding
                x = self.base_embedding(x)
        else:
            # If MSA usage is disabled or features not provided, use base embedding
            x = self.base_embedding(x)

        # Pack padded sequence for LSTM if lengths are provided
        if lengths is not None:
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

            # LSTM processing
            lstm_out, _ = self.lstm(x_packed)

            # Unpack the sequence
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            # If no lengths provided, run LSTM normally
            lstm_out, _ = self.lstm(x)

        # Create attention mask for transformer
        if mask is not None:
            # Create a mask where 1 means keep, 0 means mask
            attn_mask = mask.eq(0)
            lstm_out = self.transformer_encoder(lstm_out, src_key_padding_mask=attn_mask)
        else:
            lstm_out = self.transformer_encoder(lstm_out)

        # Final output
        output = self.output_layer(lstm_out)

        return output


class RNA3DStructureModel(nn.Module):
    """Model for predicting RNA 3D structure"""

    def __init__(self, input_dim=4, hidden_dim=128, num_layers=3, dropout=0.2):
        super(RNA3DStructureModel, self).__init__()

        self.encoder = RNAEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # Output dimension of encoder
        encoder_output_dim = hidden_dim * 2  # bidirectional

        # Coordinate prediction head
        self.coord_predictor = nn.Sequential(
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_dim, 3)  # x, y, z coordinates
        )

        # Distance prediction head (for pairwise distances)
        self.distance_predictor = nn.Sequential(
            nn.Linear(encoder_output_dim * 2, encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_output_dim, 1)
        )

    def forward(self, one_hot, msa_features=None, mask=None, lengths=None):
        # Get sequence encodings
        encodings = self.encoder(one_hot, msa_features, mask, lengths)

        # Predict coordinates directly
        coords = self.coord_predictor(encodings)

        # Apply mask if provided
        if mask is not None:
            coords = coords * mask.unsqueeze(-1)

        # For distance prediction, we need to compute pairwise features
        batch_size, seq_len, hidden_dim = encodings.size()

        # Return predicted coordinates
        return {
            'coordinates': coords,
            'encodings': encodings
        }

    def predict_distances(self, encodings, mask=None):
        """Predict pairwise distances between residues"""
        batch_size, seq_len, hidden_dim = encodings.size()

        # Create pairwise features
        # This is computationally expensive for long sequences,
        # so we'll use a more efficient approach
        distances = []

        for b in range(batch_size):
            enc = encodings[b]

            # Create all pairs of encodings
            enc_i = enc.unsqueeze(1).expand(-1, seq_len, -1)  # (seq_len, seq_len, hidden_dim)
            enc_j = enc.unsqueeze(0).expand(seq_len, -1, -1)  # (seq_len, seq_len, hidden_dim)

            # Concatenate encodings
            pairs = torch.cat([enc_i, enc_j], dim=-1)  # (seq_len, seq_len, hidden_dim*2)

            # Predict distances
            dist = self.distance_predictor(pairs).squeeze(-1)  # (seq_len, seq_len)

            # Apply mask if provided
            if mask is not None:
                m = mask[b]
                mask_2d = m.unsqueeze(0) * m.unsqueeze(1)
                dist = dist * mask_2d

            distances.append(dist)

        return torch.stack(distances)


class RNA3DPredictor:
    """Class for predicting RNA 3D structure"""

    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.2, learning_rate=1e-3,
                 weight_decay=1e-5, device=None):
        """
        Initialize the 3D structure predictor.

        Args:
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            device: Torch device to use
        """
        # Set device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create model
        self.model = RNA3DStructureModel(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=5, factor=0.5
        )

        self.best_model_state = None
        self.best_val_loss = float('inf')

    def train(self, train_dataloader, val_dataloader, num_epochs=50, patience=10):
        """
        Train the model.

        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            patience: Early stopping patience
        """
        print("Starting training...")

        train_losses = []
        val_losses = []

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_samples = 0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Training)"):
                # Move data to device
                one_hot = batch['one_hot'].to(self.device)
                mask = batch['mask'].to(self.device)
                lengths = batch['lengths']

                # Get coordinates if available
                coords_target = None
                if 'coordinates' in batch:
                    coords_target = batch['coordinates'].to(self.device)

                # Get MSA features if available
                msa_features = None
                if 'msa_features' in batch:
                    msa_features = {
                        'conservation': batch['msa_features']['conservation'].to(self.device),
                        'gaps': batch['msa_features']['gaps'].to(self.device)
                    }

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(one_hot, msa_features, mask, lengths)
                coords_pred = outputs['coordinates']

                # Calculate loss only on available coordinates
                if coords_target is not None:
                    # Create a mask for non-NaN coordinates
                    valid_mask = ~torch.isnan(coords_target).any(dim=-1)

                    # Apply both masks
                    combined_mask = (mask * valid_mask).bool()

                    # Calculate MSE loss on valid coordinates
                    if combined_mask.sum() > 0:
                        mse_loss = F.mse_loss(
                            coords_pred[combined_mask],
                            coords_target[combined_mask]
                        )

                        # Backward pass and optimization
                        mse_loss.backward()
                        self.optimizer.step()

                        # Update statistics
                        batch_loss = mse_loss.item()
                        batch_size = combined_mask.sum().item()
                        train_loss += batch_loss * batch_size
                        train_samples += batch_size

            # Calculate average training loss
            avg_train_loss = train_loss / train_samples if train_samples > 0 else float('inf')
            train_losses.append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            val_samples = 0

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)"):
                    # Move data to device
                    one_hot = batch['one_hot'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    lengths = batch['lengths']

                    # Get coordinates if available
                    coords_target = None
                    if 'coordinates' in batch:
                        coords_target = batch['coordinates'].to(self.device)

                    # Get MSA features if available
                    msa_features = None
                    if 'msa_features' in batch:
                        msa_features = {
                            'conservation': batch['msa_features']['conservation'].to(self.device),
                            'gaps': batch['msa_features']['gaps'].to(self.device)
                        }

                    # Forward pass
                    outputs = self.model(one_hot, msa_features, mask, lengths)
                    coords_pred = outputs['coordinates']

                    # Calculate loss only on available coordinates
                    if coords_target is not None:
                        # Create a mask for non-NaN coordinates
                        valid_mask = ~torch.isnan(coords_target).any(dim=-1)

                        # Apply both masks
                        combined_mask = (mask * valid_mask).bool()

                        # Calculate MSE loss on valid coordinates
                        if combined_mask.sum() > 0:
                            mse_loss = F.mse_loss(
                                coords_pred[combined_mask],
                                coords_target[combined_mask]
                            )

                            # Update statistics
                            batch_loss = mse_loss.item()
                            batch_size = combined_mask.sum().item()
                            val_loss += batch_loss * batch_size
                            val_samples += batch_size

            # Calculate average validation loss
            avg_val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
            val_losses.append(avg_val_loss)

            # Update learning rate
            self.scheduler.step(avg_val_loss)

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}")

            # Check for improvement
            if avg_val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}")
                best_val_loss = avg_val_loss
                self.best_val_loss = best_val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored model with best validation loss: {self.best_val_loss:.4f}")

        # Plot training progress
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        return train_losses, val_losses

    def predict(self, dataloader, num_samples=5, temperature=1.0):
        """
        Generate predictions for sequences in the dataloader.

        Args:
            dataloader: Data loader for sequences
            num_samples: Number of structure samples to generate
            temperature: Temperature for sampling (higher = more diverse)

        Returns:
            Dictionary with predictions
        """
        print("Generating predictions...")
        self.model.eval()

        predictions = {}

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Move data to device
                one_hot = batch['one_hot'].to(self.device)
                mask = batch['mask'].to(self.device)
                lengths = batch['lengths']
                target_ids = batch['target_ids']

                # Get MSA features if available
                msa_features = None
                if 'msa_features' in batch:
                    msa_features = {
                        'conservation': batch['msa_features']['conservation'].to(self.device),
                        'gaps': batch['msa_features']['gaps'].to(self.device)
                    }

                # Get base predictions
                outputs = self.model(one_hot, msa_features, mask, lengths)
                base_coords = outputs['coordinates'].cpu().numpy()

                # For each sequence in the batch
                for i, target_id in enumerate(target_ids):
                    seq_len = lengths[i].item()

                    # Extract coordinates for this sequence
                    coords = base_coords[i, :seq_len]

                    # Generate different structures through sampling
                    structures = [coords]  # First structure is the base prediction

                    for _ in range(num_samples - 1):
                        # Create a perturbed structure
                        # We'll use a simple approach: add noise and refine
                        perturbed = coords + np.random.normal(0, temperature, coords.shape)

                        # In a real implementation, we would refine this structure
                        # using energy minimization or other techniques
                        # For now, we'll just add it as is
                        structures.append(perturbed)

                    # Store predictions
                    predictions[target_id] = structures

        return predictions

    def save_model(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_model_state': self.best_model_state
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_model_state = checkpoint['best_model_state']
        print(f"Model loaded from {path}")

    def format_predictions_for_submission(self, predictions, test_sequences):
        """
        Format predictions for submission.

        Args:
            predictions: Dictionary with predicted structures
            test_sequences: DataFrame with test sequences

        Returns:
            DataFrame formatted for submission
        """
        # Create a list to store submission rows
        submission_rows = []

        # For each test sequence
        for _, row in test_sequences.iterrows():
            target_id = row['target_id']
            sequence = row['sequence']

            # Skip if no prediction for this target
            if target_id not in predictions:
                print(f"Warning: No prediction for {target_id}")
                continue

            # Get predicted structures (num_samples of them)
            structures = predictions[target_id]
            num_samples = len(structures)

            # For each residue in the sequence
            for i, nt in enumerate(sequence):
                # Create a submission row
                submission_row = {
                    'ID': f"{target_id}_{i + 1}",
                    'resname': nt,
                    'resid': i + 1
                }

                # Add coordinates for each structure
                for j, structure in enumerate(structures):
                    if j < 5:  # We only need 5 structures for submission
                        submission_row[f'x_{j + 1}'] = structure[i][0]
                        submission_row[f'y_{j + 1}'] = structure[i][1]
                        submission_row[f'z_{j + 1}'] = structure[i][2]

                # If less than 5 structures, duplicate the last one
                for j in range(num_samples, 5):
                    submission_row[f'x_{j + 1}'] = structures[-1][i][0]
                    submission_row[f'y_{j + 1}'] = structures[-1][i][1]
                    submission_row[f'z_{j + 1}'] = structures[-1][i][2]

                submission_rows.append(submission_row)

        # Create a DataFrame from rows
        submission_df = pd.DataFrame(submission_rows)

        return submission_df

    def visualize_structure(self, target_id, coords, title=None):
        """
        Visualize a 3D structure.

        Args:
            target_id: Target ID for the structure
            coords: 3D coordinates (N x 3)
            title: Plot title
        """
        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot coordinates as points
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=20, c=range(len(coords)))

        # Connect consecutive residues with lines
        for i in range(len(coords) - 1):
            ax.plot(coords[i:i + 2, 0], coords[i:i + 2, 1], coords[i:i + 2, 2], 'k-', alpha=0.5)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'3D Structure for {target_id}')

        # Show the plot
        plt.tight_layout()
        plt.show()

        # Also create a 2D projection using PCA
        if len(coords) > 2:
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(coords)

            plt.figure(figsize=(8, 6))
            plt.scatter(coords_2d[:, 0], coords_2d[:, 1], s=20, c=range(len(coords)))

            # Connect consecutive residues with lines
            for i in range(len(coords_2d) - 1):
                plt.plot(coords_2d[i:i + 2, 0], coords_2d[i:i + 2, 1], 'k-', alpha=0.5)

            plt.title(f'2D Projection for {target_id}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.grid(True)
            plt.show()


# Example of how to use this class with the other modules
if __name__ == "__main__":
    from rna_structure_predictor import RNAStructurePredictor
    from secondary_structure_predictor import SecondaryStructurePredictor
    import torch
    from torch.utils.data import DataLoader

    # Initialize the base predictor
    rna_predictor = RNAStructurePredictor(temporal_cutoff="2022-05-27")
    rna_predictor.load_data()

    # Load MSAs for a subset of targets for demonstration
    sample_targets = rna_predictor.train_sequences['target_id'].iloc[:10].values
    rna_predictor.load_msas(sample_targets)

    # Prepare datasets
    datasets = rna_predictor.prepare_datasets()

    # Create datasets
    train_dataset = RNAStructure3DDataset(
        datasets['train_data'][:100],  # Use a subset for quick demonstration
        datasets['train_structures']
    )

    val_dataset = RNAStructure3DDataset(
        datasets['val_data'],
        datasets['val_structures']
    )

    test_dataset = RNAStructure3DDataset(
        datasets['test_data']
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=RNAStructure3DDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=RNAStructure3DDataset.collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=RNAStructure3DDataset.collate_fn
    )

    # Initialize 3D predictor
    predictor_3d = RNA3DPredictor(
        hidden_dim=64,  # Smaller for demonstration
        num_layers=2,
        dropout=0.2
    )

    # Train for just a few epochs for demonstration
    predictor_3d.train(train_loader, val_loader, num_epochs=3, patience=5)

    # Generate predictions
    predictions = predictor_3d.predict(test_loader, num_samples=5, temperature=1.0)

    # Visualize a sample prediction
    sample_target_id = sample_targets[0]
    if sample_target_id in predictions:
        for i, structure in enumerate(predictions[sample_target_id]):
            if i < 2:  # Just show a couple of structures for demonstration
                predictor_3d.visualize_structure(
                    sample_target_id,
                    structure,
                    title=f"Predicted Structure {i + 1} for {sample_target_id}"
                )

    # Format for submission
    submission_df = predictor_3d.format_predictions_for_submission(
        predictions,
        rna_predictor.test_sequences
    )

    print(submission_df.head())