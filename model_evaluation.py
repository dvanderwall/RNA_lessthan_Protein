import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import pdist, squareform
import argparse
import datetime
import json
import time
from tqdm import tqdm
import glob

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, SUBMISSIONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def calculate_rmsd(coords1, coords2):
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


def align_structures(target_coords, mobile_coords):
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


def evaluate_prediction(pred_df, true_df):
    """
    Evaluate a prediction against ground truth.

    Args:
        pred_df: DataFrame with predictions
        true_df: DataFrame with ground truth

    Returns:
        Dictionary with evaluation results
    """
    # Merge on ID
    merged = pd.merge(pred_df, true_df, on=['ID', 'resname', 'resid'], how='inner', suffixes=('_pred', '_true'))

    # Calculate RMSD for each structure
    target_ids = merged['ID'].str.split('_').str[0].unique()
    results = {}

    for target_id in target_ids:
        target_rows = merged[merged['ID'].str.startswith(f"{target_id}_")]

        # Extract target resids
        resids = target_rows['resid'].values

        # Check which ground truth structures are available
        true_cols = [col for col in target_rows.columns if col.startswith('x_') and col.endswith('_true')]
        num_true_structures = len(true_cols) // 3

        # Check which predicted structures are available
        pred_cols = [col for col in target_rows.columns if col.startswith('x_') and col.endswith('_pred')]
        num_pred_structures = len(pred_cols) // 3

        print(
            f"Target {target_id}: {len(resids)} residues, {num_true_structures} true structures, {num_pred_structures} predictions")

        # Extract coordinates
        true_coords = []
        for i in range(1, num_true_structures + 1):
            x_col = f'x_{i}_true'
            y_col = f'y_{i}_true'
            z_col = f'z_{i}_true'

            if x_col in target_rows and y_col in target_rows and z_col in target_rows:
                coords = np.array([
                    target_rows[x_col].values,
                    target_rows[y_col].values,
                    target_rows[z_col].values
                ]).T

                # Check for NaN values
                if not np.isnan(coords).any():
                    true_coords.append(coords)

        pred_coords = []
        for i in range(1, num_pred_structures + 1):
            x_col = f'x_{i}_pred'
            y_col = f'y_{i}_pred'
            z_col = f'z_{i}_pred'

            coords = np.array([
                target_rows[x_col].values,
                target_rows[y_col].values,
                target_rows[z_col].values
            ]).T

            # Check for NaN values
            if not np.isnan(coords).any():
                pred_coords.append(coords)

        # Calculate RMSDs
        target_results = {
            'target_id': target_id,
            'num_residues': len(resids),
            'rmsd_values': []
        }

        for pred_idx, pred_struct in enumerate(pred_coords):
            best_rmsd = float('inf')
            best_true_idx = -1

            for true_idx, true_struct in enumerate(true_coords):
                # Align and calculate RMSD
                aligned_pred = align_structures(true_struct, pred_struct)
                rmsd = calculate_rmsd(true_struct, aligned_pred)

                if rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_true_idx = true_idx

            if best_true_idx >= 0:
                target_results['rmsd_values'].append({
                    'pred_idx': pred_idx,
                    'true_idx': best_true_idx,
                    'rmsd': best_rmsd
                })

        # Calculate mean RMSD
        if target_results['rmsd_values']:
            rmsds = [r['rmsd'] for r in target_results['rmsd_values']]
            target_results['mean_rmsd'] = np.mean(rmsds)
            target_results['min_rmsd'] = np.min(rmsds)
            target_results['max_rmsd'] = np.max(rmsds)
        else:
            target_results['mean_rmsd'] = None
            target_results['min_rmsd'] = None
            target_results['max_rmsd'] = None

        results[target_id] = target_results

    # Calculate overall metrics
    all_rmsds = []
    for target_id, target_results in results.items():
        if target_results['rmsd_values']:
            all_rmsds.extend([r['rmsd'] for r in target_results['rmsd_values']])

    overall_results = {
        'targets': results,
        'all_rmsds': all_rmsds
    }

    if all_rmsds:
        overall_results['mean_rmsd'] = np.mean(all_rmsds)
        overall_results['median_rmsd'] = np.median(all_rmsds)
        overall_results['min_rmsd'] = np.min(all_rmsds)
        overall_results['max_rmsd'] = np.max(all_rmsds)
        overall_results['std_rmsd'] = np.std(all_rmsds)

    return overall_results


def visualize_evaluation(eval_results, output_path=None):
    """
    Visualize evaluation results.

    Args:
        eval_results: Dictionary with evaluation results
        output_path: Path to save visualization
    """
    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot RMSD distribution
    all_rmsds = eval_results['all_rmsds']

    if all_rmsds:
        axes[0].hist(all_rmsds, bins=20, alpha=0.7)
        axes[0].axvline(eval_results['mean_rmsd'], color='r', linestyle='--',
                        label=f"Mean: {eval_results['mean_rmsd']:.2f} Å")
        axes[0].axvline(eval_results['median_rmsd'], color='g', linestyle='--',
                        label=f"Median: {eval_results['median_rmsd']:.2f} Å")
        axes[0].set_xlabel('RMSD (Å)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('RMSD Distribution')
        axes[0].legend()

        # Plot RMSD vs sequence length
        target_lengths = []
        target_mean_rmsds = []

        for target_id, target_results in eval_results['targets'].items():
            if target_results['mean_rmsd'] is not None:
                target_lengths.append(target_results['num_residues'])
                target_mean_rmsds.append(target_results['mean_rmsd'])

        axes[1].scatter(target_lengths, target_mean_rmsds, alpha=0.7)
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('Mean RMSD (Å)')
        axes[1].set_title('RMSD vs Sequence Length')
        axes[1].grid(True, alpha=0.3)

        # Add correlation coefficient
        if len(target_lengths) > 1:
            corr = np.corrcoef(target_lengths, target_mean_rmsds)[0, 1]
            axes[1].text(0.05, 0.95, f"Correlation: {corr:.2f}",
                         transform=axes[1].transAxes,
                         verticalalignment='top', bbox={'alpha': 0.5, 'pad': 10})

    else:
        axes[0].text(0.5, 0.5, "No RMSD data available",
                     ha='center', va='center')
        axes[1].text(0.5, 0.5, "No RMSD data available",
                     ha='center', va='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")

    plt.show()


def evaluate_submission(submission_path, validation_path=None):
    """
    Evaluate a submission file against validation data.

    Args:
        submission_path: Path to submission file
        validation_path: Path to validation data file (if None, use default)

    Returns:
        Evaluation results
    """
    # Load submission file
    submission_df = pd.read_csv(submission_path)

    # Determine if it's a validation or test submission
    is_validation = False

    # Check if submission contains validation targets
    validation_ids = None

    if validation_path:
        # Use provided validation path
        validation_df = pd.read_csv(validation_path)
        validation_ids = set(validation_df['ID'].str.split('_').str[0])
    else:
        # Use default validation path
        default_validation_path = os.path.join(DATA_DIR, "validation_labels.csv")
        if os.path.exists(default_validation_path):
            validation_df = pd.read_csv(default_validation_path)
            validation_ids = set(validation_df['ID'].str.split('_').str[0])

    # Check if submission contains validation targets
    submission_ids = set(submission_df['ID'].str.split('_').str[0])

    if validation_ids and submission_ids.intersection(validation_ids):
        is_validation = True
        print("Submission contains validation targets. Evaluating against validation data.")

        # Filter submission to only include validation targets
        submission_df = submission_df[submission_df['ID'].str.split('_').str[0].isin(validation_ids)]

        # Evaluate
        eval_results = evaluate_prediction(submission_df, validation_df)

        # Visualize
        visualization_path = os.path.join(
            OUTPUT_DIR,
            f"validation_eval_{os.path.basename(submission_path).split('.')[0]}.png"
        )
        visualize_evaluation(eval_results, visualization_path)

        # Format for terminal output
        print("\nEvaluation Results:")
        print(f"Mean RMSD: {eval_results['mean_rmsd']:.2f} Å")
        print(f"Median RMSD: {eval_results['median_rmsd']:.2f} Å")
        print(f"Min RMSD: {eval_results['min_rmsd']:.2f} Å")
        print(f"Max RMSD: {eval_results['max_rmsd']:.2f} Å")
        print(f"Std Dev: {eval_results['std_rmsd']:.2f} Å")

        print("\nTarget-wise Results:")
        for target_id, results in eval_results['targets'].items():
            if results['mean_rmsd'] is not None:
                print(f"{target_id}: Mean RMSD = {results['mean_rmsd']:.2f} Å, "
                      f"Min = {results['min_rmsd']:.2f} Å, "
                      f"Max = {results['max_rmsd']:.2f} Å, "
                      f"Residues = {results['num_residues']}")

        return eval_results
    else:
        print("Submission contains test targets only. No ground truth available for evaluation.")

        # Check if submission has the correct format
        required_columns = ['ID', 'resname', 'resid',
                            'x_1', 'y_1', 'z_1',
                            'x_2', 'y_2', 'z_2',
                            'x_3', 'y_3', 'z_3',
                            'x_4', 'y_4', 'z_4',
                            'x_5', 'y_5', 'z_5']

        missing_columns = [col for col in required_columns if col not in submission_df.columns]

        if missing_columns:
            print(f"Error: Submission is missing required columns: {missing_columns}")
            return None

        # Check for NaN values
        nan_columns = submission_df.columns[submission_df.isna().any()].tolist()

        if nan_columns:
            print(f"Warning: Submission contains NaN values in columns: {nan_columns}")
            print(f"Number of NaN values: {submission_df[nan_columns].isna().sum().sum()}")

        # Check number of targets
        test_ids = submission_df['ID'].str.split('_').str[0].unique()
        print(f"Submission contains {len(test_ids)} test targets")

        # Check number of residues per target
        for target_id in test_ids:
            target_rows = submission_df[submission_df['ID'].str.startswith(f"{target_id}_")]
            print(f"Target {target_id}: {len(target_rows)} residues")

        return {
            'submission_df': submission_df,
            'test_ids': test_ids
        }


def create_submission(predictions, output_path=None):
    """
    Create a submission file from predictions.

    Args:
        predictions: Dictionary with predictions
        output_path: Path to save submission file

    Returns:
        Path to submission file
    """
    # Create submission rows
    submission_rows = []

    for target_id, ensemble in predictions.items():
        # Get sequence for this target
        # In a real implementation, we would get this from the test data
        sequence = "A" * len(ensemble[0])  # Placeholder

        # For each residue
        for i in range(len(sequence)):
            submission_row = {
                'ID': f"{target_id}_{i + 1}",
                'resname': sequence[i],
                'resid': i + 1
            }

            # Add coordinates for each structure
            for j, structure in enumerate(ensemble[:5]):  # Take first 5 structures
                submission_row[f'x_{j + 1}'] = structure[i][0]
                submission_row[f'y_{j + 1}'] = structure[i][1]
                submission_row[f'z_{j + 1}'] = structure[i][2]

            submission_rows.append(submission_row)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save to file
    if output_path is None:
        output_path = os.path.join(
            SUBMISSIONS_DIR,
            f"submission_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    return output_path


def find_latest_submission():
    """
    Find the latest submission file in the submissions directory.

    Returns:
        Path to latest submission file
    """
    submission_files = glob.glob(os.path.join(SUBMISSIONS_DIR, "submission_*.csv"))

    if not submission_files:
        print("No submission files found")
        return None

    latest_file = max(submission_files, key=os.path.getmtime)
    return latest_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Evaluate RNA Structure Predictions')

    # Add arguments
    parser.add_argument('--submission', type=str, help='Path to submission file')
    parser.add_argument('--validation', type=str, help='Path to validation data file')
    parser.add_argument('--output', type=str, help='Output directory')

    # Parse arguments
    args = parser.parse_args()

    # Determine submission path
    submission_path = args.submission

    if submission_path is None:
        # Find latest submission file
        submission_path = find_latest_submission()

        if submission_path is None:
            print("No submission file found or specified")
            return

    # Evaluate submission
    print(f"Evaluating submission: {submission_path}")
    eval_results = evaluate_submission(submission_path, args.validation)

    print("Evaluation complete")
    return eval_results


if __name__ == "__main__":
    main()