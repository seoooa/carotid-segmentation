"""
Post-processing utilities using MONAI transforms
"""

import os
import nibabel as nib
import numpy as np
import torch
import click
from pathlib import Path
from typing import Optional, Union, List
from monai.transforms import KeepLargestConnectedComponent, Compose
from monai.data import MetaTensor


def apply_keep_largest_component(
    input_path: Union[str, Path], 
    output_path: Optional[Union[str, Path]] = None,
    applied_labels: Optional[Union[int, List[int]]] = None,
    is_onehot: Optional[bool] = None,
    independent: bool = True,
    connectivity: Optional[int] = None,
    num_components: int = 1
) -> str:
    """
    Applies KeepLargestConnectedComponent to a NIfTI file.
    
    Args:
        input_path: Path to the input NIfTI file
        output_path: Path to the output file (None for auto-generation)
        applied_labels: Label values to apply (None for all non-zero labels)
        is_onehot: Whether the data is one-hot encoded
        independent: Whether to process each label independently
        connectivity: Connectivity definition (1, 2, 3 for 6-, 18-, 26-connectivity)
        num_components: Number of connected components to keep
    
    Returns:
        str: Path to the output file
    """
    # Load input file
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"Loading input file: {input_path}")
    nii_img = nib.load(input_path)
    data = nii_img.get_fdata()
    
    # Convert numpy array to torch tensor
    # MONAI expects channel-first shape, so add channel dimension
    if data.ndim == 3:
        # Add channel dimension: (H, W, D) -> (1, H, W, D)
        tensor_data = torch.from_numpy(data).unsqueeze(0).float()
    else:
        tensor_data = torch.from_numpy(data).float()
    
    print(f"Data shape: {tensor_data.shape}")
    print(f"Unique values: {torch.unique(tensor_data)}")
    
    # Define KeepLargestConnectedComponent transform
    transform = KeepLargestConnectedComponent(
        applied_labels=applied_labels,
        is_onehot=is_onehot,
        independent=independent,
        connectivity=connectivity,
        num_components=num_components
    )
    
    # Apply transformation
    print("Applying KeepLargestConnectedComponent...")
    processed_data = transform(tensor_data)
    
    # Convert torch tensor to numpy array
    if isinstance(processed_data, torch.Tensor):
        processed_array = processed_data.squeeze(0).numpy()  # Remove channel dimension
    else:
        processed_array = processed_data.squeeze(0).numpy()
    
    # Set output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_largest_component{input_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save result
    print(f"Saving result: {output_path}")
    output_nii = nib.Nifti1Image(processed_array, nii_img.affine, nii_img.header)
    nib.save(output_nii, output_path)
    
    print(f"Post-processing completed! Saved file: {output_path}")
    return str(output_path)


def batch_process_largest_component(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.nii.gz",
    **kwargs
) -> List[str]:
    """
    Performs batch processing on multiple NIfTI files in a directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory (None for same as input directory)
        pattern: File pattern (default: "*.nii.gz")
        **kwargs: Additional arguments for apply_keep_largest_component function
    
    Returns:
        List[str]: List of paths to processed files
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    input_files = list(input_dir.glob(pattern))
    if not input_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return []
    
    processed_files = []
    for input_file in input_files:
        try:
            output_file = output_dir / f"{input_file.stem}_largest_component{input_file.suffix}"
            result_path = apply_keep_largest_component(
                input_path=input_file,
                output_path=output_file,
                **kwargs
            )
            processed_files.append(result_path)
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")
    
    return processed_files


def compare_before_after(
    original_path: Union[str, Path],
    processed_path: Union[str, Path]
) -> dict:
    """
    Provides comparison information before and after processing.
    
    Args:
        original_path: Path to the original file
        processed_path: Path to the processed file
    
    Returns:
        dict: Comparison information
    """
    # Load original file
    orig_nii = nib.load(original_path)
    orig_data = orig_nii.get_fdata()
    
    # Load processed file
    proc_nii = nib.load(processed_path)
    proc_data = proc_nii.get_fdata()
    
    # Calculate statistics
    orig_unique, orig_counts = np.unique(orig_data, return_counts=True)
    proc_unique, proc_counts = np.unique(proc_data, return_counts=True)
    
    comparison = {
        "original": {
            "unique_values": orig_unique.tolist(),
            "value_counts": {str(val): int(count) for val, count in zip(orig_unique, orig_counts)},
            "total_voxels": int(orig_data.size),
            "non_zero_voxels": int(np.count_nonzero(orig_data))
        },
        "processed": {
            "unique_values": proc_unique.tolist(),
            "value_counts": {str(val): int(count) for val, count in zip(proc_unique, proc_counts)},
            "total_voxels": int(proc_data.size),
            "non_zero_voxels": int(np.count_nonzero(proc_data))
        }
    }
    
    return comparison


def process_subject_output(
    arch_name: str,
    subject_id: str,
    base_dir: str = "/home/seoooa/project/coronary-artery",
    applied_labels: Optional[Union[int, List[int]]] = None,
    connectivity: Optional[int] = 3,
    num_components: int = 1
) -> str:
    """
    Post-processes the output file for a specific architecture and Subject ID.
    
    Args:
        arch_name: Model architecture name (e.g., 'SegResNet')
        subject_id: Subject ID (e.g., '790')
        base_dir: Project base directory
        applied_labels: Label values to apply
        connectivity: Connectivity definition (default: 3 for 26-connectivity)
        num_components: Number of connected components to keep
    
    Returns:
        str: Path to the post-processed file
    """
    base_path = Path(base_dir)
    
    # Construct input file path
    input_file = base_path / "result" / arch_name / "test" / f"Subj_{subject_id}_outputs.nii.gz"
    
    # Construct output file path
    output_dir = base_path / "result" / "postprocessing" / arch_name
    output_file = output_dir / f"Subj_{subject_id}_postprocessing.nii.gz"
    
    # Execute post-processing
    result_path = apply_keep_largest_component(
        input_path=input_file,
        output_path=output_file,
        applied_labels=applied_labels,
        connectivity=connectivity,
        num_components=num_components
    )
    
    return result_path


@click.command()
@click.option('--arch_name', '-a', required=True, help='Model architecture name (e.g., SegResNet)')
@click.option('--id', '-s', required=True, help='Subject ID (e.g., 02409947)')
@click.option('--base-dir', '-d', default='C:/SEOA/virtual/viceral-fat', 
              help='Project base directory path')
@click.option('--applied-labels', '-l', multiple=True, type=int,
              help='Label values to apply (e.g., -l 1 -l 2 or specify multiple times)')
@click.option('--connectivity', '-c', default=3, type=int,
              help='Connectivity definition (1=6-connectivity, 2=18-connectivity, 3=26-connectivity)')
@click.option('--num-components', '-n', default=1, type=int,
              help='Number of connected components to keep')
@click.option('--compare/--no-compare', default=True,
              help='Whether to output comparison information before/after processing')
def main(arch_name, id, base_dir, applied_labels, connectivity, num_components, compare):
    """
    Post-process segmentation results for specific architecture and Subject ID.
    Example usage:
        python post_processing.py --arch_name SegResNet --id 790
        python post_processing.py -a SegResNet -s 790 -l 1 -c 3 -n 1
    """
    try:
        # Process applied_labels
        labels = list(applied_labels) if applied_labels else None
        
        print(f"Starting post-processing")
        print(f"   - Architecture: {arch_name}")
        print(f"   - Subject ID: {id}")
        print(f"   - Base directory: {base_dir}")
        print(f"   - Applied labels: {labels}")
        print(f"   - Connectivity: {connectivity}")
        print(f"   - Number of components: {num_components}")
        print("-" * 50)
        
        # Execute post-processing
        result_path = process_subject_output(
            arch_name=arch_name,
            subject_id=id,
            base_dir=base_dir,
            applied_labels=labels,
            connectivity=connectivity,
            num_components=num_components
        )
        
        print(f"Post-processing completed!")
        print(f"   Output file: {result_path}")
        
        # Output comparison information
        if compare:
            print("\nComparison before/after processing:")
            base_path = Path(base_dir)
            input_file = base_path / "result" / arch_name / "test" / f"Subj_{id}_outputs.nii.gz"
            
            comparison = compare_before_after(input_file, result_path)
            
            print(f"Original non-zero voxels: {comparison['original']['non_zero_voxels']:,}")
            print(f"Processed non-zero voxels: {comparison['processed']['non_zero_voxels']:,}")
            
            # Calculate voxel count change
            original_count = comparison['original']['non_zero_voxels']
            processed_count = comparison['processed']['non_zero_voxels']
            if original_count > 0:
                reduction_ratio = (1 - processed_count / original_count) * 100
                print(f"Voxel reduction ratio: {reduction_ratio:.2f}%")
            
            print(f"Original unique values: {comparison['original']['unique_values']}")
            print(f"Processed unique values: {comparison['processed']['unique_values']}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return 1
    except Exception as e:
        print(f"Error occurred: {e}")
        return 1
    
    return 0


# Example usage
if __name__ == "__main__":
    main()
