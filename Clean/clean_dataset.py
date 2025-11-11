"""
Data Cleaning Script for Shopping Behavior Dataset
Removes null values, duplicate rows, and blank rows
"""

import pandas as pd
import os
from datetime import datetime


def clean_dataset(input_file, output_file=None):
    """
    Clean the dataset by removing null values, duplicates, and blank rows

    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    output_file : str, optional
        Path to save the cleaned CSV file. If None, creates a default name.

    Returns:
    --------
    dict : Statistics about the cleaning process
    """

    print("=" * 60)
    print("DATA CLEANING PROCESS")
    print("=" * 60)
    print(f"\nReading dataset from: {input_file}")

    # Read the dataset
    df = pd.read_csv(input_file)

    # Store initial statistics
    initial_rows = len(df)
    initial_cols = len(df.columns)

    print(f"\nInitial dataset shape: {df.shape}")
    print(f"  - Rows: {initial_rows}")
    print(f"  - Columns: {initial_cols}")

    # Check for null values
    print("\n" + "-" * 60)
    print("CHECKING FOR NULL VALUES")
    print("-" * 60)
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls > 0:
        print(f"Found {total_nulls} null values:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  - {col}: {count} null values")
    else:
        print("No null values found.")

    # Remove rows with any null values
    df_cleaned = df.dropna()
    rows_after_null_removal = len(df_cleaned)
    rows_removed_nulls = initial_rows - rows_after_null_removal

    # Check for duplicate rows
    print("\n" + "-" * 60)
    print("CHECKING FOR DUPLICATE ROWS")
    print("-" * 60)
    duplicates = df_cleaned.duplicated().sum()
    print(f"Found {duplicates} duplicate rows")

    # Remove duplicate rows
    df_cleaned = df_cleaned.drop_duplicates()
    rows_after_duplicate_removal = len(df_cleaned)

    # Check for blank rows (rows where all values are empty strings or whitespace)
    print("\n" + "-" * 60)
    print("CHECKING FOR BLANK ROWS")
    print("-" * 60)

    # Identify rows where all string columns are empty or whitespace
    blank_mask = df_cleaned.apply(lambda row: all(
        str(val).strip() == '' if isinstance(val, str) else False
        for val in row
    ), axis=1)
    blank_rows = blank_mask.sum()
    print(f"Found {blank_rows} blank rows")

    # Remove blank rows
    df_cleaned = df_cleaned[~blank_mask]
    rows_after_blank_removal = len(df_cleaned)

    # Reset index
    df_cleaned = df_cleaned.reset_index(drop=True)

    # Final statistics
    final_rows = len(df_cleaned)
    total_rows_removed = initial_rows - final_rows

    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"Initial rows:              {initial_rows}")
    print(f"Rows removed (nulls):      {rows_removed_nulls}")
    print(f"Rows removed (duplicates): {duplicates}")
    print(f"Rows removed (blanks):     {blank_rows}")
    print(f"Total rows removed:        {total_rows_removed}")
    print(f"Final rows:                {final_rows}")
    print(f"Data retention rate:       {(final_rows/initial_rows)*100:.2f}%")

    # Save cleaned dataset
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{base_name}_cleaned.csv"

    df_cleaned.to_csv(output_file, index=False)
    print(f"\n[SUCCESS] Cleaned dataset saved to: {output_file}")

    # Create statistics dictionary
    stats = {
        'initial_rows': initial_rows,
        'initial_cols': initial_cols,
        'rows_removed_nulls': rows_removed_nulls,
        'rows_removed_duplicates': duplicates,
        'rows_removed_blanks': blank_rows,
        'total_rows_removed': total_rows_removed,
        'final_rows': final_rows,
        'final_cols': len(df_cleaned.columns),
        'retention_rate': (final_rows/initial_rows)*100,
        'output_file': output_file
    }

    # Additional data quality checks
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    print(f"\nColumn names:")
    for i, col in enumerate(df_cleaned.columns, 1):
        print(f"  {i}. {col}")

    print(f"\nData types:")
    print(df_cleaned.dtypes.to_string())

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE!")
    print("=" * 60)

    return df_cleaned, stats


if __name__ == "__main__":
    # Define input and output file paths
    input_file = "data/shopping_behavior_updated.csv"
    output_file = "data/shopping_behavior_cleaned.csv"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        exit(1)

    # Clean the dataset
    cleaned_df, stats = clean_dataset(input_file, output_file)

    print(f"\nYou can now use the cleaned dataset: {stats['output_file']}")
