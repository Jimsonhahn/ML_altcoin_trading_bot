#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Timestamp Conversion Fix Script for Cryptocurrency Trading Data

This script addresses the timestamp conversion issue in cryptocurrency trading data files
by properly parsing and converting timestamp strings to a format that's compatible with
pandas and ML applications.

Usage:
    python fix_timestamps.py [--directory DATA_DIR] [--backup]

Options:
    --directory DATA_DIR    Directory containing the cryptocurrency data files (default: data/market_data/binance)
    --backup                Create backup of original files before modifying
"""

import os
import argparse
import glob
import pandas as pd
import numpy as np
import shutil
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def detect_timestamp_format(df):
    """Detect the format of the timestamp column."""
    if 'timestamp' not in df.columns:
        return None

    sample = df['timestamp'].iloc[0]
    if isinstance(sample, str):
        # Try to determine if it's a date string format
        formats = [
            ('%Y-%m-%d', '%Y-%m-%d'),
            ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S'),
            ('%Y%m%d', '%Y-%m-%d'),
            ('%d/%m/%Y', '%Y-%m-%d'),
            ('%m/%d/%Y', '%Y-%m-%d')
        ]

        for input_format, output_format in formats:
            try:
                datetime.strptime(sample, input_format)
                return input_format, output_format
            except ValueError:
                continue

        # If no string format matches, it might be a numeric timestamp
        try:
            float(sample)
            return 'numeric', None
        except ValueError:
            pass

    # If it's already a timestamp or datetime object
    if isinstance(sample, (pd.Timestamp, datetime, np.datetime64)):
        return 'datetime', None

    return None, None


def convert_timestamps(df, backup=False, filename=None):
    """Convert timestamps in the dataframe to a proper datetime format."""
    modified = False

    # Check if 'timestamp' column exists
    if 'timestamp' in df.columns:
        input_format, output_format = detect_timestamp_format(df)

        if input_format is None:
            logger.warning(f"Could not detect timestamp format in {filename}")
            return df, modified

        logger.info(f"Detected timestamp format: {input_format}")

        # Make a backup of the original data
        original_timestamp = df['timestamp'].copy()

        try:
            if input_format == 'datetime':
                # Already in correct format
                logger.info("Timestamps already in datetime format")
            elif input_format == 'numeric':
                # Convert numeric timestamp to datetime
                # Try milliseconds first (common in crypto data)
                if df['timestamp'].iloc[0] > 1000000000000:  # If in milliseconds (13 digits)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:  # If in seconds (10 digits)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                modified = True
            else:
                # Convert string timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], format=input_format)
                modified = True

            logger.info("Timestamp conversion successful")

        except Exception as e:
            logger.error(f"Error converting timestamps: {str(e)}")
            if backup:
                # Restore original data if conversion failed
                df['timestamp'] = original_timestamp
            return df, False

    return df, modified


def fix_file(filepath, output_dir=None, backup=False):
    """Fix timestamp issues in a single file."""
    try:
        logger.info(f"Processing file: {filepath}")

        # Create backup if requested
        if backup:
            backup_path = filepath + '.bak'
            shutil.copy2(filepath, backup_path)
            logger.info(f"Created backup: {backup_path}")

        # Read the CSV file
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            return False

        # Print columns before conversion
        logger.info(f"Columns before conversion: {df.columns.tolist()}")
        logger.info(f"Data types before conversion: {df.dtypes}")

        # Fix timestamps
        df, modified = convert_timestamps(df, backup, filepath)

        if not modified:
            logger.info(f"No modifications needed for {filepath}")
            return True

        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, os.path.basename(filepath))
        else:
            output_path = filepath

        # Save the fixed file
        df.to_csv(output_path, index=False)
        logger.info(f"Fixed file saved to: {output_path}")

        # Print columns after conversion
        logger.info(f"Columns after conversion: {df.columns.tolist()}")
        logger.info(f"Data types after conversion: {df.dtypes}")

        return True

    except Exception as e:
        logger.error(f"Unexpected error processing {filepath}: {str(e)}")
        return False


def fix_all_files(directory, output_dir=None, backup=False):
    """Process all CSV files in the directory."""
    # Get list of CSV files
    pattern = os.path.join(directory, "*.csv")
    files = glob.glob(pattern)

    if not files:
        logger.warning(f"No CSV files found in {directory}")
        return

    logger.info(f"Found {len(files)} CSV files")

    success_count = 0
    for file in files:
        if fix_file(file, output_dir, backup):
            success_count += 1

    logger.info(f"Successfully processed {success_count} out of {len(files)} files")


def create_test_data(df, output_path):
    """Create a small sample data file for testing ML components."""
    if len(df) > 100:
        sample_df = df.sample(min(100, len(df)))
    else:
        sample_df = df

    sample_df.to_csv(output_path, index=False)
    logger.info(f"Created test data sample: {output_path}")


def main():
    """Main function to fix timestamp issues in data files."""
    parser = argparse.ArgumentParser(description="Fix timestamp conversion issues in cryptocurrency data files")
    parser.add_argument("--directory", default="data/market_data/binance", help="Directory containing data files")
    parser.add_argument("--output", default=None, help="Output directory (default: overwrite original files)")
    parser.add_argument("--backup", action="store_true", help="Create backup of original files")
    parser.add_argument("--create-test-data", action="store_true", help="Create small test data samples")
    parser.add_argument("--test-data-dir", default="data/test_samples", help="Directory for test data samples")

    args = parser.parse_args()

    logger.info("=== Cryptocurrency Data Timestamp Fix ===")
    logger.info(f"Processing files in: {args.directory}")

    # Fix all files
    fix_all_files(args.directory, args.output, args.backup)

    # Create test data samples if requested
    if args.create_test_data:
        os.makedirs(args.test_data_dir, exist_ok=True)

        # Find one fixed file for each asset and timeframe to create test samples
        pattern = os.path.join(args.directory if not args.output else args.output, "*.csv")
        files = glob.glob(pattern)

        # Group files by asset and timeframe
        processed_patterns = set()
        for file in files:
            filename = os.path.basename(file)

            # Extract asset and timeframe (e.g., BTC_USDT_1d)
            parts = filename.split('_')
            if len(parts) >= 3:
                pattern = f"{parts[0]}_{parts[1]}_{parts[2]}"

                if pattern not in processed_patterns:
                    processed_patterns.add(pattern)

                    try:
                        df = pd.read_csv(file)
                        output_path = os.path.join(args.test_data_dir, f"{pattern}_sample.csv")
                        create_test_data(df, output_path)
                    except Exception as e:
                        logger.error(f"Error creating test sample for {file}: {str(e)}")

        logger.info(f"Created {len(processed_patterns)} test data samples in {args.test_data_dir}")

    logger.info("=== Processing Complete ===")


if __name__ == "__main__":
    main()