#!/usr/bin/env python3
"""
Country-Specific CO2 Data Extractor
Extracts emission data for a specific country from the OWID CO2 dataset.
"""

import pandas as pd
import sys
import os
from datetime import datetime

def extract_country_data(country_name, dataset_path="owid-co2-data.csv", output_dir=None):
    """
    Extract country-specific data from CO2 dataset.
    
    Parameters:
    -----------
    country_name : str
        Name of the country to extract (case-insensitive)
    dataset_path : str
        Path to the OWID CO2 dataset CSV file
    output_dir : str, optional
        Directory to save the output file (defaults to ~/Downloads)
    
    Returns:
    --------
    str : Path to the saved file
    """
    
    # Set default output directory to Downloads
    if output_dir is None:
        output_dir = os.path.expanduser("~/Downloads")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the dataset
        print(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded successfully. Total records: {len(df)}")
        
        # Filter data for the specified country (case-insensitive)
        df_country = df[df["country"].str.lower() == country_name.lower()]
        
        if df_country.empty:
            print(f"\n❌ No data found for country: {country_name}")
            print("\nAvailable countries (first 20):")
            unique_countries = df["country"].unique()[:20]
            for i, c in enumerate(unique_countries, 1):
                print(f"  {i}. {c}")
            print(f"  ... and {len(df['country'].unique()) - 20} more countries")
            return None
        
        # Sort by year
        df_country = df_country.sort_values(by="year").reset_index(drop=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_country_name = country_name.replace(" ", "_").lower()
        output_filename = f"{safe_country_name}_co2_data_{timestamp}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save to CSV
        df_country.to_csv(output_path, index=False)
        
        # Display summary
        print(f"\n✅ Data extraction successful!")
        print(f"Country: {country_name}")
        print(f"Records found: {len(df_country)}")
        print(f"Year range: {df_country['year'].min()} - {df_country['year'].max()}")
        print(f"Columns: {len(df_country.columns)}")
        print(f"\nSaved to: {output_path}")
        
        return output_path
        
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at {dataset_path}")
        print("Please ensure the file exists in the current directory.")
        return None
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return None


def main():
    """Command-line interface for the extractor."""
    
    if len(sys.argv) < 2:
        print("Usage: python extract_country_data.py <country_name> [dataset_path] [output_dir]")
        print("\nExamples:")
        print("  python extract_country_data.py Ghana")
        print("  python extract_country_data.py 'South Africa'")
        print("  python extract_country_data.py Nigeria owid-co2-data.csv ~/Documents")
        sys.exit(1)
    
    country = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) > 2 else "owid-co2-data.csv"
    output = sys.argv[3] if len(sys.argv) > 3 else None
    
    extract_country_data(country, dataset, output)


if __name__ == "__main__":
    main()
