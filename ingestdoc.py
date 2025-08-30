#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any


def extract_text_data(file_path: str) -> Dict[str, Any]:
    """Extract basic data from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {
        'filename': os.path.basename(file_path),
        'filepath': file_path,
        'size_bytes': os.path.getsize(file_path),
        'line_count': len(content.splitlines()),
        'char_count': len(content),
        'word_count': len(content.split())
    }


def extract_json_data(file_path: str) -> Dict[str, Any]:
    """Extract data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        base_info = {
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'size_bytes': os.path.getsize(file_path),
            'type': 'json'
        }
        
        if isinstance(data, dict):
            base_info.update({
                'json_keys': ','.join(data.keys()) if data else '',
                'json_key_count': len(data) if data else 0
            })
        elif isinstance(data, list):
            base_info.update({
                'json_array_length': len(data),
                'json_key_count': 0
            })
        
        return base_info
    except json.JSONDecodeError:
        return extract_text_data(file_path)


def extract_file_data(file_path: str) -> Dict[str, Any]:
    """Extract data from a file based on its type."""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.json':
        return extract_json_data(file_path)
    else:
        return extract_text_data(file_path)


def append_to_csv(data: Dict[str, Any], csv_path: str) -> None:
    """Append extracted data to CSV file."""
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(data)


def main():
    parser = argparse.ArgumentParser(
        description='Extract data from a file and append it to a CSV file'
    )
    parser.add_argument(
        'input_file',
        help='Path to the input file to process'
    )
    parser.add_argument(
        '-o', '--output',
        default='ingested_data.csv',
        help='Output CSV file path (default: ingested_data.csv)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.verbose:
            print(f"Processing file: {args.input_file}")
        
        extracted_data = extract_file_data(args.input_file)
        
        if args.verbose:
            print(f"Extracted data: {extracted_data}")
        
        append_to_csv(extracted_data, args.output)
        
        if args.verbose:
            print(f"Data appended to: {args.output}")
        else:
            print(f"Ingested {args.input_file} → {args.output}")
    
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()