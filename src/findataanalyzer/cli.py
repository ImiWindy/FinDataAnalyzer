"""Command line interface for FinDataAnalyzer."""

import argparse
import sys

from findataanalyzer.core.analyzer import DataAnalyzer


def main():
    """Run the main CLI command."""
    parser = argparse.ArgumentParser(description="Financial Data Analysis Tool")
    parser.add_argument(
        "--data", 
        type=str, 
        help="Path to financial data file"
    )
    parser.add_argument(
        "--analyze", 
        action="store_true", 
        help="Run data analysis"
    )
    
    args = parser.parse_args()
    
    if args.data and args.analyze:
        analyzer = DataAnalyzer()
        results = analyzer.analyze_file(args.data)
        print(results)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 