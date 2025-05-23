"""Script to run the dashboard."""

import argparse
import sys
from findataanalyzer.visualization.dashboard import Dashboard
from findataanalyzer.utils.config import load_config


def main():
    """Run the dashboard."""
    parser = argparse.ArgumentParser(description="Run the FinDataAnalyzer dashboard")
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Create and run dashboard
    dashboard = Dashboard(title="FinDataAnalyzer Dashboard")
    dashboard.run_server(debug=args.debug, port=args.port)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 