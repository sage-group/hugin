import argparse

#import pystac_client as pyc

def main():
    parser = argparse.ArgumentParser(
        description="HuginEO Directory to STAC Tools"
    )

    parser.add_argument(
        "--input-directory",
        type=str,
        required=True,
        help="Directory containing assets",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        required=True,
        help="Output directory for STAC Catalog"
    )
    parser.parse_args()

if __name__ == "__main__":
    main()