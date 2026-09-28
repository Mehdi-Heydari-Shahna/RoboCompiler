"""Compatibility alias for the v22 standard-library integrity checker."""
import sys
from verify_package import main
if __name__ == '__main__':
    sys.exit(0 if main() else 1)
