#!/bin/bash
# Download large test data files from external storage
# These files are not included in the repository to keep it lightweight

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/.."

echo "========================================="
echo "  aiter Test Data Downloader"
echo "========================================="
echo ""

# TODO: Configure actual storage location (S3, GCS, etc.)
# STORAGE_URL="https://storage.example.com/aiter-test-data"

echo "NOTE: Test data storage not yet configured."
echo ""
echo "Large test data files have been removed from the repository"
echo "to reduce its size. If you need these files for testing:"
echo ""
echo "  1. Contact the maintainers for access to test data"
echo "  2. Or generate your own test data using the provided tools"
echo ""
echo "Files that were moved to external storage:"
echo "  - op_tests/test_jenga_vsa/*.pt (PyTorch tensors)"
echo "  - op_tests/dump_data/*.pt (Debug data)"
echo "  - Large benchmark CSV files"
echo ""
echo "For most development work, these files are not required."
echo ""
echo "No action taken - storage location needs to be configured."
exit 1

# Example download implementation (uncomment when storage is configured):
# download_file() {
#     local remote_path=$1
#     local local_path=$2
#     
#     if [ -f "$local_path" ]; then
#         echo "  ✓ $local_path already exists, skipping"
#         return
#     fi
#     
#     echo "  Downloading $remote_path..."
#     mkdir -p "$(dirname "$local_path")"
#     curl -fSL "${STORAGE_URL}/${remote_path}" -o "$local_path"
#     echo "  ✓ Downloaded to $local_path"
# }

# download_file "test_jenga_vsa/jenga_query_normal.pt" "$REPO_ROOT/op_tests/test_jenga_vsa/jenga_query_normal.pt"
# download_file "test_jenga_vsa/jenga_value.pt" "$REPO_ROOT/op_tests/test_jenga_vsa/jenga_value.pt"
# download_file "test_jenga_vsa/jenga_key.pt" "$REPO_ROOT/op_tests/test_jenga_vsa/jenga_key.pt"

# echo "Done!"
# exit 0
