#!/bin/bash
# Setup environment for al-genomics-benchmark
# Source this file: source setup_env.sh

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add BLAST+ to PATH
export PATH="$PATH:${PROJECT_ROOT}/external/blast/ncbi-blast-2.17.0+/bin"

# Add HashFrag to PATH
export PATH="$PATH:${PROJECT_ROOT}/external/hashFrag/src"

# Verify installations
echo "Environment setup complete!"
echo ""
echo "BLAST+ version: $(blastn -version | head -1)"
echo "HashFrag: $(which hashFrag)"
echo ""
echo "You can now run HashFrag commands and create splits."
