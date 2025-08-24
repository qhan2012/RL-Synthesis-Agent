#!/bin/bash
"""
Simple wrapper script for temporary files cleanup
Author: Created for RL Synthesis Agent project
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧹 RL Synthesis Agent - Temporary Files Cleanup${NC}"
echo -e "${BLUE}=================================================${NC}"
echo ""

# Check if Python script exists
if [ ! -f "cleanup_temp_files_safe.py" ]; then
    echo -e "${RED}❌ Error: cleanup_temp_files_safe.py not found${NC}"
    exit 1
fi

# Function to show usage
show_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 scan          # Scan and show temporary files"
    echo "  $0 dry-run       # Simulate cleanup operation"
    echo "  $0 move          # Actually move temp files (with confirmation)"
    echo "  $0 move --force  # Move temp files without confirmation"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 scan          # Safe scan to see what would be cleaned"
    echo "  $0 dry-run       # Test run without moving files"
    echo "  $0 move          # Clean up with confirmation prompt"
    echo ""
}

# Check arguments
if [ $# -eq 0 ]; then
    show_usage
    echo -e "${BLUE}Running default scan...${NC}"
    echo ""
    python cleanup_temp_files_safe.py --scan
    exit 0
fi

case "$1" in
    "scan")
        echo -e "${GREEN}🔍 Scanning for temporary files...${NC}"
        python cleanup_temp_files_safe.py --scan
        ;;
    "dry-run")
        echo -e "${YELLOW}🔄 Running dry-run simulation...${NC}"
        python cleanup_temp_files_safe.py --dry-run
        ;;
    "move")
        if [ "$2" = "--force" ]; then
            echo -e "${RED}⚠️  Moving files without confirmation...${NC}"
            python cleanup_temp_files_safe.py --move --force
        else
            echo -e "${YELLOW}🚚 Moving temporary files...${NC}"
            python cleanup_temp_files_safe.py --move
        fi
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac
