#!/bin/bash
# trustcv PyPI Deployment Script

echo "========================================="
echo "trustcv PyPI Deployment Helper"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo "1. Checking Python..."
if command -v python3 &> /dev/null; then
    echo -e "   ${GREEN}✓${NC} Python $(python3 --version)"
else
    echo -e "   ${RED}✗${NC} Python not found"
    exit 1
fi

# Step 2: Check/Install build tools
echo ""
echo "2. Checking build tools..."
if python3 -c "import build, twine" 2>/dev/null; then
    echo -e "   ${GREEN}✓${NC} Build tools installed"
else
    echo -e "   ${YELLOW}!${NC} Installing build tools..."
    pip install --upgrade build twine
fi

# Step 3: Run tests
echo ""
echo "3. Running tests..."
if python3 test_all.py > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓${NC} All tests passed"
else
    echo -e "   ${RED}✗${NC} Tests failed. Run 'python test_all.py' for details"
    exit 1
fi

# Step 4: Clean previous builds
echo ""
echo "4. Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/ 2>/dev/null
echo -e "   ${GREEN}✓${NC} Cleaned"

# Step 5: Build package
echo ""
echo "5. Building package..."
if python3 -m build > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓${NC} Package built"
    echo "   Files created:"
    ls -lh dist/ | tail -n +2 | awk '{print "     - " $NF " (" $5 ")"}'
else
    echo -e "   ${RED}✗${NC} Build failed"
    exit 1
fi

# Step 6: Deployment instructions
echo ""
echo "========================================="
echo -e "${GREEN}Package ready for deployment!${NC}"
echo "========================================="
echo ""
echo "To deploy to TestPyPI (recommended first):"
echo -e "  ${YELLOW}python3 -m twine upload --repository testpypi dist/*${NC}"
echo ""
echo "To test installation from TestPyPI:"
echo -e "  ${YELLOW}pip install --index-url https://test.pypi.org/simple/ trustcv${NC}"
echo ""
echo "To deploy to production PyPI:"
echo -e "  ${YELLOW}python3 -m twine upload dist/*${NC}"
echo ""
echo "Note: You'll need PyPI tokens. See PYPI_DEPLOYMENT_GUIDE.md for details."
echo ""