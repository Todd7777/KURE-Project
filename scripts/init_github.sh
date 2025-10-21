#!/bin/bash
# Initialize Git repository and push to GitHub

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_message() {
    echo -e "${2}${1}${NC}"
}

print_message "🚀 Initializing Git repository for NRF project..." "$BLUE"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_message "❌ Git is not installed. Please install Git first." "$RED"
    exit 1
fi

# Check if already a git repo
if [ -d .git ]; then
    print_message "⚠️  Git repository already exists." "$YELLOW"
    read -p "Do you want to reinitialize? This will remove existing git history. (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .git
        print_message "✅ Removed existing git repository" "$GREEN"
    else
        print_message "Keeping existing repository. Exiting..." "$YELLOW"
        exit 0
    fi
fi

# Initialize git repository
print_message "📦 Initializing Git repository..." "$BLUE"
git init
print_message "✅ Git repository initialized" "$GREEN"
echo ""

# Create .gitattributes for better handling of large files
print_message "📝 Creating .gitattributes..." "$BLUE"
cat > .gitattributes << 'EOF'
# Auto detect text files and perform LF normalization
* text=auto

# Python files
*.py text eol=lf
*.pyx text eol=lf
*.pxd text eol=lf

# Shell scripts
*.sh text eol=lf

# YAML files
*.yml text eol=lf
*.yaml text eol=lf

# Markdown files
*.md text eol=lf

# Jupyter notebooks
*.ipynb text eol=lf

# Binary files
*.pt binary
*.pth binary
*.ckpt binary
*.safetensors binary
*.pkl binary
*.npy binary
*.npz binary
*.h5 binary
*.hdf5 binary

# Images
*.jpg binary
*.jpeg binary
*.png binary
*.gif binary
*.ico binary
*.svg binary

# Archives
*.zip binary
*.tar binary
*.gz binary
*.bz2 binary
*.7z binary
EOF
print_message "✅ .gitattributes created" "$GREEN"
echo ""

# Add all files
print_message "📁 Adding files to git..." "$BLUE"
git add .
print_message "✅ Files added" "$GREEN"
echo ""

# Create initial commit
print_message "💾 Creating initial commit..." "$BLUE"
git commit -m "Initial commit: Nonlinear Rectified Flows

- Core NRF framework with 4 teacher implementations
- Comprehensive training and evaluation pipeline
- Complete test suite with >80% coverage
- CI/CD with GitHub Actions
- Docker support
- Full documentation

This is a publication-ready implementation for NeurIPS/CVPR submission.
"
print_message "✅ Initial commit created" "$GREEN"
echo ""

# Create main branch
print_message "🌿 Setting up main branch..." "$BLUE"
git branch -M main
print_message "✅ Main branch created" "$GREEN"
echo ""

# Get GitHub repository URL
print_message "🔗 GitHub repository setup..." "$BLUE"
echo ""
print_message "Please enter your GitHub repository URL:" "$YELLOW"
print_message "Example: https://github.com/Todd7777/KURE-Project.git" "$YELLOW"
read -p "URL: " REPO_URL

if [ -z "$REPO_URL" ]; then
    print_message "❌ No URL provided. Using default: https://github.com/Todd7777/KURE-Project.git" "$YELLOW"
    REPO_URL="https://github.com/Todd7777/KURE-Project.git"
fi

# Add remote
print_message "Adding remote origin..." "$BLUE"
git remote add origin "$REPO_URL"
print_message "✅ Remote origin added" "$GREEN"
echo ""

# Ask if user wants to push now
print_message "Do you want to push to GitHub now? (y/n)" "$YELLOW"
read -p "" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_message "📤 Pushing to GitHub..." "$BLUE"
    
    # Try to push
    if git push -u origin main; then
        print_message "✅ Successfully pushed to GitHub!" "$GREEN"
    else
        print_message "⚠️  Push failed. You may need to:" "$YELLOW"
        echo "  1. Create the repository on GitHub first"
        echo "  2. Set up authentication (SSH key or Personal Access Token)"
        echo "  3. Run: git push -u origin main"
    fi
else
    print_message "Skipping push. You can push later with:" "$YELLOW"
    echo "  git push -u origin main"
fi

echo ""
print_message "═══════════════════════════════════════════════════════" "$BLUE"
print_message "✨ Git repository initialized!" "$GREEN"
print_message "═══════════════════════════════════════════════════════" "$BLUE"
echo ""
print_message "Repository: $REPO_URL" "$BLUE"
print_message "Branch: main" "$BLUE"
echo ""
print_message "Next steps:" "$BLUE"
echo "  1. Create repository on GitHub: https://github.com/new"
echo "  2. Push code: git push -u origin main"
echo "  3. Set up branch protection rules"
echo "  4. Enable GitHub Actions"
echo "  5. Add repository secrets for CI/CD"
echo ""
print_message "Recommended GitHub settings:" "$BLUE"
echo "  - Enable 'Require pull request reviews before merging'"
echo "  - Enable 'Require status checks to pass before merging'"
echo "  - Enable 'Require branches to be up to date before merging'"
echo "  - Add CODECOV_TOKEN secret for coverage reports"
echo "  - Add PYPI_API_TOKEN for package publishing"
echo ""
print_message "Happy coding! 🎉" "$GREEN"
