#!/bin/bash

# Quick Start Script for Multi-Target Baking

echo "=========================================="
echo "Multi-Target Baking Quick Start"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with:"
    echo "  BREAD_API_KEY=your_bread_api_key"
    echo "  ANTHROPIC_API_KEY=your_anthropic_api_key"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Workflow:"
echo ""
echo "1. Run baseline evaluation:"
echo "     python evaluate.py"
echo ""
echo "2. Run multi-target bake:"
echo "     python run_bake.py"
echo ""
echo "3. After baking completes, run post-bake evaluation:"
echo "     python evaluate.py --model YOUR_BAKED_MODEL_NAME"
echo ""
echo "=========================================="

