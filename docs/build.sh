#!/bin/bash
# Build documentation locally using uv

echo "Building documentation with Sphinx using uv..."
cd "$(dirname "$0")"
uv run sphinx-build -b html . _build/html

if [ $? -eq 0 ]; then
    echo "✅ Documentation built successfully!"
    echo "📂 Output directory: docs/_build/html/"
    echo "🌐 Open docs/_build/html/index.html in your browser to view"
else
    echo "❌ Documentation build failed!"
    exit 1
fi