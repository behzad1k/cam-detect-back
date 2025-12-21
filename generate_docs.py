#!/usr/bin/env python3
"""
Generate API Documentation
Run this script to update API documentation after code changes
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.docs.openapi_generator import generate_openapi_spec

if __name__ == "__main__":
    print("🔄 Generating API documentation...")
    
    # Generate OpenAPI spec and markdown docs
    generate_openapi_spec(
        app,
        output_file="docs/api_documentation.json"
    )
    
    print("\n✅ Documentation generation complete!")
    print("\nGenerated files:")
    print("  - docs/api_documentation.json (OpenAPI/Swagger spec)")
    print("  - docs/api_documentation.md (Markdown documentation)")
    print("\nYou can now:")
    print("  1. Import JSON into Postman/Insomnia")
    print("  2. View in Swagger UI at http://localhost:8000/docs")
    print("  3. Share Markdown with frontend team")
