#!/usr/bin/env python3
"""
Generate a manifest.json file listing all available daily paper archives.
This helps the web interface know which dates have available data.
"""

import json
import os
from datetime import datetime
from pathlib import Path


def generate_manifest():
    """Scan the output directory and create a manifest of available dates."""
    output_dir = Path("out")
    manifest = {
        "dates": [],
        "last_updated": datetime.now().isoformat()
    }
    
    # Pattern for daily archive files: output_YYYY_MMDD.md
    daily_files = sorted(output_dir.glob("output_????_????.md"))
    
    for file_path in daily_files:
        # Extract date from filename
        filename = file_path.name
        try:
            # Parse the date from filename format: output_YYYY_MMDD.md
            date_str = filename.replace("output_", "").replace(".md", "")
            year = date_str[:4]
            month = date_str[5:7]
            day = date_str[7:9]
            
            # Validate the date
            date_obj = datetime(int(year), int(month), int(day))
            
            # Add to manifest
            manifest["dates"].append({
                "date": date_obj.strftime("%Y-%m-%d"),
                "filename": filename,
                "displayDate": date_obj.strftime("%B %d, %Y")
            })
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid filename: {filename} - {e}")
            continue
    
    # Sort dates in descending order (most recent first)
    manifest["dates"].sort(key=lambda x: x["date"], reverse=True)
    
    # Write manifest file
    manifest_path = Path("manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Generated manifest.json with {len(manifest['dates'])} dates")
    print(f"Date range: {manifest['dates'][-1]['date']} to {manifest['dates'][0]['date']}")
    
    return manifest_path


if __name__ == "__main__":
    generate_manifest()