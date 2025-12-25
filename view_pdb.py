#!/usr/bin/env python3
"""
Interactive PDB Viewer - Opens in browser with 3Dmol.js
Usage: python view_pdb.py <pdb_file_or_directory>
"""
import os
import sys
import glob
import re
import webbrowser
import tempfile
import argparse

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Protein Viewer</title>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; }}
        .container {{ display: flex; height: 100vh; }}
        .sidebar {{ width: 250px; padding: 20px; background: #16213e; overflow-y: auto; }}
        .viewer {{ flex: 1; position: relative; }}
        #viewer {{ width: 100%; height: 100%; }}
        h2 {{ color: #e94560; margin-top: 0; }}
        .btn {{ display: block; width: 100%; padding: 10px; margin: 5px 0; background: #0f3460; color: #eee; border: none; border-radius: 5px; cursor: pointer; text-align: left; }}
        .btn:hover {{ background: #e94560; }}
        .btn.active {{ background: #e94560; }}
        select, input {{ width: 100%; padding: 8px; margin: 5px 0; background: #0f3460; color: #eee; border: 1px solid #e94560; border-radius: 5px; }}
        label {{ display: block; margin-top: 15px; color: #e94560; }}
        .controls {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #0f3460; }}
        .info {{ font-size: 12px; color: #888; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>🧬 Protein Viewer</h2>
            
            <label>Structure:</label>
            <select id="pdb-select" onchange="loadPDB(this.value)">
                {pdb_options}
            </select>
            
            <div class="controls">
                <label>Style:</label>
                <select id="style-select" onchange="updateStyle()">
                    <option value="cartoon">Cartoon</option>
                    <option value="stick">Stick</option>
                    <option value="sphere">Sphere</option>
                    <option value="line">Line</option>
                </select>
                
                <label>Color:</label>
                <select id="color-select" onchange="updateStyle()">
                    <option value="spectrum">Rainbow (N→C)</option>
                    <option value="chain">By Chain</option>
                    <option value="ss">Secondary Structure</option>
                    <option value="residue">By Residue</option>
                </select>
                
                <label>Background:</label>
                <select id="bg-select" onchange="updateBackground()">
                    <option value="#1a1a2e">Dark</option>
                    <option value="#ffffff">White</option>
                    <option value="#000000">Black</option>
                </select>
                
                <button class="btn" onclick="viewer.spin(true)">🔄 Spin On</button>
                <button class="btn" onclick="viewer.spin(false)">⏹️ Spin Off</button>
                <button class="btn" onclick="viewer.zoomTo()">🔍 Reset View</button>
            </div>
            
            <div class="info">
                <p>🖱️ <b>Controls:</b></p>
                <p>• Left drag: Rotate</p>
                <p>• Scroll: Zoom</p>
                <p>• Right drag: Pan</p>
            </div>
        </div>
        <div class="viewer">
            <div id="viewer"></div>
        </div>
    </div>
    
    <script>
        const pdbData = {pdb_data};
        let viewer = $3Dmol.createViewer('viewer', {{backgroundColor: '#1a1a2e'}});
        
        function loadPDB(name) {{
            viewer.removeAllModels();
            viewer.addModel(pdbData[name], 'pdb');
            updateStyle();
            viewer.zoomTo();
            viewer.render();
        }}
        
        function updateStyle() {{
            const style = document.getElementById('style-select').value;
            const color = document.getElementById('color-select').value;
            
            let styleObj = {{}};
            let colorSpec = color === 'spectrum' ? 'spectrum' : 
                           color === 'chain' ? 'chain' :
                           color === 'ss' ? 'ssPyMol' : 'amino';
            
            if (style === 'cartoon') {{
                styleObj = {{cartoon: {{color: colorSpec}}}};
            }} else if (style === 'stick') {{
                styleObj = {{stick: {{colorscheme: colorSpec}}}};
            }} else if (style === 'sphere') {{
                styleObj = {{sphere: {{colorscheme: colorSpec, scale: 0.3}}}};
            }} else if (style === 'line') {{
                styleObj = {{line: {{colorscheme: colorSpec}}}};
            }}
            
            viewer.setStyle({{}}, styleObj);
            viewer.render();
        }}
        
        function updateBackground() {{
            const bg = document.getElementById('bg-select').value;
            viewer.setBackgroundColor(bg);
            viewer.render();
        }}
        
        // Load first structure
        const firstPDB = Object.keys(pdbData)[0];
        if (firstPDB) loadPDB(firstPDB);
    </script>
</body>
</html>
"""

def create_viewer(pdb_path):
    """Create an HTML viewer for PDB file(s)."""
    pdb_data = {}
    
    if os.path.isdir(pdb_path):
        # Load all PDBs from directory
        pdb_files = glob.glob(os.path.join(pdb_path, "step_*.pdb"))
        pdb_files.sort(key=lambda x: int(re.search(r"step_(\d+)", x).group(1)))
        
        final = os.path.join(pdb_path, "final_best.pdb")
        if os.path.exists(final):
            pdb_files.append(final)
        
        for pdb_file in pdb_files:
            name = os.path.basename(pdb_file)
            with open(pdb_file, 'r') as f:
                pdb_data[name] = f.read()
    else:
        # Single file
        with open(pdb_path, 'r') as f:
            pdb_data[os.path.basename(pdb_path)] = f.read()
    
    # Generate options HTML
    options = "\n".join([f'<option value="{name}">{name}</option>' for name in pdb_data.keys()])
    
    # Escape PDB data for JavaScript
    import json
    pdb_json = json.dumps(pdb_data)
    
    html = HTML_TEMPLATE.format(pdb_options=options, pdb_data=pdb_json)
    
    # Write to temp file and open
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html)
        temp_path = f.name
    
    print(f"Opening viewer in browser...")
    webbrowser.open(f"file://{temp_path}")
    print(f"HTML saved to: {temp_path}")

def main():
    parser = argparse.ArgumentParser(description="Interactive PDB Viewer (opens in browser)")
    parser.add_argument("path", help="PDB file or directory containing PDB files")
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"Error: {args.path} not found")
        sys.exit(1)
    
    create_viewer(args.path)

if __name__ == "__main__":
    main()

