import os
import argparse
import glob
import re
from PIL import Image
from pymol import cmd

def animate_pdbs(input_dir, output_gif, width=800, height=600, delay=200):
    """
    Renders PDB files in a directory into an animated GIF using PyMOL and Pillow.
    """
    # 1. Collect and sort PDB files
    pdb_files = glob.glob(os.path.join(input_dir, "step_*.pdb"))
    # Sort by the number in the filename
    pdb_files.sort(key=lambda x: int(re.search(r"step_(\d+)", x).group(1)))
    
    # Add final_best.pdb or best.pdb if they exist
    best_pdb = os.path.join(input_dir, "final_best.pdb")
    if os.path.exists(best_pdb):
        pdb_files.append(best_pdb)
    elif os.path.exists(os.path.join(input_dir, "best.pdb")):
        pdb_files.append(os.path.join(input_dir, "best.pdb"))

    if not pdb_files:
        print(f"No PDB files found in {input_dir}")
        return

    print(f"Found {len(pdb_files)} structures to animate...")

    # 2. Setup PyMOL
    cmd.reinitialize()
    cmd.set("ray_shadows", 0)
    cmd.set("antialias", 2)
    cmd.bg_color("white")
    
    # Create temp directory for PNGs
    tmp_dir = os.path.join(input_dir, "_tmp_pngs")
    os.makedirs(tmp_dir, exist_ok=True)
    
    frames = []
    
    # 3. Load first structure to set orientation
    cmd.load(pdb_files[0], "ref")
    cmd.show_as("cartoon", "ref")
    cmd.color("marine", "ref")
    cmd.orient("ref")
    # Save the view
    view = cmd.get_view()
    cmd.delete("ref")

    # 4. Render each structure
    for i, pdb_path in enumerate(pdb_files):
        print(f"Rendering frame {i+1}/{len(pdb_files)}: {os.path.basename(pdb_path)}")
        obj_name = f"frame_{i}"
        cmd.load(pdb_path, obj_name)
        cmd.show_as("cartoon", obj_name)
        
        # Color by pLDDT if possible (B-factors in ESMFold output)
        # 0-50: Red, 50-70: Orange, 70-90: Yellow, 90-100: Blue
        cmd.spectrum("b", "rainbow_rev", selection=obj_name, minimum=50, maximum=90)
        
        cmd.set_view(view)
        
        png_path = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        cmd.png(png_path, width=width, height=height, ray=0)
        
        # Open with PIL
        frames.append(Image.open(png_path).convert("RGB"))
        
        cmd.delete(obj_name)

    # 5. Save GIF
    if frames:
        print(f"Saving animation to {output_gif}...")
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=delay,
            loop=0
        )
        print("Success!")
    
    # 6. Cleanup
    for f in glob.glob(os.path.join(tmp_dir, "*.png")):
        os.remove(f)
    os.rmdir(tmp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a GIF animation of protein folding steps.")
    parser.add_argument("input_dir", help="Directory containing step_*.pdb files")
    parser.add_argument("--output", "-o", default="animation.gif", help="Output GIF filename")
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--delay", type=int, default=300, help="Delay between frames in ms")
    
    args = parser.parse_args()
    
    # PyMOL needs to be initialized
    import pymol
    pymol.finish_launching(['pymol', '-qc']) # Headless
    
    animate_pdbs(args.input_dir, args.output, args.width, args.height, args.delay)
    
    pymol.cmd.quit()

