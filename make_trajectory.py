import os
import glob
import re

def combine_pdbs(input_dir, output_pdb):
    """
    Combines individual PDB files into a single multi-model PDB file.
    """
    pdb_files = glob.glob(os.path.join(input_dir, "step_*.pdb"))
    pdb_files.sort(key=lambda x: int(re.search(r"step_(\d+)", x).group(1)))
    
    best_pdb = os.path.join(input_dir, "final_best.pdb")
    if os.path.exists(best_pdb):
        pdb_files.append(best_pdb)

    if not pdb_files:
        print(f"No PDB files found in {input_dir}")
        return

    print(f"Combining {len(pdb_files)} structures into {output_pdb}...")
    
    with open(output_pdb, "w") as outfile:
        for i, pdb_path in enumerate(pdb_files):
            outfile.write(f"MODEL     {i+1}\n")
            with open(pdb_path, "r") as infile:
                for line in infile:
                    if line.startswith(("ATOM", "HETATM", "TER")):
                        outfile.write(line)
            outfile.write("ENDMDL\n")
    
    print("Success!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine PDB steps into a trajectory PDB.")
    parser.add_argument("input_dir", help="Directory containing step_*.pdb files")
    parser.add_argument("output", help="Output multi-model PDB filename")
    args = parser.parse_args()
    
    combine_pdbs(args.input_dir, args.output)

