#!/usr/bin/env python
"""Build vocabulary from regular SMILES (Stage A) by adding dummy anchors."""

import csv
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from PolyDiffusion.src.chem.vocab import AnchorSafeVocab

def main():
    input_file = "Data/small_molecules_part.csv"
    output_file = "PolyDiffusion/vocab.txt"
    limit = 500000

    print(f"Building vocabulary from {input_file}...")

    # Read SMILES and add dummy anchors to make them AP-SMILES
    corpus = []
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if 'SMILES' in row and row['SMILES']:
                # Add dummy anchors to make it valid AP-SMILES
                ap_smiles = f"[*:1]{row['SMILES']}[*:2]"
                corpus.append(ap_smiles)
            if len(corpus) >= limit:
                break

    if not corpus:
        print("ERROR: No SMILES found in input file!")
        return

    print(f"Building vocab from {len(corpus)} examples...")
    vocab = AnchorSafeVocab.build(corpus)

    # Save vocabulary
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(output_path)

    print(f"Wrote vocab with {len(vocab)} tokens to {output_file}")

if __name__ == "__main__":
    main()
