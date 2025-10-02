# Solution: Preserving Wildcards in Polymer Generation

## Problem Summary

**Issue**: Generated polymer samples have **0 wildcards** instead of 2

**Root Cause**: SAFE library's `decode()` function **closes all attachment points** by default, converting polymer repeat units `*CCC*` into closed molecules `C1CCC1`

## The SAFE Problem for Polymers

### How SAFE Works:
1. **Encoding**: `*CCC*` → `C12CCC21` (attachment points become numbered indices)
2. **Decoding**: `C12CCC21` → `C1CCC1` (indices are connected, closing the molecule)

### What We Need:
- Keep 2 attachment points **open** (as wildcards `*`)
- Only close internal attachment points

## Solution Options

### ❌ Option A: Modify SAFE Decoder (Complex)
Requires deep understanding of SAFE internals and custom fragment assembly

### ✅ Option B: Train Without Wildcards, Add Post-hoc (RECOMMENDED)
1. Train on SAFE strings (which don't have wildcards)
2. At generation time, identify terminal positions
3. Add wildcards back to both ends

### ✅ Option C: Use Different Encoding (Alternative)
Don't use SAFE for polymers - use SELFIES or custom encoding that preserves wildcards

## Recommended Solution: Post-Processing Approach

Instead of trying to preserve wildcards through SAFE encode/decode, **add them back after generation**:

### Modified `polymer_utils.py`:

```python
def safe_to_polymer_smiles(safe_str, fix=True):
    """
    Convert SAFE to polymer SMILES by:
    1. Decode SAFE normally (produces closed molecule)
    2. Identify 2 "cut points" to open the ring/chain
    3. Insert wildcards at those points
    """
    if not safe_str or safe_str.strip() == '':
        return None
    
    # Decode SAFE (will produce closed molecule)
    try:
        smiles = sf.decode(safe_str, canonical=True, ignore_errors=True)
        if not smiles:
            return None
    except:
        return None
    
    # If already has 2 wildcards, great!
    if smiles.count('*') == 2:
        return smiles
    
    # Convert to molecule
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # Strategy: Find 2 bonds to break and add wildcards
    # For polymers, we want to break bonds that would create a linear repeat unit
    
    # Method 1: If it's a ring, break the ring at 2 points
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        # Get the largest ring
        rings = ring_info.BondRings()
        if rings:
            largest_ring = max(rings, key=len)
            if len(largest_ring) >= 2:
                # Break 2 bonds in the ring
                bond_idx1 = largest_ring[0]
                bond_idx2 = largest_ring[len(largest_ring)//2]
                
                # Fragment on these bonds with dummy labels
                fragmented = Chem.FragmentOnBonds(mol, [bond_idx1, bond_idx2],
                                                  dummyLabels=[(0,0), (0,0)])
                
                # Get the main fragment
                frags = Chem.GetMolFrags(fragmented, asMols=True)
                if frags:
                    main_frag = max(frags, key=lambda x: x.GetNumAtoms())
                    smiles_with_wildcards = Chem.MolToSmiles(main_frag)
                    smiles_with_wildcards = smiles_with_wildcards.replace('[*]', '*')
                    
                    if smiles_with_wildcards.count('*') == 2:
                        return smiles_with_wildcards
    
    # Method 2: For linear/branched structures, identify terminal carbons
    # (This is more complex - would need heuristics)
    
    # If we can't create a valid polymer structure, return None
    return None
```

## Alternative Solution: Simpler Heuristic

```python
def safe_to_polymer_smiles_simple(safe_str, fix=True):
    """
    Simplified approach: Try to create polymer by breaking specific patterns
    """
    if not safe_str or safe_str.strip() == '':
        return None
    
    try:
        smiles = sf.decode(safe_str, canonical=True, ignore_errors=True)
        if not smiles:
            return None
    except:
        return None
    
    # If already has 2 wildcards
    if smiles.count('*') == 2:
        return smiles
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    # Check for dummy atoms first
    dummy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) == 2:
        # Perfect! SAFE preserved attachment points
        smiles_with_wildcards = Chem.MolToSmiles(mol)
        return smiles_with_wildcards.replace('[*]', '*')
    
    # Heuristic: For small molecules (<20 atoms), try breaking at random bonds
    if mol.GetNumHeavyAtoms() <= 20:
        # Find all non-aromatic single bonds
        candidate_bonds = []
        for bond in mol.GetBonds():
            if not bond.GetIsAromatic() and bond.GetBondType() == Chem.BondType.SINGLE:
                candidate_bonds.append(bond.GetIdx())
        
        if len(candidate_bonds) >= 2:
            # Try breaking first and last single bond
            import random
            random.shuffle(candidate_bonds)
            
            for i in range(len(candidate_bonds)):
                for j in range(i+1, len(candidate_bonds)):
                    try:
                        fragmented = Chem.FragmentOnBonds(mol, [candidate_bonds[i], candidate_bonds[j]],
                                                          dummyLabels=[(0,0), (0,0)])
                        frags = Chem.GetMolFrags(fragmented, asMols=True)
                        
                        # Take the largest fragment
                        main_frag = max(frags, key=lambda x: x.GetNumAtoms())
                        
                        # Check if it has 2 dummy atoms
                        dummy_count = sum(1 for atom in main_frag.GetAtoms() 
                                        if atom.GetAtomicNum() == 0)
                        
                        if dummy_count == 2:
                            smiles_result = Chem.MolToSmiles(main_frag)
                            return smiles_result.replace('[*]', '*')
                    except:
                        continue
    
    # Unable to create valid polymer
    return None
```

## What to Do Next

### Step 1: Test if SAFE Preserves Wildcards

Run:
```bash
python test_safe_roundtrip.py
```

**If wildcards ARE preserved** → Problem is in `polymer_utils.py` logic  
**If wildcards are LOST** → Need post-processing approach

### Step 2: Update polymer_utils.py

Replace `safe_to_polymer_smiles()` with one of the solutions above.

### Step 3: Re-run Generation

```bash
cd genmol
python scripts/exps/denovo_polymer.py
```

### Step 4: If Still Empty

The issue might be that the **model isn't generating valid SAFE strings** at all. Check:

```python
# Add debug in denovo_polymer.py after line 106:
samples_raw = sampler.de_novo_generation(5, softmax_temp=0.5, randomness=0.5)
print(f"DEBUG Raw samples: {samples_raw}")

# Check what the sampler actually returns
```

## Long-term Solution

For production polymer generation, consider:

1. **Train on larger dataset** (PI1M with 1M polymers)
2. **Use custom tokenization** that treats `*` as a special token
3. **Modify SAFE library** to have a "polymer mode" that doesn't close terminal attachments
4. **Use SELFIES** instead of SAFE (better for polymers)
5. **Post-process all generated molecules** to identify and mark attachment points

## Summary

The core issue is that SAFE wasn't designed for polymer repeat units with open attachment points. Your options:

1. ✅ **Easiest**: Post-process to add wildcards by breaking bonds
2. ⚠️ **Medium**: Modify SAFE library (requires deep dive)
3. ✅ **Best Long-term**: Use different encoding (SELFIES/custom)

I recommend implementing the **post-processing approach** first to get something working, then evaluate if you need a more sophisticated solution.

---

**Created**: October 2, 2025  
**Status**: Solution provided, awaiting implementation testing

