---
name: model_updater
description: Update Model.md to reflect the current architecture and components of the PolyDiffusion model. Keeps documentation synchronized with code without detailed change tracking. (project)
allowed-tools: Read, Grep, Glob, Edit, Bash
---

# Model.md Updater

## Purpose

Keep `Results/Model.md` synchronized with the current state of the PolyDiffusion model architecture. This skill updates the documentation to reflect how the model works NOW, not track historical changes.

**Key principle**: This is NOT a changelog - it's a living document that describes the current model architecture, components, and behavior.

## When to Use This Skill

Trigger this skill after making changes to:
- Model architecture files (`src/models/*.py`)
- Core components (embeddings, attention, heads, layers)
- Training stage definitions (Stage A, B, C configurations or logic)
- Sampling workflow and algorithms
- Conditioning mechanisms (ConditionNet, cross-attention)
- Diffusion process implementation
- Any model component that affects the architecture overview

## Workflow

### Step 1: Understand Current State

**Ask the user:**
- "What model components or architecture were recently modified?"
- "Which files were changed?" (if not obvious from context)

**Then read the modified files** to understand the current implementation.

### Step 2: Section Mapping

Map code changes to Model.md sections:

| Code Area | Model.md Section |
|-----------|------------------|
| Model architecture files (dit_token.py, modules.py) | Core Components (Section 2) |
| Training stage configs/scripts | High-Level Overview (Section 1) |
| New modules, layers, embeddings | Core Components table |
| Sampling logic (sample_cli.py, diffusion_token.py) | Sampling Workflow (Section 3) |
| Architectural differences, design choices | Comparison with GPT-2 (Section 4) |
| Component file locations | References (Section 5) |

### Step 3: Read Current Documentation

```bash
# Read the current Model.md
Read Results/Model.md
```

Identify which sections need updates based on Step 2 mapping.

### Step 4: Update Documentation to Reflect Current State

**IMPORTANT**: REPLACE outdated information with current functionality. Do NOT append or create changelog entries.

For each affected section:

**If updating Core Components table:**
- Update component descriptions to match current implementation
- Add new components if they exist in code
- Remove components that no longer exist
- Update file references to current locations
- Ensure "Key Differences from GPT-2" column reflects actual differences

**If updating High-Level Overview:**
- Update training stage descriptions to match current configs
- Ensure stage purposes align with actual training scripts
- Update any architectural diagrams or descriptions

**If updating Sampling Workflow:**
- Update steps to match current sampling implementation
- Ensure method names and parameters are accurate
- Update noise schedule descriptions if changed

**If updating GPT-2 Comparison:**
- Update comparison points to reflect current architecture
- Add new distinguishing features
- Remove comparison points that are no longer accurate

**If updating References:**
- Add links to new files
- Update file paths if files moved
- Ensure all referenced components have corresponding links

### Step 5: Apply Updates

Use the Edit tool to update Model.md:

```
Edit Results/Model.md
- old_string: [exact current text from Model.md]
- new_string: [updated text reflecting current implementation]
```

**Best practices:**
- Make focused edits (one section at a time)
- Preserve markdown formatting and table structure
- Keep technical accuracy paramount
- Maintain consistent terminology with codebase
- Include file path references for traceability

### Step 6: Validation Checklist

After updates, verify:
- [ ] Component table matches actual model components in code
- [ ] Training stage descriptions match current configs
- [ ] Sampling workflow reflects current implementation
- [ ] File references point to existing files
- [ ] Technical details are accurate (layer counts, dimensions, etc.)
- [ ] Comparisons with GPT-2 are still valid
- [ ] Markdown formatting is preserved
- [ ] No outdated information remains

## Important Guidelines

1. **Replace, Don't Append**: Update existing content to reflect current state. Don't add "Update:" or "Changed:" entries.

2. **Source of Truth**: The code is the source of truth. Model.md describes what the code does NOW.

3. **Component-Focused**: Model.md is about architecture. For workflow details, use Pipeline.md instead.

4. **Precision**: Be technically precise. Include dimensions, types, and specific implementation details.

5. **File References**: Always include file paths when describing components (e.g., `src/models/dit_token.py`).

6. **Table Maintenance**: Keep the Core Components table well-formatted and comprehensive.

## Example Interactions

### Example 1: New Model Component Added

**User**: "I added a new TemporalAttention layer to the model"

**Skill Actions**:
1. Read the implementation file to understand the component
2. Identify Core Components table needs update
3. Add new row to table with:
   - Component name: "TemporalAttention"
   - Description: [based on code implementation]
   - Key differences: [how it differs from standard attention]
   - File reference: [path to implementation]
4. Update any relevant sections mentioning attention mechanisms
5. Apply edits using Edit tool

### Example 2: Training Stage Modified

**User**: "I changed Stage B to use a different loss function"

**Skill Actions**:
1. Read stage_b.yaml and training script
2. Identify High-Level Overview section needs update
3. Update Stage B description to reflect new loss function
4. If loss function is new component, add to Core Components
5. Apply edits to replace old loss function mention with new one

### Example 3: Sampling Algorithm Changed

**User**: "Updated the sampling to use DDIM instead of DDPM"

**Skill Actions**:
1. Read sampling implementation code
2. Identify Sampling Workflow section needs update
3. Update workflow steps to describe DDIM process
4. Update any algorithm-specific details
5. Check if GPT-2 comparison needs update
6. Apply edits to replace DDPM references with DDIM

## Tips for Best Results

1. **Read Before Edit**: Always read modified files first to understand current state
2. **Verify File Paths**: Check that referenced files exist at specified paths
3. **Maintain Structure**: Keep Model.md's existing section structure
4. **Technical Depth**: Model.md is technical documentation - include implementation details
5. **Component Table**: Keep this comprehensive - it's the core reference
6. **Cross-Reference**: Mention Pipeline.md for workflow details when appropriate

## File Locations Reference

### Model.md Structure
- **Location**: `Results/Model.md`
- **Section 1**: High-Level Overview (3 training stages)
- **Section 2**: Core Components (table with 11+ components)
- **Section 3**: Sampling Workflow (3-step process)
- **Section 4**: Comparison with GPT-2
- **Section 5**: References (links to source files)

### Key Model Files to Reference
- `src/models/dit_token.py` - Main DiT model implementation
- `src/models/modules.py` - Model components and modules
- `src/models/diffusion_token.py` - Diffusion process
- `configs/stage_*.yaml` - Training stage configurations
- `scripts/sample_cli.py` - Sampling implementation

### Related Skills
- Use `pipeline_updater` for workflow and training pipeline updates
- Use `model_updater` (this skill) for architecture and component updates

## Summary

The model_updater skill keeps Model.md synchronized with the PolyDiffusion codebase by REPLACING outdated architecture descriptions with current implementations. It focuses on components, stages, and architectural decisions - not workflow or procedures.

**To use**: Invoke this skill after modifying model architecture, components, or core implementation details.

**Remember**: Model.md describes how the model IS, not how it changed.

---

*Last Updated: 2025-11-13*
