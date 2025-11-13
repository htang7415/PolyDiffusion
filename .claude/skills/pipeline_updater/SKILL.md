---
name: pipeline-updater
description: Update Pipeline.md to reflect the current state and working mechanisms of the latest model. Keeps documentation synchronized with code without detailed change tracking.
allowed-tools: Read, Grep, Glob, Edit, Bash
---

# Pipeline Documentation Updater

## Purpose
This skill keeps [Pipeline.md](Results/Pipeline.md) synchronized with the current codebase. It updates documentation to describe **how the latest model works now**, replacing outdated information with current functionality. This is NOT a changelog - it maintains a living document that reflects the present state of the system.

## When to Use This Skill

Invoke this skill after making changes to:
- **Training scripts**: `src/train/train_stage_*.py`, `src/train/common.py`
- **Sampling code**: `src/sampling/*.py`, `scripts/sample_cli.py`
- **Configuration files**: `configs/*.yaml`
- **Data handling**: `src/data/*.py`, `scripts/preprocess_*.py`
- **Model architecture**: `src/models/*.py`
- **Evaluation scripts**: `scripts/evaluate_stage.py`
- **Vocabulary/tokenization**: `src/chem/vocab_*.py`, `src/chem/ap_smiles.py`

## Workflow

### Step 1: Understand Current State
First, identify what code was modified:
1. Ask the user: "What code did you modify?"
2. Read the modified files to understand how they currently work
3. Use Grep to find related code patterns if needed
4. Identify which Pipeline.md sections describe this functionality

### Step 2: Section Mapping

Map code changes to Pipeline.md sections:

| Code Area | Pipeline.md Section | Section Number |
|-----------|-------------------|----------------|
| Model architecture (`src/models/*.py`) | Architecture Overview | Section 1 |
| Training scripts (`src/train/*.py`) | Training Workflows | Section 4 |
| Data handling (`src/data/*.py`) | Data Specifications | Section 3 |
| Evaluation (`scripts/evaluate_*.py`) | Evaluation & Metrics | Section 5 |
| Sampling (`src/sampling/*.py`, `scripts/sample_cli.py`) | Sampling Playbooks | Section 6 |
| Common issues/solutions | Troubleshooting | Section 7 |
| File locations | Reference Index | Section 8 |

### Step 3: Read Current Documentation
Read [Pipeline.md](Results/Pipeline.md) to understand current content and structure.

### Step 4: Update Documentation to Reflect Current State

**Key Principle**: REPLACE outdated information with current information. Don't track what changed - just describe how it works now.

For each affected section:

#### Section 1: Architecture Overview
**When**: Model architecture is modified

**What to do**: Replace component descriptions to reflect current architecture
- Update class names and their roles
- Update configuration parameters to match current configs
- Describe how the current architecture works
- Keep section length similar by replacing old info with new info

#### Section 3: Data Specifications
**When**: Data handling code is modified

**What to do**: Replace data format descriptions with current requirements
- Update required/optional columns to match current code
- Update preprocessing commands to current syntax
- Replace file format examples with current formats
- Keep section length similar

#### Section 4: Training Workflows
**When**: Training scripts are modified

**What to do**: Replace training commands and descriptions with current versions
- Update CLI commands to match current script arguments
- Update config parameter descriptions to match current YAML files
- Replace expected output examples with current logging format
- Update checkpoint behavior description
- Keep section length similar - replace old commands with new ones

Example: If `--num-epochs` changed to `--epochs`, replace the old flag everywhere.

#### Section 5: Evaluation & Metrics
**When**: Evaluation code is modified

**What to do**: Replace metric descriptions with current metrics
- Update metric definitions to match current implementations
- Replace evaluation command examples with current syntax
- Update interpretation guidance if metric meaning changed
- If new metrics added, replace less important old ones to keep length similar

#### Section 6: Sampling Playbooks
**When**: Sampling code is modified

**What to do**: Replace sampling commands with current versions
- Update CLI commands to match current script
- Update parameter descriptions and defaults
- Replace output format examples
- Update usage tips if sampling behavior changed
- Keep section length similar - replace old examples with new ones

#### Section 7: Troubleshooting
**When**: Common issues are resolved or new ones discovered

**What to do**: Update troubleshooting entries
- Replace solved issues with current issues
- Update solutions if resolution approach changed
- Keep section focused on currently relevant problems
- Remove obsolete issues to maintain length

#### Section 8: Reference Index
**When**: File locations change

**What to do**: Update file path references
- Replace old file paths with current ones
- Update descriptions if file purposes changed
- Keep section length similar

### Step 5: Apply Updates

**Important**: REPLACE outdated content, don't append to it

1. Use the Edit tool to replace outdated text with current information
2. Preserve existing markdown formatting and section structure
3. Maintain similar section lengths - if adding new content, remove old content
4. Cross-reference with actual code to ensure accuracy
5. Update "Last updated: YYYY-MM" at the bottom of Pipeline.md

### Step 6: Validation Checklist

Before completing, verify:
- [ ] File paths in examples are correct and exist
- [ ] Config parameters match actual YAML files
- [ ] CLI flags match current script arguments
- [ ] Code snippets use correct class/function names
- [ ] Cross-references between sections are consistent
- [ ] "Last updated: YYYY-MM" date updated at bottom
- [ ] Markdown formatting is clean and consistent
- [ ] Document length hasn't grown significantly

## Important Guidelines

1. **REPLACE, Don't Append**: Replace outdated content with current content - don't make the document longer
2. **Describe Current State**: Write "the model does X" not "we changed it to do X"
3. **No Change Tracking**: Don't document what changed - document how it works now
4. **Preserve Structure**: Keep section organization and numbering
5. **Be Specific**: Use actual file paths, class names, and parameter names from current code
6. **Use Examples**: Provide concrete CLI commands that work with current code
7. **Maintain Length**: If adding new content, remove old/obsolete content
8. **Update Date Only**: Just update "Last updated: YYYY-MM" at bottom - no detailed changelog

## Example Interactions

### Example 1: New Parameter Added

**User**: "I added a `--temperature` parameter to the sampling script"

**Skill Actions**:
1. Read `scripts/sample_cli.py` to see how temperature works
2. Read Pipeline.md Section 6 (Sampling Playbooks)
3. **Replace** the sampling command examples with new versions that include `--temperature`
4. Add temperature parameter description to the parameter list
5. Update "Last updated: 2025-11" at bottom
6. Done - NO changelog entry, just updated examples

### Example 2: Training Script Changed

**User**: "I modified train_stage_b.py to use a different loss function"

**Skill Actions**:
1. Read `src/train/train_stage_b.py` to understand current loss function
2. Read Pipeline.md Section 4.2 (Stage B Training)
3. **Replace** the loss function description with current implementation
4. Update any relevant config parameters
5. Update "Last updated: 2025-11" at bottom
6. Done - documentation now reflects current training approach

### Example 3: Architecture Change

**User**: "I added a new attention mechanism to the model"

**Skill Actions**:
1. Read `src/models/*.py` to understand current architecture
2. Read Pipeline.md Section 1 (Architecture Overview)
3. **Replace** the architecture description to include new attention mechanism
4. Update component list with new class names
5. Update config parameters if new ones added
6. Update "Last updated: 2025-11" at bottom
7. Done - documentation describes current architecture

## Tips for Best Results

- **Focus on "Now"**: Describe how the current code works, not what changed
- **Replace, Don't Add**: Swap old info for new info to maintain document length
- **Be Accurate**: Always verify against actual current code
- **Be Thorough**: Check all related sections that mention the modified component
- **Be Concise**: Keep descriptions clear and focused on current functionality
- **Match Style**: Follow existing documentation formatting and tone

## File Locations Reference

- **Pipeline.md**: `Results/Pipeline.md` (target document to update)
- **Configs**: `configs/*.yaml` (model/training configurations)
- **Training**: `src/train/train_stage_*.py` (training scripts)
- **Sampling**: `src/sampling/*.py`, `scripts/sample_cli.py` (generation)
- **Data**: `src/data/*.py` (data loading and preprocessing)
- **Models**: `src/models/*.py` (model architecture)
- **Evaluation**: `scripts/evaluate_stage.py` (metrics and evaluation)
- **Vocabulary**: `src/chem/vocab_*.py` (tokenization)

---

## Summary

This skill keeps Pipeline.md as a **living document** that describes the current state of the PolyDiffusion system. It updates documentation to reflect current functionality by **replacing outdated information** with current information, maintaining consistent document length, and avoiding detailed change tracking.

**Usage**: Invoke with `/pipeline-updater` after modifying any code component.

---

**Last Updated**: 2025-11
**Skill Version**: 2.0 (Revised to focus on current state, not change history)
