# PORTAL: Perturbation Output via Reporter Transcriptional Activity in Lineages

Tang, A., Ardy, R.C., Mendes, R.E., & Norman, T.M. Scaling perturbations: beyond genome-scale CRISPR screens. bioRxiv (2026). https://www.biorxiv.org/content/10.64898/2026.01.16.699948v1

Analysis code for PORTAL screens with clonal barcoding and dual transcriptional reporters.

All data will be made publicly available upon publication at [SRA: to be deposited] and [Zenodo: to be deposited].

## Overview

This is not a general-purpose software package, but specific analysis code developed for our manuscript. The scripts process dual-guide CRISPR screen data with:
- Clonal barcoding to track lineages
- Dual transcriptional reporters (identity + reporter)
- Pairwise genetic interaction analysis
- Hierarchical clustering and complex/cluster recall

## System Requirements

- **Python**: 3.11.13 (tested)
- **Operating System**: Red Hat Enterprise Linux 8.7 (tested)
- **RAM**: 16-32 GB for standard analysis, 64-128 GB for large-scale screens

### Installation

```bash
git clone https://github.com/norman-lab-msk/PORTAL.git
cd PORTAL
pip install -r requirements.txt
```

## Analysis Scripts

### Core Analysis

#### `PORTAL.py`
Main analysis classes for processing screen data and calculating genetic interactions.

**Classes:**
- `PORTAL`: Single-guide screen analysis
- `DualGuidePORTAL`: Dual-guide screen analysis

**Usage:**
```python
from PORTAL import DualGuidePORTAL

# Load UMI count data and run complete analysis pipeline
screen = DualGuidePORTAL.from_csv(
    'umi_counts.csv',
    input_rep_data='library_representation.csv',
    sample_columns=['sample', 'rep']
)

# Run standard analysis (QC, phenotypes, GI, clustering)
screen.run_standard_analysis(
    min_thres=0.05,              # Filter overrepresented lineages
    control_prefix='non',         # Control guide prefix
    pseudocount=10,              # Log transformation
    gi_kind='per_gene_quadratic' # GI calculation method
)

# Access results
reporter_gi_matrix = screen.reporter_GI
reporter_phenotypes = screen.reporter_guide_matrix
identity_phenotypes = screen.identity_guide_matrix
```

**Input format (CSV):**
- `p1_identity`, `p2_identity`: Guide protospacer sequences
- `barcode_mapped`: Clonal barcode
- `UMI_identity`: UMI count for identity transcript
- `UMI_reporter`: UMI count for reporter transcript
- `sample`: Sample identifier
- `rep`: Replicate identifier

### Statistical Analysis

#### `cluster_recall.py`
Neighborhood enrichment analysis for recalling known complexes and transcriptional programs from hierarchical clustering.

**Usage:**
```python
from cluster_recall import recall_and_process_multiple_gene_sets_jointly

# Define gene sets (complexes, transcriptional clusters)
gene_sets = {
    'SAGA_complex': ['KAT2A', 'SUPT3H', 'TAF5L', ...],
    'Mediator_complex': ['MED1', 'MED12', 'MED14', ...],
}

# Neighborhood enrichment with permutation testing
results = recall_and_process_multiple_gene_sets_jointly(
    data_matrix=screen.reporter_GI,
    hierarchical_order=screen.reporter_order,
    gene_sets_dict=gene_sets,
    n_permutations=1000,
    p_value_threshold=0.05,
    window_size=10
)
```

#### `fast_mwu_analysis.py`
Fast implementation of Mann-Whitney U tests for single-guide pilot screen statistical testing. Speeds up guide-level effect testing by efficiently comparing each guide against all control guides using Numba parallelization.

**Usage:**
```python
from fast_mwu_analysis import run_mwu_analysis

# Statistical testing
results = run_mwu_analysis(
    pert_df=screen.filtered_umi_df,  # Perturbation data
    control_df=control_data,          # Control data
    group_cols=['guide'],             # Grouping columns
    outcomes=['reporter_residual', 'identity_resid']
)
```

#### `gsea_plots.py`
Modified version of gseapy plotting functions for validating reporter phenotypes against transcription factor motif signatures. Custom modifications: linewidth control and dot color scaling (vmin set to 0.05 FDR).

#### Perturb-seq Comparison

Comparison to Perturb-seq data in the pilot screen uses the `Perturbseq_GI` package:
https://github.com/thomasmaxwellnorman/Perturbseq_GI

#### `sqrt_pcp.py`
Implementation of square-root Principal Component Pursuit (√PCP) for genetic interaction matrix recovery from sparse measurements, with handling of missing values (NaNs) (Zhang et al., NeurIPS 2021).

**Usage:**
```python
from sqrt_pcp import sqrt_pcp

# Reconstruct GI matrix from downsampled data
result = sqrt_pcp(
    gi_matrix_sparse,
    lambda_param=None,  # Uses default: 1/√n1
    mu=None             # Uses default: √(n2/2)
)
L = result['L']  # Low-rank component
S = result['S']  # Sparse component
```

### Visualization

#### `PORTAL_viz.py`
Visualization functions for genetic interaction heatmaps and network diagrams.

**Usage:**
```python
from PORTAL_viz import plot_split_symmetric_matrix

# Plot split heatmap (upper: GI scores, lower: phenotypes)
fig, ax = plot_split_symmetric_matrix(
    screen=screen,
    cluster_annotations=cluster_results
)
```

#### `umap_viz.py`
UMAP network visualization of genetic interactions overlaid on transcriptional phenotype space.

**Usage:**
```python
from umap_viz import calculate_cluster_centers_density, add_complex_edges_curved

# Calculate cluster centers
cluster_centers = calculate_cluster_centers_density(
    umap_coords, cluster_labels
)

# Add curved edges between clusters on UMAP
edges = add_complex_edges_curved(
    ax, cluster_interaction_matrix, cluster_centers,
    threshold_quantile_positive=0.975,
    threshold_quantile_negative=0.025
)
```

### Data Processing

Processing notebooks in the `data_processing/` subdirectory handle FASTQ file processing to produce UMI count matrices. The modules `process_files.py` and `parallel_process_files.py` are imported within these notebooks.

## Analysis Notebooks

### Main Analysis (reproduce manuscript figures)

- **`Fig1_pilot_screen.ipynb`**: Pilot screen analysis and validation
- **`Fig3_cloning_viral_representations.ipynb`**: Library representation analysis
- **`Fig3_pacbio_per_base_identity_and_SV.ipynb`**: PacBio sequencing QC
- **`Fig4_5_GI_puro_screen.ipynb`**: Genetic interaction map analysis
- **`Fig6_downsampling.ipynb`**: Downsampling and √PCP recovery analysis
- **`Fig6_single_cell_screen.ipynb`**: Single-cell screen analysis

### Data Processing (in `data_processing/` subdirectory)

Notebooks for processing raw FASTQ files to UMI count matrices:
- `lentiviral_library_process_reads.ipynb`
- `pilot_screen_process_reads.ipynb` and `pilot_screen_collapse_to_UMI.ipynb`
- `puro_GI_process_reads_round1.ipynb`, `puro_GI_process_reads_round2.ipynb`, and `puro_GI_collapse_to_UMI.ipynb`
- `single-cell_process_reads.ipynb` and `single-cell_collapse_to_UMI.ipynb`

## Reproducing Manuscript Results

### Data Availability
- Raw sequencing data: [SRA: to be deposited]
- Processed UMI count matrices: [Zenodo: to be deposited]

### Reproduce Figures

Figures can be reproduced using the corresponding analysis notebooks listed above. Each notebook contains the complete analysis pipeline for its associated figure.

**Computational requirements:**
- Most analyses: <1 hour (standard workstation with 64 GB RAM)
- Downsampling analysis (Fig6): >10 hours (due to multiple subsampling iterations)
