# ARFF Preparation for CLUS - PCTs Dimensionality Reduction

This repository contains scripts and utilities for preparing ARFF (Attribute-Relation File Format) files from deep learning feature extractions for use with [CLUS+](https://github.com/knowledge-technologies/clus), a predictive clustering machine learning system.

This work is based on the research presented in:

> Sintija Stevanoska, Marjan Stoimchev, Jurica Levatić, and Sašo Džeroski. **"Dimensionality Reduction for Efficient Semi-supervised Learning from Remote Sensing Images."** In *Machine Learning and Principles and Practice of Knowledge Discovery in Databases: International Workshops of ECML PKDD 2024, Vilnius, Lithuania, September 9--13, 2024, Revised Selected Papers, Part III, Communications in Computer and Information Science, vol. 2560*. Springer, Cham, 2026 (In press).

## About CLUS+

[CLUS+](https://github.com/knowledge-technologies/clus) is a Java-based decision tree and rule induction software developed by the Department of Knowledge Technologies at Jožef Stefan Institute. It implements a predictive clustering framework that handles:

- **Hierarchical multi-label classification (MLC)**
- **Hierarchical multi-target regression**
- **Semi-supervised learning (SSL)**
- **Ensemble methods** (Random Forest, Extra Trees)
- **Predictive Clustering Trees (PCT)**

This repository prepares data specifically for semi-supervised and supervised learning with PCTs and Random Forests on multi-label and multi-class image classification tasks.

## Repository Structure

```
mlc/
├── clus_utils/              # Core utilities
│   ├── CreateArff.py        # Main class for ARFF generation
│   └── __init__.py
│
├── configs/                 # Dataset configurations
│   ├── config.py            # Dataset definitions (AID, UCM, RESISC45, etc.)
│   └── __init__.py
│
├── config.py                # Legacy land cover class mappings
├── utils.py                 # Helper functions (paths, messages, sorting)
├── other_imports.py         # Common imports (pandas, numpy, tqdm)
│
├── create_arff.py           # Legacy ARFF creation script
├── create_settings.py       # Legacy settings generation
├── run_create_arff.py       # Main entry point for ARFF generation
│
├── run_clus_ssl.sh          # Universal CLUS execution script
├── run_clus_ssl_pct.sh      # PCT method execution
├── run_clus_ssl_forest.sh   # Random Forest execution
├── run_clus_ssl_mcc_pct.sh  # Multi-class PCT
├── run_clus_ssl_mcc_forest.sh  # Multi-class Random Forest
├── run_mcc_pct_pca.sh       # MCC with PCA preprocessing
├── run_mcc_forest_pca.sh    # MCC Random Forest with PCA
└── run_clus_multi_dataset.sh   # Multi-dataset execution (legacy)
```

## Prerequisites

### 1. CLUS+ Setup

First, build and install CLUS+:

```bash
# Clone CLUS+ repository
git clone https://github.com/knowledge-technologies/clus.git

# Navigate and build
cd clus/ClusProject
mvn clean package

# The executable JAR will be at: target/clus-<version>-deps.jar
```

### 2. Directory Structure

The expected directory structure for this repository relative to CLUS is:

```
parent_directory/
├── Clus_plus/
│   └── Clus.jar              # CLUS+ executable (renamed from clus-<version>-deps.jar)
│
└── mlc/                      # This repository
    ├── clus_utils/
    ├── configs/
    ├── FEATURES/             # Input features (not in git)
    │   ├── original/
    │   └── pca/
    ├── Arfs_mlc/             # Generated ARFF outputs (not in git)
    ├── Arfs_mcc/             # Generated ARFF outputs (not in git)
    └── RESULTS/              # CLUS execution results (not in git)
```

**Important:** Place your built CLUS JAR at `../Clus_plus/Clus.jar` relative to this repository.

### 3. Python Dependencies

```bash
pip install numpy pandas tqdm scikit-learn
```

### 4. Java Runtime

- Java 1.8 JDK or JRE (required to run CLUS+)

## Input Data Format

This pipeline expects extracted deep learning features stored as NumPy arrays:

```
FEATURES/original/
└── <dataset>_features/<model_type>/
    ├── train_features_<model>.npy    # Shape: (n_train_samples, n_features)
    ├── train_targets.npy              # Shape: (n_train_samples, n_classes) for MLC
    ├── val_features_<model>.npy
    ├── val_targets.npy
    ├── test_features_<model>.npy
    └── test_targets.npy
```

**Example:**
```
FEATURES/original/AID_mlc_features/effnet_b2/
├── train_features_effnet_b2.npy    # (8000, 1408)
├── train_targets.npy                # (8000, 17)
├── val_features_effnet_b2.npy
├── val_targets.npy
├── test_features_effnet_b2.npy
└── test_targets.npy
```

## Supported Configurations

### Datasets

**Multi-Label Classification (MLC):**
- `AID_mlc` (17 classes)
- `UCM_mlc` (17 classes)
- `DFC_15` (8 classes)
- `Ankara` (29 classes)
- `MLRSNet_5_percent` (60 classes)

**Multi-Class Classification (MCC):**
- `AID_mcc` (30 classes)
- `UCM_mcc` (21 classes)
- `RESISC45` (45 classes)
- `OPTIMAL-31` (31 classes)
- `RSSCN7` (7 classes)
- `Ankara` (29 classes)

### Deep Learning Models

- VGG16, VGG19
- ResNet34, ResNet50, ResNet152
- EfficientNet B0, B1, B2

### CLUS Methods

- **PCT** (Predictive Clustering Trees)
- **RForest** (Random Forest)

### Learning Settings

- **SSL** (Semi-Supervised Learning)
- **SL** (Supervised Learning)

### Labeled Data Percentages

- 1%, 5%, 10%, 25%

## Usage

### Step 1: Generate ARFF Files

Edit `run_create_arff.py` to configure your experiment:

```python
model_types = ["effnet_b2"]
datasets = ["AID_mlc"]
methods = ["PCT", "RForest"]
learning_settings = ["sl", "ssl"]
seeds = ["0"]
learning_task = "mlc"  # or "mcc"
percentage_labeled_examples = ["25"]
apply_pca = False
```

Run the script:

```bash
python run_create_arff.py
```

This generates:
- `Arfs_mlc/PCT/AID/effnet_b2/seed_0/labeled_25/ssl/`
  - `train_features.arff`
  - `test_features.arff`
  - `features.s` (CLUS settings file)

### Step 2: Run CLUS

Execute CLUS with the generated files using one of the bash scripts:

```bash
# For multi-label PCT
bash run_clus_ssl_pct.sh Arfs_mlc

# For multi-label Random Forest
bash run_clus_ssl_forest.sh Arfs_mlc

# For multi-class PCT
bash run_clus_ssl_mcc_pct.sh Arfs_mcc

# For multi-class Random Forest
bash run_clus_ssl_mcc_forest.sh Arfs_mcc
```

The scripts will:
1. Iterate through all configured combinations (datasets, models, seeds, percentages)
2. Navigate to each ARFF directory
3. Execute CLUS: `java -jar Clus.jar -ssl features.s`
4. Output results to `features.out` and `log.txt`

### Step 3: Analyze Results

Results are generated in the same directory as the ARFF files:
- `features.out` - Contains predictions and evaluation metrics
- `log.txt` - Execution logs and errors

## Core Classes

### `CreateSettingsSSL` (clus_utils/CreateArff.py)

Main class for generating ARFF files and CLUS settings.

```python
cs = CreateSettingsSSL(
    output_dir="Arfs_mlc",           # Output directory name
    features_paths=[...],             # List of .npy feature files
    targets_paths=[...],              # List of .npy target files
    dataset="AID_mlc",                # Dataset name
    model_type="effnet_b2",           # Model type
    percentage_labeled="25",          # Percentage of labeled data
    method="PCT",                     # PCT or RForest
    learning_setting="ssl",           # ssl or sl
    learning_task="mlc",              # mlc or mcc
    apply_pca=False,                  # PCA preprocessing
    seed="0"                          # Random seed
)
cs.create()
```

**Key Methods:**
- `create()` - Main orchestration method
- `_create_arff_dense()` - Writes ARFF files
- `create_general_settings()` - Base CLUS configuration
- `create_settings_ssl_pct()` - Semi-supervised PCT settings
- `create_settings_ssl_rf()` - Semi-supervised Random Forest settings

## ARFF File Format

Generated ARFF files follow WEKA format:

```
@relation AID_mlc_effnet_b2

@attribute feature_1 numeric
@attribute feature_2 numeric
...
@attribute feature_1408 numeric

@attribute class_1 {0, 1}
@attribute class_2 {0, 1}
...
@attribute class_17 {0, 1}

@data
0.123,0.456,...,1,0,1,0,...
```

## CLUS Settings File (.s)

Example settings file structure:

```ini
[General]
RandomSeed = 0
Verbose = 1

[Data]
File = ../../../train_features.arff
TestSet = ../../../test_features.arff

[Attributes]
Target = 1409-1425          # Columns for 17 multi-label targets
Descriptive = 1-1408        # Feature columns

[SemiSupervised]
SemiSupervisedMethod = PCT
PercentageLabeled = 25
InternalFolds = 4
PruningWhenTuning = Yes

[Ensemble]
Iterations = 100            # For Random Forest
Threads = 8
```

## Configuration Examples

### Multi-Label Classification (MLC)

```python
# In configs/config.py
class AID_mlc(BaseConfig):
    extension = 'jpg'
    n_classes = 17              # Number of labels
    image_size = 600
    dataset_name = "AID"

    out_features = {
        "Vgg16": 3,
        "Vgg19": 5,
        "resnet34": 8,
        "effnet_b2": 11,
    }
```

### Multi-Class Classification (MCC)

```python
class AID_mcc(BaseConfig):
    extension = 'jpg'
    n_classes = 30              # Number of classes
    image_size = 600
    dataset_name = "AID"
```

## Execution Scripts Explained

### `run_clus_ssl_pct.sh`

Executes PCT method for multi-label classification:

```bash
bash run_clus_ssl_pct.sh Arfs_mlc
```

Loops through:
- Datasets: AID
- Methods: PCT
- Models: effnet_b2
- Seeds: 0, 1, 42
- Labeled percentages: 1%, 5%, 10%, 25%
- Learning: SSL

### `run_clus_ssl.sh`

Universal script supporting all configurations:

```bash
bash run_clus_ssl.sh
```

Edit the script to customize datasets, methods, and parameters.

## Troubleshooting

### Issue: "CLUS jar not found"

**Solution:** Ensure CLUS is at `../Clus_plus/Clus.jar` relative to this repository.

```bash
# Check the path
ls ../Clus_plus/Clus.jar
```

### Issue: "Feature files not found"

**Solution:** Verify your features are in the correct directory structure:

```bash
ls FEATURES/original/AID_mlc_features/effnet_b2/
```

### Issue: "Java heap space error"

**Solution:** The scripts use `-Xmx8G` for 8GB heap. Adjust in the bash scripts:

```bash
java -jar -Xmx16G ../../../../../../../../Clus_plus/Clus.jar -ssl features.s
```

### Issue: "Import errors"

**Solution:** Ensure you're running from the repository root and all dependencies are installed:

```bash
cd /path/to/mlc
pip install numpy pandas tqdm scikit-learn
python run_create_arff.py
```

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{Stevanoska2026_DimReductionSSL,
  author    = {Sintija Stevanoska and Marjan Stoimchev and Jurica Levati{\'c} and Sa{\v{s}}o D{\v{z}}eroski},
  title     = {Dimensionality Reduction for Efficient Semi-supervised Learning from Remote Sensing Images},
  booktitle = {Machine Learning and Principles and Practice of Knowledge Discovery in Databases:
               International Workshops of ECML PKDD 2024, Vilnius, Lithuania, September 9--13, 2024,
               Revised Selected Papers, Part~III, Communications in Computer and Information Science, vol.~2560},
  publisher = {Springer},
  address   = {Cham},
  year      = {2026},
  note      = {In press},
  keywords  = {myconf}
}
```

If you use CLUS+ in your research, please also cite:

```
Knowledge Technologies Department, Jožef Stefan Institute
CLUS+: https://github.com/knowledge-technologies/clus
```

## License

This repository contains scripts for preparing data for CLUS+. Refer to the [CLUS+ repository](https://github.com/knowledge-technologies/clus) for CLUS licensing information.

## Contact

For issues related to:
- **CLUS+ software:** See [CLUS+ GitHub Issues](https://github.com/knowledge-technologies/clus/issues)
- **This repository:** Open an issue on this repository

## References

- CLUS+ Repository: https://github.com/knowledge-technologies/clus
- WEKA ARFF Format: https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/
