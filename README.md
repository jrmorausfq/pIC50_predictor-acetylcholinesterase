# pIC50_predictor-acetylcholinesterase
Python pipeline to predict pIC50 values from molecular structures provided as SMILES
# pIC50 Predictor

Python pipeline to predict pIC50 values from molecular structures provided as SMILES.

The workflow performs the following steps:

1. Reads molecules from a SMILES file.
2. Generates 3D SDF structures using Open Babel/Pybel.
3. Calculates molecular descriptors using ToMoCoMD-CARDD.
4. Evaluates the applicability domain using AMBIT.
5. Predicts pIC50 values using a pre-trained Weka model.

> Note: this repository contains the workflow code. External programs such as ToMoCoMD-CARDD, AMBIT, Java, Open Babel, and Weka must be installed or downloaded separately as described below.

---

## Recommended repository structure

```text
pIC50-predictor/
├── pIC50_predictor.py
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── ZENODO.md
├── to_predict.smiles              # input file with SMILES to predict
├── model.model                    # pre-trained Weka model
├── model_training.csv             # training descriptors + experimental pIC50
├── ToMoCoMD/
│   ├── README.md
│   ├── ToMoCoMD-CARDD_CLI.jar     # download manually
│   └── chemical_datasets/
├── Ambit/
│   ├── README.md
│   └── example-ambit-appdomain-jar-with-dependencies.jar
├── examples/
├── data/
├── models/
└── results/
```

The current script uses fixed file names and expects the following files in the repository root:

```text
to_predict.smiles
model.model
model_training.csv
```

It also expects the following paths:

```text
ToMoCoMD/ToMoCoMD-CARDD_CLI.jar
Ambit/example-ambit-appdomain-jar-with-dependencies.jar
```

---

## System requirements

Linux, macOS, or Windows with Anaconda/Miniconda is recommended.

### Required external software

| Software | Purpose in the workflow | How to obtain it |
|---|---|---|
| Python 3.9–3.11 | Runs the script | Official Python distribution or Conda |
| Java JDK/JRE | Runs `.jar` files for ToMoCoMD, AMBIT, and Weka | OpenJDK or equivalent Java distribution |
| Open Babel + Pybel | Converts SMILES to 3D SDF | Recommended installation through Conda |
| Weka / python-weka-wrapper3 | Loads the `.model` file and performs predictions | Install with `pip` |
| ToMoCoMD-CARDD | Calculates molecular descriptors | Download from the official website: https://tomocomd.com/ |
| AMBIT Applicability Domain | Evaluates the applicability domain | Place the required `.jar` file inside the `Ambit/` folder |

### Python packages

```text
numpy
pandas
openbabel / pybel
python-weka-wrapper3
```

---

## Recommended installation with Conda

Create the environment:

```bash
conda env create -f environment.yml
conda activate pic50-predictor
```

Check Java:

```bash
java -version
```

Check Open Babel:

```bash
obabel -V
```

Check the Python dependencies:

```bash
python -c "import pandas, numpy; from openbabel import pybel; import weka.core.jvm as jvm; print('OK')"
```

---

## Alternative manual installation

```bash
conda create -n pic50-predictor python=3.10 -y
conda activate pic50-predictor
conda install -c conda-forge openbabel pandas numpy openjdk=11 -y
pip install python-weka-wrapper3
```

---

## ToMoCoMD installation

ToMoCoMD-CARDD must be downloaded from the official website:

https://tomocomd.com/

After downloading it:

1. Create the `ToMoCoMD/` folder in the repository root.
2. Copy the following file into that folder:

```text
ToMoCoMD-CARDD_CLI.jar
```

3. Confirm that the path is exactly:

```text
ToMoCoMD/ToMoCoMD-CARDD_CLI.jar
```

4. The script will automatically generate the 3D SDF file at:

```text
ToMoCoMD/chemical_datasets/molecules3d.sdf
```

> Important: third-party `.jar` files should not be uploaded to GitHub unless their license explicitly allows redistribution. Instead, document how to download them from the official source.

---

## AMBIT installation

The script expects the following file:

```text
Ambit/example-ambit-appdomain-jar-with-dependencies.jar
```

Create the `Ambit/` folder and place the required `.jar` file inside it.

AMBIT is used to evaluate the applicability domain using four techniques:

- DENSITY
- EUCLIDEAN
- CITYBLOCK
- RANGE

The consensus criterion considers a molecule within the applicability domain when at least 3 out of the 4 techniques accept it.

---

## Input files

### 1. `to_predict.smiles`

File containing the molecules to be predicted in SMILES format. Example:

```text
CCO ethanol
CC(=O)O acetic_acid
```

### 2. `model_training.csv`

CSV file used as reference for the applicability-domain analysis. It must contain the descriptors used to train the model and, in the last column, the experimental pIC50 value.

Example structure:

```text
Descriptor1,Descriptor2,...,DescriptorN,pIC50
0.123,1.456,...,3.210,6.45
```

### 3. `model.model`

Pre-trained Weka model used for pIC50 prediction.

---

## Running the workflow

From the repository root:

```bash
python pIC50_predictor.py
```

During execution, the script generates intermediate and final files, including:

```text
ToMoCoMD/chemical_datasets/molecules3d.sdf
test_descriptors.csv
test_descriptors_inDomain.csv
predicted_pIC50.csv
```

The expected final output file is:

```text
predicted_pIC50.csv
```

with columns similar to:

```text
Molecule_ID,pIC50_predicted
1,6.123
2,5.876
```

---

## Output files

| File | Description |
|---|---|
| `molecules3d.sdf` | 3D structures generated from SMILES |
| `test_descriptors.csv` | Molecular descriptors calculated with ToMoCoMD |
| `test_descriptors_inDomain.csv` | Molecules retained within the applicability domain |
| `predicted_pIC50.csv` | Final pIC50 predictions |

---

## Reproducibility

To ensure reproducibility, report the following in the associated article or supplementary material:

- Python version;
- Java version;
- Open Babel version;
- ToMoCoMD-CARDD version;
- AMBIT version;
- Weka or `python-weka-wrapper3` version;
- ToMoCoMD download date;
- exact Weka model file used;
- repository version or Zenodo DOI.

---

## Limitations

- The current script uses fixed paths and file names.
- The Weka model must be compatible with the descriptors and headers in `model_training.csv`.
- The descriptor file generated by ToMoCoMD must contain the string `user_specified_headings` in its file name, because the script searches for that pattern automatically.
- Predictions are performed only for molecules retained within the applicability domain.
- External binaries such as ToMoCoMD and AMBIT are not included in this repository.

---

## Citation

If you use this code, please cite the archived Zenodo version. After creating the DOI, update this section:

```text
Author(s). pIC50 Predictor. Version 1.0.0. Zenodo. https://doi.org/XX.XXXX/zenodo.XXXXXXX
```

You may also use GitHub's **Cite this repository** button if the `CITATION.cff` file is present.

---

## License

This code is distributed under the MIT License. See the `LICENSE` file.

This license applies only to the code in this repository and does not apply to third-party software such as ToMoCoMD-CARDD, AMBIT, Weka, Open Babel, or Java.
