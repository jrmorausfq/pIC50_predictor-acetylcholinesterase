import os
import subprocess as sp
import shutil
import time
import glob
import pandas as pd
import numpy as np
from openbabel import pybel
import weka.core.jvm as jvm

# Función para generar un SDF 3D a partir de un archivo de SMILES,
# guardándolo directamente en "ToMoCoMD/chemical_datasets/"
def create3DSDF_from_smiles(input_smiles):
    """
    Lee un archivo de SMILES y genera un SDF 3D con hidrógenos y optimización UFF.
    Se guarda en ToMoCoMD/chemical_datasets/ con el nombre "molecules3d.sdf".
    """
    print("Generando SDF 3D a partir de SMILES...")
    try:
        molecules = list(pybel.readfile("smi", input_smiles))
    except Exception as e:
        print(f"Error al leer el archivo de SMILES: {e}")
        exit()
    
    tomocomd_path = os.path.abspath("ToMoCoMD")
    chemical_datasets_path = os.path.join(tomocomd_path, "chemical_datasets")
    if not os.path.exists(chemical_datasets_path):
        os.makedirs(chemical_datasets_path)
    
    output_sdf = os.path.join(chemical_datasets_path, "molecules3d.sdf")
    outfile = pybel.Outputfile("sdf", output_sdf, overwrite=True)
    
    for mol in molecules:
        mol.addh()
        mol.make3D(forcefield='uff', steps=50)
        mol.localopt(forcefield='uff', steps=2000)
        outfile.write(mol)
    outfile.close()
    print(f"Archivo SDF 3D generado: {output_sdf}")
    return output_sdf

# Función para ejecutar ToMoCoMD y calcular descriptores
def run_tomocomd(sdf_file):
    """
    Ejecuta el jar de ToMoCoMD utilizando el SDF ubicado en "ToMoCoMD/chemical_datasets/".
    Luego busca recursivamente el archivo CSV generado con los descriptores.
    """
    print("Ejecutando ToMoCoMD para calcular descriptores...")
    tomocomd_path = os.path.abspath("ToMoCoMD")
    jar_file = os.path.join(tomocomd_path, "ToMoCoMD-CARDD_CLI.jar")
    if not os.path.exists(jar_file):
        print("No se encontró 'ToMoCoMD-CARDD_CLI.jar' en la carpeta 'ToMoCoMD'.")
        exit()
    
    cmd = f'cd {tomocomd_path}; java -jar ToMoCoMD-CARDD_CLI.jar'
    status, output = sp.getstatusoutput(cmd)
    print("Salida de ToMoCoMD:")
    print(output)
    
    # Buscar recursivamente el archivo CSV que contenga "user_specified_headings" en su nombre
    search_path = os.path.join(tomocomd_path, "Calculation")
    csv_files = glob.glob(os.path.join(search_path, "**", "*user_specified_headings*.csv"), recursive=True)
    print("Archivos encontrados:", csv_files)
    if csv_files:
        descriptors_csv = csv_files[0]
        print(f"Se encontró el archivo de descriptores: {descriptors_csv}")
        return descriptors_csv
    else:
        print("No se encontró el archivo de descriptores. Verifica la ejecución de ToMoCoMD.")
        exit()

# Función para cargar el CSV de descriptores (archivo test)
def load_descriptors(descriptors_csv):
    """
    Carga el CSV con los descriptores y elimina filas con datos faltantes.
    Si existe la columna de nombres (por ejemplo, 'molecules'), se elimina.
    """
    print("Cargando descriptores desde CSV...")
    try:
        df = pd.read_csv(descriptors_csv)
        df = df.dropna()
        if 'molecules' in df.columns:
            df = df.drop('molecules', axis=1)
    except Exception as e:
        print(f"Error al leer el archivo de descriptores: {e}")
        exit()
    print("Descriptores cargados con éxito.")
    return df

# Función para evaluar el dominio de aplicación usando AMBIT y calcular consenso
def evaluate_applicability_domain(training_csv, test_csv, model_name, ambit_path):
    """
    Lee los CSV de training y test y aplica 4 técnicas de dominio de aplicación.
    - En training se elimina la última columna (pIC50 experimental) para el análisis,
      pero se conserva el header original completo.
    - En test, si hay una columna extra (nombres de moléculas), se elimina la primera.
    Para el análisis se renombran los DataFrames a encabezados estándar (Desc0, Desc1, …).
    Se copian los archivos preparados a la carpeta Ambit para que el jar los encuentre.
    Se ejecutan los modos: DENSITY, EUCLIDEAN, CITYBLOCK y RANGE, con parámetros extra cuando corresponda.
    Se genera un CSV para cada modo.
    Se calcula el consenso: se consideran en dominio aquellas moléculas cuya suma de flags es menor a 2.
    Finalmente, se reestablecen los encabezados originales (del training) en el archivo filtrado para que
    coincidan con el modelo. Se añade temporalmente una columna dummy (con valor 0) para el target si no existe.
    Retorna una tupla (CSV filtrado, target_header).
    """
    print("Evaluando dominio de aplicación usando AMBIT...")
    # Leer el training original para obtener encabezados y target
    try:
        training_orig = pd.read_csv(training_csv)
    except Exception as e:
        print(f"Error al leer el archivo de training original: {e}")
        exit()
    target_header = training_orig.columns[-1]  # Última columna: pIC50 experimental
    originalHeaders = list(training_orig.columns[:-1])  # Solo descriptores
    
    try:
        training = pd.read_csv(training_csv)
        test = pd.read_csv(test_csv)
    except Exception as e:
        print(f"Error al cargar los CSV: {e}")
        exit()
    
    # En training: eliminar la última columna (pIC50 experimental)
    training = training.iloc[:, :-1]
    # En test: si tiene una columna extra (nombres), eliminar la primera
    if test.shape[1] == training.shape[1] + 1:
        test = test.iloc[:, 1:]
    
    if training.shape[1] != test.shape[1]:
        print("Error: El número de descriptores no coincide entre training y test.")
        exit()
    
    num_desc = training.shape[1]
    # Encabezados estandarizados para AMBIT
    standardHeaders = [f'Desc{j}' for j in range(num_desc)]
    standardHeaderString = ",".join(standardHeaders)
    print("Encabezados estandarizados para AMBIT:", standardHeaders)
    
    training_std = training.copy()
    test_std = test.copy()
    training_std.columns = standardHeaders
    test_std.columns = standardHeaders
    
    # Guardar archivos intermedios para AMBIT
    training_ambit_csv = training_csv.replace('.csv', '_AmbitInput.csv')
    test_ambit_csv = test_csv.replace('.csv', '_AmbitInput.csv')
    training_std.to_csv(training_ambit_csv, index=False)
    test_std.to_csv(test_ambit_csv, index=False)
    print(f"Archivos intermedios generados: {training_ambit_csv} y {test_ambit_csv}")
    
    # Copiar los archivos intermedios a la carpeta Ambit
    for file in [training_ambit_csv, test_ambit_csv]:
        dest = os.path.join(ambit_path, os.path.basename(file))
        shutil.copy(file, dest)
        print(f"Copiado {file} a {dest}")
    
    # Definir modos y parámetros extra
    techniques = {
        "DENSITY": {"mode": "_modeDENSITY", "extra": "-r 0.9"},
        "EUCLIDEAN": {"mode": "_modeEUCLIDEAN", "extra": ""},
        "CITYBLOCK": {"mode": "_modeCITYBLOCK", "extra": ""},
        "RANGE": {"mode": "_modeRANGE", "extra": "-r 0.9"}
    }
    
    # Ejecutar cada técnica con el jar de AMBIT
    for tech_name, params in techniques.items():
        command = (
            f'cd {ambit_path}; java -jar example-ambit-appdomain-jar-with-dependencies.jar '
            f'-m {params["mode"]} -t {os.path.basename(training_ambit_csv)} '
            f'-s {os.path.basename(test_ambit_csv)} -f {standardHeaderString} {params["extra"]} '
            f'-o {model_name}_{tech_name}.csv'
        )
        status, output = sp.getstatusoutput(command)
        print(f"AMBIT ({tech_name}): {output}")
    
    # Leer las salidas y extraer la bandera de dominio (se asume que la penúltima columna es la bandera)
    domain_flags = []
    for tech_name in techniques.keys():
        csv_file = os.path.join(ambit_path, f'{model_name}_{tech_name}.csv')
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip')
        except Exception as e:
            print(f"Error al leer {csv_file}: {e}")
            exit()
        if df.shape[1] >= 2:
            flag = df.iloc[:, -2]
            domain_flags.append(flag)
        else:
            print(f"El archivo {csv_file} no tiene el formato esperado.")
            exit()
    
    consensus = pd.concat(domain_flags, axis=1)
    consensus['Domain_Count'] = consensus.sum(axis=1)
    in_domain = consensus['Domain_Count'] < 2  # Al menos 3 de 4 deben aprobar
    test_in_domain = test_std[in_domain]
    
    # Reestablecer los encabezados originales (del training) para el test filtrado
    test_in_domain.columns = originalHeaders
    # Agregar la columna target (dummy con 0) si no existe
    if target_header not in test_in_domain.columns:
        test_in_domain = test_in_domain.copy()  # Para evitar SettingWithCopyWarning
        test_in_domain.loc[:, target_header] = 0
    
    filtered_csv = test_csv.replace('.csv', '_inDomain.csv')
    test_in_domain.to_csv(filtered_csv, index=False)
    print(f"Aplicabilidad dominio: {filtered_csv} con {len(test_in_domain)} moléculas en dominio de un total de {len(test_std)}")
    return filtered_csv, target_header

# Función para predecir pIC50 usando el modelo Weka
def predict_pIC50(test_csv, model_file, output_csv, target_header):
    """
    Carga el CSV de test (filtrado por dominio, con encabezados originales y columna dummy para el target),
    establece el índice de clase (última columna) y para cada instancia aplica classify_instance para predecir el pIC50.
    Guarda los resultados en un archivo CSV.
    """
    print("Iniciando predicción con Weka...")
    # Iniciar la JVM sin verificar si ya está iniciada
    jvm.start(system_cp=True, packages=True)
    
    from weka.core.converters import Loader
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(test_csv)
    data.class_index = data.num_attributes - 1
    print(f"Número de instancias de test: {data.num_instances}")
    
    try:
        tupleModel = weka.core.serialization.SerializationHelper.read(model_file)
    except Exception as e:
        from weka.classifiers import Classifier
        try:
            tupleModel = Classifier.deserialize(model_file)
        except Exception as e2:
            print(f"Error al cargar el modelo: {e2}")
            jvm.stop()
            exit()
    if isinstance(tupleModel, (list, tuple)):
        classifier = tupleModel[0]
    else:
        classifier = tupleModel

    predicted_values = []
    molecule_ids = []
    for i in range(int(data.num_instances)):
        inst = data.get_instance(i)
        pred = classifier.classify_instance(inst)
        predicted_values.append(pred)
        molecule_ids.append(i + 1)
    
    pred_df = pd.DataFrame({"Molecule_ID": molecule_ids, "pIC50_predicted": predicted_values})
    pred_df.to_csv(output_csv, index=False)
    print(f"Predicciones guardadas en: {output_csv}")
    jvm.stop()

# Función principal que orquesta todo el proceso
def main():
    print("=== Inicio del proceso ===")
    # Definir archivos de entrada/salida y modelo
    input_smiles = "to_predict.smiles"            # Archivo con la lista de SMILES a predecir
    model_file = "model.model"                      # Modelo Weka pre-entrenado para pIC50
    training_csv = "model_training.csv"             # CSV de training (12 columnas: 11 descriptores y pIC50)
    output_predictions_csv = "predicted_pIC50.csv"  # Archivo CSV final con las predicciones
    test_descriptors_csv = "test_descriptors.csv"   # Archivo donde se guardarán los descriptores calculados (test)
    
    # 1. Generar el SDF 3D en ToMoCoMD/chemical_datasets/
    sdf_file = create3DSDF_from_smiles(input_smiles)
    
    # 2. Ejecutar ToMoCoMD para calcular descriptores
    descriptors_csv = run_tomocomd(sdf_file)
    
    # 3. Cargar los descriptores y guardarlos en CSV para AMBIT
    descriptor_df = load_descriptors(descriptors_csv)
    descriptor_df.to_csv(test_descriptors_csv, index=False)
    print(f"Descriptores guardados en: {test_descriptors_csv}")
    
    # 4. Evaluar el dominio de aplicación usando AMBIT (4 técnicas y consenso)
    ambit_path = os.path.abspath("Ambit")  # Carpeta donde se encuentra el jar de AMBIT
    filtered_test_csv, target_header = evaluate_applicability_domain(training_csv, test_descriptors_csv, "Model", ambit_path)
    
    # 5. Predecir pIC50 usando el CSV filtrado (con encabezados originales y columna dummy para el target)
    predict_pIC50(filtered_test_csv, model_file, output_predictions_csv, target_header)
    
    print("Proceso de predicción completado.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    finish_time = round((time.time() - start_time) / 60.0, 3)
    print(f"\nEl proceso completo tomó {finish_time} minutos.")
