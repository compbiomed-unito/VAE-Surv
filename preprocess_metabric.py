import pandas as pd
import numpy as np

import sksurv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split

#pd.set_option('future.no_silent_downcasting', True)

def preprocess_clinic():
    # ------------------------------
    # Step 1. Define the mapping function.
    # ------------------------------
    def map_binary_and_ordinal(df):
        df = df.copy()
        # Map binary categorical features to 0/1.
        binary_mapping = {
            'type_of_breast_surgery': {'MASTECTOMY': 0, 'BREAST CONSERVING': 1},
            'er_status_measured_by_ihc': {'Negative': 0, 'Positve': 1},  # note the typo “Positve”
            'er_status': {'Negative': 0, 'Positive': 1},
            'her2_status': {'Negative': 0, 'Positive': 1},
            'inferred_menopausal_state': {'Post': 0, 'Pre': 1},
            'primary_tumor_laterality': {'Left': 0, 'Right': 1},
            'pr_status': {'Negative': 0, 'Positive': 1}
        }
        for col, mapping in binary_mapping.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)
        
        # Convert an ordinal feature (cellularity) to numeric.
        # We assume: Low < Moderate < High.
        if 'cellularity' in df.columns:
            df['cellularity'] = df['cellularity'].replace({'Low': 1, 'Moderate': 2, 'High': 3})
            
        return df
    
    # Wrap the mapping function in a FunctionTransformer.
    mapper_transformer = FunctionTransformer(map_binary_and_ordinal)
    
    # ------------------------------
    # Step 2. Define column groups.
    # ------------------------------
    binary_cols = [
        'type_of_breast_surgery', 'er_status_measured_by_ihc', 'er_status',
        'her2_status', 'inferred_menopausal_state', 'primary_tumor_laterality',
        'pr_status', 'chemotherapy', 'hormone_therapy', 'radio_therapy'
    ]
    
    continuous_cols = [
        'age_at_diagnosis', 'neoplasm_histologic_grade', 'lymph_nodes_examined_positive',
        'mutation_count', 'nottingham_prognostic_index', 'tumor_size', 'tumor_stage',
        'cellularity'  # after mapping, cellularity is numeric (ordinal).
    ]
    
    categorical_nominal_cols = [
        'pam50_+_claudin-low_subtype', 'her2_status_measured_by_snp6',
        'tumor_other_histologic_subtype', 'integrative_cluster',
        '3-gene_classifier_subtype', 'cancer_type_detailed'
    ]
    
    # ------------------------------
    # Step 3. Create sub-pipelines for each group.
    # ------------------------------
    # Continuous pipeline: impute missing values using the median then scale.
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Binary pipeline: impute missing values using the most frequent value.
    binary_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    
    # Categorical nominal pipeline: impute missing values using the mode then one-hot encode.
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # ------------------------------
    # Step 4. Create a ColumnTransformer.
    # ------------------------------
    column_transformer = ColumnTransformer(transformers=[
        ('num', num_pipeline, continuous_cols),
        ('bin', binary_pipeline, binary_cols),
        ('cat', cat_pipeline, categorical_nominal_cols)
    ])
    
    # ------------------------------
    # Step 5. Build the full preprocessing pipeline.
    # ------------------------------
    preprocessor = Pipeline(steps=[
        ('mapper', mapper_transformer),       # Map binary and ordinal columns.
        ('column_transformer', column_transformer)  # Apply the transformations.
    ])
    
    preprocessor.set_output(transform="pandas")

    return preprocessor


def preprocess(data_path):
    df = pd.read_csv(data_path, low_memory=False).set_index('patient_id')
    y = df[['overall_survival_months', 'overall_survival']]
    df = df.drop(['overall_survival_months', 'overall_survival'], axis=1)
    to_remove = ['death_from_cancer', 'cohort', 'oncotree_code', 'cancer_type']
    df = df.drop(to_remove, axis=1)
    
    
    # Convert survival labels to structured array format used by `sksurv`
    y_structured = np.array(
        [(bool(event), time) for event, time in zip(y['overall_survival'], y['overall_survival_months'])],
        dtype=[('event', 'bool'), ('time', 'float32')]
    )
    
    # separate clinical, gene_mutation and rna expression data
    df_gen = df.iloc[:,24:]
    gene_mut = ['_mut' in c for c in df_gen.columns]
    df_gene_mut = df_gen.iloc[:,gene_mut]
    df_gene_mut = df_gene_mut.map(lambda x: 0 if x == '0' else 1)
    
    df_clin = df.iloc[:,:24]
    df_clin = df_clin.replace({'Metaplastic':'Other'})
    
    df_gene_exp = df_gen.iloc[:,[not i for i in gene_mut]]
    
    
    # Train-test split
    df_clin_train, df_clin_test, df_gene_mut_train, df_gene_mut_test, df_gene_exp_train, df_gene_exp_test, y_train, y_test = train_test_split(
        df_clin, df_gene_mut, df_gene_exp, y_structured,
        test_size=0.2,         
        random_state=0,       
        stratify=y_structured['event']
    )

    
    preprocessor = preprocess_clinic()
    preprocessor.fit(df_clin_train)  # Imputation (and scaling) parameters are learned from df_clin_train
    
    df_clin_train_processed = preprocessor.transform(df_clin_train)
    df_clin_test_processed = preprocessor.transform(df_clin_test)
    
    df_final_train = pd.concat([df_clin_train_processed, df_gene_exp_train, df_gene_mut_train], axis=1)
    df_final_test = pd.concat([df_clin_test_processed, df_gene_exp_test, df_gene_mut_test], axis=1)

    mask = [item in pd.concat([df_gene_exp, df_gene_mut],axis=1) for item in df_final_train]
    
    return df_final_train, df_final_test, y_train, y_test, mask


