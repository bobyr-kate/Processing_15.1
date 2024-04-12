import re
import multiprocessing
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem, Descriptors
import math
import logging
import time
from functools import wraps

my_logger = logging.getLogger('my_logger')

# Log settings
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s => %(name)s => %(levelname)s => %(message)s => %(filename)s',
    datefmt = '%d.%m.%Y %H:%M:%S',
    filename = '15_1_log',
    filemode = 'w'
)

class MoleculeProcessingException(Exception):
    pass

# new function for counting time 
def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {round(execution_time, 3)} seconds")
        my_logger.info(f"Execution time of {func.__name__}: {round(execution_time, 3)} seconds")
        return result
    return wrapper


# New function for processing errors catching
def molFromSmiles_f(column_name, chunk_df: pd.DataFrame,):
    for index, row in chunk_df.iterrows():
        try:
            mol = AllChem.MolFromSmiles(row[column_name])
            chunk_df.at[index, 'mol'] = mol
        except TypeError as e:
            my_logger.error(f"Mol processing TypeError: {e}")
            continue  
    return chunk_df

# New function for processing errors catching
def mol_props_to_compute_f(mol_props_to_compute, mol_props_funcs, chunk_df: pd.DataFrame,):
    for index, row in chunk_df.iterrows():
        for prop in mol_props_to_compute:
            try:
                prop_meaning = mol_props_funcs[prop](row['mol'])
                chunk_df.at[index, f'{prop}'] = prop_meaning
            except TypeError as e:
                my_logger.error(f"Parameters processing TypeError: {e}")
                continue  
    return chunk_df

class MolecularPropertiesProcessor:

    def __init__(
            self,
            filename,
            output_file_name: str,
    ):
        
        self.mols_df = pd.read_csv(filename, encoding='utf-8')
        self.output_file_name = output_file_name
        self.smiles_col = self._column_finder("^clean_smiles$")
        # deleted finder "^Molecule Name$" column, because it is epsent in data


    def _column_finder(self, match_str):
        matcher = re.compile(match_str, re.IGNORECASE)
        column_to_find = next(filter(matcher.match, self.mols_df.columns))
        if not column_to_find:
            raise MoleculeProcessingException(f"No {match_str} column found in a dataframe")
        return column_to_find

    def _prepare_data(self):
        del self.mols_df['Unnamed: 0']
        self.mols_df = self.mols_df[
            [self.smiles_col]
            + list(self.mols_df.columns.difference([self.smiles_col]))
        ]
        self.mols_df.drop_duplicates(subset=self.smiles_col, inplace=True)
        self.mols_df

    def _compute_molecule_properties_chunk(
            self,
            chunk_df: pd.DataFrame,
    ) -> pd.DataFrame:
            """ Compute molecule properties for chunk dataframe """
            # use new function molFromSmiles_f for handle errors opportunity
            chunk_df = molFromSmiles_f(self.smiles_col, chunk_df)

            mol_props_funcs = {
                            "Molecular weight": lambda mol: Descriptors.MolWt(mol),
                            "TPSA": lambda mol: Descriptors.TPSA(mol),
                            "logP": lambda mol: Descriptors.MolLogP(mol),
                            "H Acceptors": lambda mol: Descriptors.NumHAcceptors(mol),
                            "H Donors": lambda mol: Descriptors.NumHDonors(mol),
                            "Ring Count": lambda mol: Descriptors.RingCount(mol),
                            "Lipinski pass": lambda mol: all([
                                Descriptors.MolWt(mol) < 500,
                                Descriptors.MolLogP(mol) < 5,
                                Descriptors.NumHDonors(mol) < 5,
                                Descriptors.NumHAcceptors(mol) < 10
                            ])
            }

            mol_props_to_compute = list(mol_props_funcs.keys())

            # use new function mol_props_to_compute_f for handle errors opportunity
            chunk_df = mol_props_to_compute_f(mol_props_to_compute, mol_props_funcs, chunk_df)

            chunk_df.drop(columns=["mol"], inplace=True)
            chunk_df.set_index(self.smiles_col, inplace=True)
            
            return chunk_df


    def _compute_molecule_properties(self) -> pd.DataFrame:
        """
        Compute molecule properties and fingerprints using RDKit
        in chunks
        """

        const_size_of_chunks = 1000
        max_amount_of_p = 4

        amount_of_chunk_df = math.ceil(len(self.mols_df) / const_size_of_chunks)

        if amount_of_chunk_df > max_amount_of_p:
            amount_of_chunk_df = max_amount_of_p
        
        list_of_chunks = np.array_split(self.mols_df, amount_of_chunk_df)

        pool = multiprocessing.Pool(processes=amount_of_chunk_df)
        p_df = pool.map(self._compute_molecule_properties_chunk, list_of_chunks)

        list_of_p = [p for p in p_df]
        result = pd.concat(list_of_p)
        return result


    def process_data(self):
        self._prepare_data()
        mol_properties_df = self._compute_molecule_properties()
        return mol_properties_df 
    
    # new functon for file saving
    def save_to_csv(self, mol_properties_df):
        mol_properties_df.to_csv(self.output_file_name)


@time_it
def main():
    
    mpp = MolecularPropertiesProcessor(
        filename = "HUGE_Data_Set.csv",
        output_file_name="result.csv",
    )

    mol_properties_df = mpp.process_data()
    mpp.save_to_csv(mol_properties_df)

if __name__ == '__main__':
    main()