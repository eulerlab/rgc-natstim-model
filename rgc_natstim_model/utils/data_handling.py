import pandas as pd
import numpy as np
import pickle

def convert_nested_dict_to_dataframe(nested_dict):
    """
    Convert a nested dictionary with numpy arrays to a pandas DataFrame,
    adding dataset_hashes and session_ids as columns.

    Parameters:
    nested_dict (dict): The nested dictionary with numpy arrays to convert.

    Returns:
    pd.DataFrame: The resulting DataFrame.
    """
    # Create a list of records
    records = []

    for dataset_hash, sessions in nested_dict.items():
        for session_id, session_data in sessions.items():
            if session_id == 'session_1_ventral2_20200701':
                continue
            # Extract all keys from session data
            keys = list(session_data.keys())
            keys.remove('roi_coords')
            # Extract the number of cells from the first numpy array (assuming all arrays have the same length)
            n_cells = session_data[keys[0]].shape[0]
            # Create individual records for each cell in all arrays
            for i in range(n_cells):
                record = {'dataset_hash': dataset_hash,
                          'session_id': session_id,
                          'neuron_id': '_'.join([''.join(session_data['date'].split('-')),  # date
                                                 str(session_data['exp_num']),
                                                 str(session_data['field_id']),
                                                 str(session_data['roi_ids'][i]),
                                                 's_5'
                                                 ])
                          }
                for key in keys:
                    value = session_data[key]
                    # If value is not a numpy array, tile it to match the number of cells
                    if not isinstance(value, np.ndarray):
                        value = np.tile(value, (n_cells, 1))
                    record[key] = value[i].squeeze()
                records.append(record)

    # Create the DataFrame from the list of records
    df = pd.DataFrame(records)
    df.set_index('neuron_id', drop=False, inplace=True)
    return df

def unPickle(filename):
    '''
    shorten 3 lines into 1 line
    '''
    with open(filename,'rb') as f:
        output = pickle.load(f)
    f.close()
    return output
    
def makePickle(filename,data):
    '''
    shorten 3 lines into 1 line
    '''
    with open(filename,'wb') as f:
        pickle.dump(data,f)
    f.close()
