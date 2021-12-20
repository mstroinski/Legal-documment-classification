def read_data(file_path, read_all=False):
    import ast
    import pandas as pd 
    
    if read_all:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, usecols=[2,3])
    lst = []

    for labels_str in df['labels']:
        lst.append(ast.literal_eval(labels_str))

    df.drop('labels', inplace=True, axis=1) 
    df['labels'] = lst
    
    return df