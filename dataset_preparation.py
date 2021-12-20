def generate_labels(all_labels, wanted_labels):
    intersected = list(wanted_labels.intersection(all_labels))
    
    if len(intersected) > 1:
        return None 
    
    if not intersected:
        return 0
    
    return intersected[0]

def prepare_dataset(dataFrame_org, wanted_labels, reduce_size=True):
    dataFrame = dataFrame_org.copy()
    dataFrame['correct_labels'] = dataFrame.apply(lambda row : generate_labels(row['labels'], wanted_labels), axis = 1)
    dataFrame.dropna(inplace=True)
    dataFrame.drop('labels', inplace=True, axis=1)
    dataFrame['correct_labels'] = dataFrame['correct_labels'].astype(int)
    
    if reduce_size:
        # create dataframe without other ('0') class
        tmp_df = dataFrame.copy()
        tmp_df = tmp_df.drop(tmp_df.query('correct_labels==0').index)
        
        # find most common class without other ('0') other
        most_common_class = tmp_df.mode(numeric_only=True)['correct_labels'][0]
        
        # calculate by what factor other ('0') class should be reduced if we want it to be 1.5 x the most common class
        n_max = dataFrame[dataFrame['correct_labels'] == most_common_class]['correct_labels'].count()
        n_bad = dataFrame[dataFrame['correct_labels'] == 0]['correct_labels'].count()
        factor = 1 - (n_max / n_bad) * 1.5

        # randomly drop rows with other ('0') class 
        dataFrame = dataFrame.drop(dataFrame.query('correct_labels==0').sample(frac=factor).index)
        
    return dataFrame