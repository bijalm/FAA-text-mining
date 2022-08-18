# Global Packages
import pandas as pd
import numpy as np
import pickle
import os

# Local files
from preprocessing import clean_narratives

def main():
    # User inputed csv file
    filename = input('Please enter the path of your narratives csv file: ')
    assert os.path.exists(filename), "I did not find the file at: "+str(filename)
    data = pd.read_csv(filename)
    
    # Preprocess narratives
    narratives = clean_narratives(data['NARRATIVE REDACTED'])

    # Load models
    conflict_model = pickle.load(open('models/conflict_model.pkl', 'rb'))
    phase_model = pickle.load(open('models/phase_model.pkl', 'rb'))
    hazard_model = pickle.load(open('models/hazard_model.pkl', 'rb'))

    # Load tf-idf vectorizers
    conflict_vectorizer = pickle.load(open('vectorizers/conflict_vectorizer.pkl','rb'))
    phase_vectorizer = pickle.load(open('vectorizers/phase_vectorizer.pkl','rb'))
    hazard_vectorizer = pickle.load(open('vectorizers/hazard_vectorizer.pkl','rb'))

    # Load multilabel binarizers
    phase_binarizer = pickle.load(open('binarizers/phase_binarizer.pkl','rb'))
    hazard_binarizer = pickle.load(open('binarizers/hazard_binarizer.pkl','rb'))

    # Narrative input for each model
    X = narratives
    conflict_tfidf = conflict_vectorizer.transform(X)
    phase_tfidf = phase_vectorizer.transform(X)
    hazard_tfidf = hazard_vectorizer.transform(X)

    # Run models for each input
    conflict_results = conflict_model.predict(conflict_tfidf)
    phase_results = phase_model.predict(phase_tfidf)
    hazard_results = hazard_model.predict(hazard_tfidf)

    # Inference of results
    conflict_infer = ['Yes' if conflict_results[i] == 0 else 'No' for i in range(len(conflict_results))]
    phases_infer = phase_binarizer.inverse_transform(phase_results)
    hazard_infer = hazard_binarizer.inverse_transform(hazard_results)

    # Create dataframe of results and export as an Excel sheet
    dict = {'Cleaned Narratives':narratives,'Conflict (Y/N)(EX TERRAIN)':conflict_infer,'Flight Phases':phases_infer,'AHCS':hazard_infer}
    df = pd.DataFrame(dict)
    df.insert(0,"Report Date",data['REPORT DATE'])
    df.to_excel('results/Results.xlsx')

    # Process completed message
    print('Classification is complete. Results are stored in the results folder.')

    # Ask user if they want to use new grouping models (ask Greg about the different labels that he provided)
    answer = input('Do you want to classify the narratives using a different label system? (yes or no): ').lower()
    while answer != 'yes' and answer != 'no':
        answer = input('Please enter yes or no: ').lower()
    
    if answer == 'yes':
        # Load models
        column1_label_model = pickle.load(open('models/column1_model.pkl', 'rb'))
        column2_label_model = pickle.load(open('models/column2_model.pkl', 'rb'))
        combined_columns_model = pickle.load(open('models/columns_1_and_2_model.pkl', 'rb'))

        # Load vectorizers
        column1_label_vectorizer = pickle.load(open('vectorizers/column1_vectorizer.pkl','rb'))
        column2_label_vectorizer = pickle.load(open('vectorizers/column_2_vectorizer.pkl','rb'))
        combined_columns_vectorizer = pickle.load(open('vectorizers/columns_1_and_2_vectorizer.pkl','rb'))

        # Narrative input for each model
        column1_tfidf = column1_label_vectorizer .transform(X)
        column2_tfidf = column2_label_vectorizer.transform(X)
        combined_columns_tfidf = combined_columns_vectorizer.transform(X)

        # Run models for each input
        column1_results = column1_label_model.predict(column1_tfidf)
        column2_results = column2_label_model.predict(column2_tfidf)
        combined_columns_results = combined_columns_model.predict(combined_columns_tfidf)

        # Load label maps
        column1_map = pickle.load(open('label-maps/Label_Map1.pkl','rb'))
        column2_map = pickle.load(open('label-maps/Label_Map3.pkl','rb'))
        combined_columns_map = pickle.load(open('label-maps/Label_Map2.pkl','rb'))
            
        # Inference of results
        column1_infer = []
        for result in column1_results:
            ret = []
            for i in np.where(result == 1)[0]:
                ret.append(column1_map[i])
            column1_infer.append(ret)
        
        column2_infer = []
        for result in column2_results:
            ret = []
            for i in np.where(result == 1)[0]:
                ret.append(column2_map[i])
            column2_infer.append(ret)
        
        combined_columns_infer = []
        for result in combined_columns_results:
            ret = []
            for i in np.where(result == 1)[0]:
                ret.append(combined_columns_map[i])
            combined_columns_infer.append(ret)

        # Create dataframe of results and export as an Excel sheet
        dict2 = {'Cleaned Narratives':narratives,'Column 1 Labels':column1_infer,'Column 2 Labels':column2_infer,'Columns 1 and 2 Combined Labels':combined_columns_infer}
        df2 = pd.DataFrame(dict2)
        df2.insert(0,"Report Date",data['REPORT DATE'])
        df2.to_excel('results/NewLabelResults.xlsx')

        # Process completed message
        print('New label classification is complete. Results are stored in the results folder.')

if __name__ == '__main__':
    main()