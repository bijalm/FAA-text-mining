# Global Packages
import pandas as pd
import numpy as np
import pickle
import os

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Local files
from preprocessing import clean_narratives, classifier

def main():
    # User inputed csv file
    filename = input('Please enter the path of your training data excel file: ')
    assert os.path.exists(filename), "I did not find the file at: "+str(filename)
    df = pd.read_excel(filename)
    
    # Preprocess narratives
    narratives = clean_narratives(df['NARRATIVE REDACTED'])
    
    # Extract AHCS labels from each data entry and create a column
    col1 = df.columns
    df["Classify"] = df.apply(lambda x: classifier(x, col1), axis=1)

    # Extract flight phases from each data entry and create a column
    miniDf = df[['NARRATIVE REDACTED', 'PHASE 1','PHASE 2', 'PHASE 3']]
    miniDf = miniDf.replace(np.nan, '', regex=True)
    miniDf['PHASE 1'] = 'Phase 1: ' + miniDf['PHASE 1']
    miniDf['PHASE 2'] = 'Phase 2: ' + miniDf['PHASE 2']
    miniDf['PHASE 3'] = 'Phase 3: ' + miniDf['PHASE 3']
    miniDf['PHASES'] = miniDf[['PHASE 1','PHASE 2', 'PHASE 3']].values.tolist()

    # Factorize columns
    df['EVENT TYPE (ADAB CLASSIFICATION)'] = pd.factorize(df['EVENT TYPE (ADAB CLASSIFICATION)'])[0]
    df['CONFLICT (Y/N) (EX TERRAIN)'] = pd.factorize(df['CONFLICT (Y/N) (EX TERRAIN)'])[0]
    df['PHASE 1'] = pd.factorize(df['PHASE 1'])[0]
    df['PHASE 2'] = pd.factorize(df['PHASE 2'])[0]
    df['PHASE 3'] = pd.factorize(df['PHASE 3'])[0]

    # Intialize TF-IDF vectorizer
    stop_words = stopwords.words('english')
    conflict_tfidf = TfidfVectorizer(min_df=0.05,max_df=0.95,stop_words=stop_words)
    hazard_tfidf = TfidfVectorizer(max_df=0.8, max_features=10000)
    phase_tfidf = TfidfVectorizer(max_df=0.8, max_features=10000)
    
    # Binarize the mutlilabels for hazards and flight phases
    hazard_multilabel_binarizer = MultiLabelBinarizer()
    hazard_multilabel_binarizer.fit(df['Classify'])
    phase_multilabel_binarizer = MultiLabelBinarizer()
    phase_multilabel_binarizer.fit(miniDf['PHASES'])

    # Create the targets
    conflict_y = df['CONFLICT (Y/N) (EX TERRAIN)']
    hazard_y = hazard_multilabel_binarizer.transform(df['Classify'])
    phase_y = phase_multilabel_binarizer.transform(miniDf['PHASES'])
    
    # Split data into training and testing data
    conflict_x_train, conflict_x_test, conflict_y_train, conflict_y_test = train_test_split(narratives, conflict_y, test_size=0.4)
    msss = MultilabelStratifiedShuffleSplit(n_splits=10, test_size=0.30, random_state=0)
    for train_index, test_index in msss.split(narratives, hazard_y):
        hazard_x_train, hazard_x_test = narratives[train_index], narratives[test_index]
        hazard_y_train, hazard_y_test = hazard_y[train_index], hazard_y[test_index]
    for train_index, test_index in msss.split(narratives, phase_y):
        phase_x_train, phase_x_test = narratives[train_index], narratives[test_index]
        phase_y_train, phase_y_test = phase_y[train_index], phase_y[test_index]

    # Fit data with vectorizer
    conflict_X_train = conflict_tfidf.fit_transform(conflict_x_train)
    conflict_X_test = conflict_tfidf.transform(conflict_x_test)
    hazard_X_train = hazard_tfidf.fit_transform(hazard_x_train)
    hazard_X_test = hazard_tfidf.transform(hazard_x_test)
    phase_X_train = phase_tfidf.fit_transform(phase_x_train)
    phase_X_test = phase_tfidf.transform(phase_x_test)

    # Intialize model and fit it with training data
    conflict_clf = LogisticRegression(max_iter=10000).fit(conflict_X_train, conflict_y_train)
    hazard_SVC = LinearSVC()
    hazard_clf = OneVsRestClassifier(hazard_SVC)
    hazard_clf.fit(hazard_X_train, hazard_y_train)
    phase_SVC = LinearSVC()
    phase_clf = OneVsRestClassifier(phase_SVC)
    phase_clf.fit(phase_X_train, phase_y_train)
        
    conflict_y_pred = conflict_clf.predict(conflict_X_test)
    print('Accuracy:',accuracy_score(conflict_y_test, conflict_y_pred))
    print(classification_report(conflict_y_test, conflict_y_pred))
            
    hazard_y_pred = hazard_clf.predict(hazard_X_test)
    print('Accuracy:',accuracy_score(hazard_y_test, hazard_y_pred))
    print(classification_report(hazard_y_test, hazard_y_pred))

    phase_y_pred = phase_clf.predict(phase_X_test)
    print('Accuracy:',accuracy_score(phase_y_test, phase_y_pred))
    print(classification_report(phase_y_test, phase_y_pred))

    # Save model, vectorizer, and binarizer
    pickle.dump(conflict_clf, open('models/conflict_model.pkl', 'wb'))
    pickle.dump(conflict_tfidf, open('vectorizers/conflict_vectorizer.pkl', 'wb'))
    pickle.dump(hazard_clf, open('models/hazard_model.pkl', 'wb'))
    pickle.dump(hazard_tfidf, open('vectorizers/hazard_vectorizer.pkl', 'wb'))
    pickle.dump(hazard_multilabel_binarizer, open('binarizers/hazard_binarizer.pkl', 'wb'))
    pickle.dump(phase_clf, open('models/phase_model.pkl', 'wb'))
    pickle.dump(phase_tfidf, open('vectorizers/phase_vectorizer.pkl', 'wb'))
    pickle.dump(phase_multilabel_binarizer, open('binarizers/phase_binarizer.pkl', 'wb'))

    # Printout of completion
    print('The models have been successfully trained.')
    print('Models have been saved under the models folder.')
    print('Tf-idf vectorizers have been saved under the vectorizers folder.')
    print('Multi-label binarizers have been saved under the binarizers folder')

if __name__ == '__main__':
    main()