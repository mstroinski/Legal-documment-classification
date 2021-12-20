from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def search_parameters(model, model_params, data, verbose=2):
    
    pipeline = Pipeline([('tfidf', TfidfVectorizer()),
                        ('model', model)])
    
    param_grid = {'tfidf__ngram_range': [(1, 1), (1, 2)]}
    for key, values in model_params.items():
        string = 'model__' + key
        param_grid[string] = values

    x_train, y_train = data
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=verbose)
    grid_search = grid_search.fit(x_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    return best_model