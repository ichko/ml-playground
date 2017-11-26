from sklearn.model_selection import train_test_split


def fast_score(model, data, y_col='y'):
    print('lets do this')

    y = data[y_col];
    X = data.drop(y_col, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, stratify=y)
    model.fit(X_train, y_train)
    print('train set score', model.score(X_train, y_train), 'test set score', model.score(X_test, y_test))

    return model