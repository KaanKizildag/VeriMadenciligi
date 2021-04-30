import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

def verileriGetir():
    import pandas as pd
    veriler = pd.read_csv('iris.csv')

    veriler.columns = [
        'sepal length',
        'sepal width',
        'petal length',
        'petal width',
        'outcome'
    ]

    return veriler
def egitimTestVeriSeti():
    veriler = verileriGetir()
    y = veriler['outcome']
    x = veriler.drop(columns=['outcome'])

    return sklearn.model_selection.train_test_split(x, y)

def olceklenmisTestVeriSeti():
    veriler = verileriGetir()
    y = veriler['outcome']
    x = veriler.drop(columns=['outcome'])
    sc = StandardScaler()
    x_olcekli = sc.fit_transform(x)

    return sklearn.model_selection.train_test_split(x_olcekli, y)
