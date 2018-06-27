# https://archive.ics.uci.edu/ml/datasets/Wine
import logging
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    # 1) Alcohol
    # 2) Malic acid
    # 3) Ash
    # 4) Alcalinity of ash
    # 5) Magnesium
    # 6) Total phenols
    # 7) Flavanoids
    # 8) Nonflavanoid phenols
    # 9) Proanthocyanins
    # 10)Color intensity
    # 11)Hue
    # 12)OD280 / OD315 of diluted wines
    # 13)Proline

    names = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoid_phenols',
             'nonflavanoid_phenols', 'proanthcynins', 'color_intensity', 'hue', 'od_ratio', 'proline']

    input_file = '../data/wine.data'
    logger.debug('reading data from %s' % input_file)
    df = pd.read_csv(input_file, names=names)
    logger.debug(df.shape)
    variables = df.columns.values
    logger.debug(variables)

    logger.debug('scores predicting using all other variables:')
    for target_column in variables:
        X = df.drop([target_column], axis=1).values
        y = df[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf_dt = DecisionTreeRegressor(max_depth=10)
        clf_dt.fit(X_train, y_train)
        logger.debug('target: %s score: %.4f' % (target_column, clf_dt.score(X_test, y_test)))

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
