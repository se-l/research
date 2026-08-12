import pandas as pd
import lightgbm as lgb

from derivatives.typess.option_frame import OptionFrame
from sklearn.model_selection import KFold, train_test_split

if __name__ == '__main__':
    '''
    1) Learn to hedge with equity, how much...
    2) Learn to hedge with equity, when...
    3) Learn to hedge with more instruments, derivatives.
    4) Learn to hedge a portfolio or underlying's and derivatives with a all underlyings, derivatives and index derivatives/stocks.    
    '''
    option_frame = OptionFrame.load_frame(equity, resolution, seq_ret_threshold, '4')

    dft = option_frame.df_options
    iv = 'hv'
    dft[f'hedging_error_{iv}'] = dft[f'dP_delta_{iv}'] / (dft[f'dP'] - dft[f'dP_theta_{iv}'] - dft[f'dP_gamma_{iv}']) - 1

    # # that moneymess may be better using Fwd K
    dft['tenor'] = (dft.index.get_level_values('expiry').to_series().reset_index(drop=True) - pd.Series(dft.index.get_level_values('ts').to_pydatetime()).apply(lambda x: x.date())).apply(lambda x: x.days / 365).values
    dft['moneyness'] = dft.index.get_level_values('strike').astype(float) / dft['spot']
    dft['right'] = dft.index.get_level_values('right')

    ix = dft.index[
        (dft[f'dP'].abs() > 0.11) &
        (dft[f'returnS'].abs() < 0.02) &
        (dft[f'returnS'].abs() > 0.001) &
        (dft[f'delta_{iv}'].abs() < 0.95) &
        (dft[f'delta_{iv}'].abs() > 0.05) &
        (dft['moneyness'] > 0.9) &
        (dft['moneyness'] < 1.1) &
        dft[f'hedging_error_{iv}'] < 1 &
        dft[f'hedging_error_{iv}'] > -1
        ]

    # By Delta Bucket, plot the applied delta vs the theoretical delta that minimizes hedging error
    # Many dimensions: moneyness, right, tenor, IV - Given Delta surface with hedging error x. learn surface leverage function minimizing hedging error.
    # Can use Basin Hopping again? IV is varying

    # # PLotting sample
    #     # delta = dft.loc[ix, f'delta_{iv}']
    #     dft.set_index(f'delta_{iv}').sort_index()[['he']].plot(style='o', alpha=0.5, figsize=(15, 10))
    #
    #     X = dft.loc[ix, 'moneyness'].values
    #     Y = dft.loc[ix, f'delta_{iv}'].values
    #     plot_surface()

    # Split your dataset into training and testing sets
    col_target = f'hedging_error_{iv}'
    df = dft[['moneyness', 'tenor', 'right', f'delta_{iv}', col_target]].reset_index(drop=True)
    df['right'] = df['right'].map({'call': 0, 'put': 1})
    df['moneyness'] = df['moneyness'].astype(float)
    df['hedging_error_hv'] = df['hedging_error_hv'].astype(float)
    df = df[df[col_target].notna()]
    X_train, X_test, y_train, y_test = train_test_split(df.drop(col_target, axis=1), df[col_target], test_size=0.2, random_state=42)

    # Create a KFold object with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize the LightGBM regressor
    lgb_regressor = lgb.LGBMRegressor()

    # Perform 5-fold cross-validation
    results = []
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_train_fold, y_train_fold = X_train.iloc[train_index], y_train.iloc[train_index]
        X_val_fold, y_val_fold = X_train.iloc[val_index], y_train.iloc[val_index]

        # Train the model on the current fold's training set
        lgb_regressor.fit(X_train_fold, y_train_fold)

        # Evaluate the model on the current fold's validation set
        y_pred = lgb_regressor.predict(X_val_fold)
        score = np.mean((y_pred - y_val_fold))

        # Append the current fold's score to the results list
        results.append(score)

    # Calculate the mean and std of the cross-validation scores
    mean_score = np.mean(results)
    std_score = np.std(results)

    # Print the mean and std of the cross-validation scores
    print(f'Mean cross-validation score: {mean_score:.4f}')
    print(f'Standard deviation of cross-validation scores: {std_score:.4f}')

    # Train the model on the entire training set
    lgb_regressor.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lgb_regressor.predict(X_test)

    # Evaluate the model on the test set
    score = np.mean((y_pred - y_test) ** 2)

    # Print the test set score
    print(f'Test set score: {score:.4f}')