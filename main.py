import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier


#Load the data from Github
url_data = "https://raw.githubusercontent.com/Theodor-Bern/SML-Project/refs/heads/main/training_data_VT2026%20(1).csv"


df = pd.read_csv(url_data)

numerical= ['temp', 'humidity', 'windspeed', 'dew', 'precip', 'snow', 'snowdepth', 'cloudcover', 'visibility']
categorical = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']



# EDA

 #We investigate the shape of the data
def get_data_overview():
    print(f"Number of observations: {df.shape[0]}")
    print(f"Number of features: {df.shape[1]}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nClass Distribution:\n{df['increase_stock'].value_counts(normalize=True)}, {df['increase_stock'].value_counts()}")
    

#Shows the disitribtion of data. For example, what is the share of january?
def sampling_distribution():
    sns.set_style("whitegrid")

    df_month = df['month'].value_counts(normalize=True).reset_index()
    df_month.columns = ['month', 'share']
    df_month = df_month.sort_values('month')



    df_hour = df['hour_of_day'].value_counts(normalize=True).reset_index()
    df_hour.columns = ['hour', 'share']
    df_hour = df_hour.sort_values('hour')


    df_day = df['day_of_week'].value_counts(normalize=True).reset_index()
    df_day.columns = ['day', 'share']
    df_day = df_day.sort_values('day')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_month, x='month', y='share', color='skyblue', edgecolor='black')
    plt.axhline(y=1/12, color='red', linestyle='--', linewidth=2, label='Ideal (1/12)')
    plt.title('Distribution per Month', fontsize=14)
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(12), month_labels)
    plt.ylim(0, 0.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_month.png', dpi=300, bbox_inches='tight')
    plt.show()

    #Plot of weekday distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_day, x='day', y='share', color='salmon', edgecolor='black')
    plt.axhline(y=1/7, color='red', linestyle='--', linewidth=2, label='Ideal (1/7)')
    plt.title('Distribution per Day of Week', fontsize=14)
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.xticks(range(7), day_labels)
    plt.ylim(0, 0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_weekday.png', dpi=300, bbox_inches='tight')
    plt.show()

    #Plot of hour distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_hour, x='hour', y='share', color='lightgreen', edgecolor='black')
    plt.axhline(y=1/24, color='red', linestyle='--', linewidth=2, label='Ideal (1/24)')
    plt.title('Share per Hour', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribution_hour.png', dpi=300, bbox_inches='tight')
    plt.show()
    
#Investigate how high demand varies for different temperatures
def temperature_trends():
 
    plt.figure(figsize=(10, 6))
    #Plot no 1
    #Compare the distribution of data for all points compared to high demand with respect to temperature
    sns.kdeplot(data=df, x='temp', label='Total distribution', fill=True, alpha=0.3)
    sns.kdeplot(data=df[df['increase_stock'] == 'high_bike_demand'],
                x='temp', label='Distribution of High Demand', fill=True, color='red')

    plt.title('Temperature correallation with high demand')
    plt.xlabel('Temperature')
    plt.ylabel('Density')
    plt.legend()

    plt.tight_layout()
    plt.savefig('extended_weather_analysis.png', dpi=150)
    plt.show()

    #Plot no2
    #Make intervals of size 5 from -10 to +40 celcius
    temp_labels = [f"{i}-{i+5}" for i in range(-10, 40, 5)]
    df['temp_bins'] = pd.cut(df['temp'], bins=range(-10, 45, 5), labels=temp_labels)

    # Count for each combination of temperature and demand
    # Vi använder unstack() för att få 'increase_stock' som egna kolumner (High/Low)
    temp_demand_counts = df.groupby(['temp_bins', 'increase_stock'], observed=True).size().unstack(fill_value=0)

    ax = temp_demand_counts.plot(kind='bar', figsize=(12, 6), color=['skyblue', 'salmon'], width=0.8)

    plt.title('Amount of observations: Low vs High Demand per Temperatureinterval', fontsize=14)
    plt.xlabel('Temperature interval (°C)', fontsize=12)
    plt.ylabel('Amount of observations mätpunkter ', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Demand')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('temp_demand_distribution.png', dpi=150)
    plt.tight_layout()
    plt.show()


        
#How does the high demand vary depending on temporal feautues
def data_distribution_across_time():
    #Hour
    plt.figure(figsize=(10, 5))
    df.groupby('hour_of_day')['increase_stock'].apply(
        lambda x: (x == 'high_bike_demand').mean()
    ).plot(kind='bar', color='blue')

    plt.title('Probability for high demand per hour', fontsize=14)
    plt.ylabel('Share of High Demand (0.0 - 1.0)')
    plt.xlabel('Hour of day')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('demand_hour.png', dpi=150)
    plt.show()

    #Weekday
    plt.figure(figsize=(10, 5))
    df.groupby('day_of_week')['increase_stock'].apply(
        lambda x: (x == 'high_bike_demand').mean()
    ).plot(kind='bar', color='blue')

    plt.title('Probability for high demand per weekday', fontsize=14)
    plt.ylabel('Share of high demand(0.0 - 1.0)')
    plt.xlabel('Weekday(0=Monday, 6=Sunday')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('demand_weekday.png', dpi=150)
    plt.show()

    #Month
    plt.figure(figsize=(10, 5))
    df.groupby('month')['increase_stock'].apply(
        lambda x: (x == 'high_bike_demand').mean()
    ).plot(kind='bar', color='blue')

    plt.title('Probability for high demand', fontsize=14)
    plt.ylabel('Share of high demand (0.0 - 1.0)')
    plt.xlabel('Month')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('demand_month.png', dpi=150)
    plt.show()
    

def correalation_matrix():
    fig, ax = plt.subplots(figsize=(10, 8))
    numerical_no_snow = [f for f in numerical if f != 'snow']
    corr_matrix = df[numerical_no_snow].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
    ax.set_title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()



#How does the demand vary depending on if its weekday, weekend or holiday?
def demand_weekday_vs_weeknd():
    df['day_type'] = df.apply(
    lambda row: 'Holiday' if row['holiday'] == 1
    else 'Weekday' if row['weekday'] == 1
    else 'Weekend',
    axis=1
)

    plt.figure(figsize=(6,4))
    df.groupby('day_type')['increase_stock'].apply(
        lambda x: (x == 'high_bike_demand').mean()
    ).plot(kind='bar')

    plt.title('Share of high demand by Day type')
    plt.ylabel('Share of high demand')
    plt.tight_layout()
    plt.show()





#Preprocessing


df['increase_stock'] = df['increase_stock'].map({'high_bike_demand':1, 'low_bike_demand':0})

#We separate the increase_stock from input and define it as the output
#Drop snow column since it only contains zero values
X = df.drop(['increase_stock', 'snow'], axis=1)
y = df['increase_stock']


#Snow not included
numeric_features = ['temp', 'humidity', 'windspeed', 'dew', 'precip', 'snowdepth', 'cloudcover', 'visibility']
categorical_features = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday', 'summertime']
    


#Bygg preprocessing till pipelines

preprocessor_sensitive = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

preprocessor_tree = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features), # Fyller NaN med median
        ('cat', SimpleImputer(strategy='constant', fill_value=-1), categorical_features) # Fyller NaN med -1
    ])



#Separate test and train data to estimate E_new
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


#Forward and backward selection XGBoost
def feature_selection_XGBoost():

    all_features = numeric_features + categorical_features

    xgb_fs = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
        n_jobs=-1
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    baseline = cross_val_score(xgb_fs, X_train[all_features], y_train, cv=cv_strategy, scoring='f1').mean() 
    
    #forward selection with sequentialfeatureselector
    sfs_forward = SequentialFeatureSelector(
        xgb_fs,
        n_features_to_select='auto',
        tol=0.001,
        direction='forward',
        scoring='f1',
        cv=cv_strategy,
        n_jobs=-1
    )
    sfs_forward.fit(X_train[all_features], y_train)

    forward_features = [f for f, s in zip(all_features, sfs_forward.get_support()) if s]
    score_fwd = cross_val_score(xgb_fs, X_train[forward_features], y_train, cv=cv_strategy, scoring='f1').mean()
    print(f'Forward selection: chose these({len(forward_features)}): {forward_features}')

    #repeat for backward selection
    sfs_backward = SequentialFeatureSelector(
        xgb_fs,
        n_features_to_select='auto',
        tol=-0.001,
        direction='backward',
        scoring='f1',
        cv=cv_strategy,
        n_jobs=-1
    )
    sfs_backward.fit(X_train[all_features], y_train)

    backward_features = [f for f, s in zip(all_features, sfs_backward.get_support()) if s]
    score_bwd = cross_val_score(xgb_fs, X_train[backward_features], y_train, cv=cv_strategy, scoring='f1').mean()
    print(f'Backward — valda features ({len(backward_features)}): {backward_features}')



    print(f"{'Method':<20} | {'Number':<6} | {'F1 (mean ± std)':<20}")
   

    for name, features in [("All features", all_features), ("Forward", forward_features), ("Backward", backward_features)]:
        scores = cross_val_score(xgb_fs, X_train[features], y_train, cv=cv_strategy, scoring='f1')
        print(f"{name:<20} | {len(features):<6} | {scores.mean():.4f} ± {scores.std():.4f}")

def XGBoost_algorithm():
    
    #We use the features after backward selection, we drop snowdepth and weekday
    xgb_numeric_features = ['temp', 'humidity', 'windspeed', 'dew', 'precip', 'cloudcover', 'visibility']
    xgb_categorical_features = ['hour_of_day', 'day_of_week', 'month', 'holiday', 'summertime']
    
    preprocessor_xgb = ColumnTransformer(transformers=[
    ('num', SimpleImputer(strategy='median'), xgb_numeric_features),
    ('cat', SimpleImputer(strategy='constant', fill_value=-1), xgb_categorical_features)
])

    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor_xgb),
        ('classifier', XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        ))
    ])

    xgb_param_grid = {
        'classifier__n_estimators': [150, 300, 400,500,700,900],
        'classifier__max_depth': [ 3, 5, 7, 9, 11, 13, 15],
        'classifier__learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
        'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'classifier__min_child_weight': [ 3, 5, 7, 9, 11],
        'classifier__gamma': [ 0.3, 0.4, 0.5, 0.6, 0.7],
        'classifier__scale_pos_weight': [1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    }
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    xgb_search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=xgb_param_grid,
        n_iter=200,     
        scoring='f1',       
        cv=cv_strategy,
        random_state=42,
        verbose=1,
        n_jobs=-1          

    )

    xgb_search.fit(X_train, y_train)

    print(f"\nBest hyperparameters:\n{xgb_search.best_params_}")
    print(f"\nBest CV F1 Score: {xgb_search.best_score_:.4f}")

    y_pred = xgb_search.best_estimator_.predict(X_test)
    y_proba = xgb_search.best_estimator_.predict_proba(X_test)[:, 1]

    print("\nPerformance on test set:")
    print(classification_report(y_test, y_pred, target_names=['Low Demand', 'High Demand']))
    
    

def naive_classifier():
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    
    print("Naive Classifier (always predicts majority class):")
    print(classification_report(y_test, y_pred, target_names=['Low Demand', 'High Demand']))




def main():
    #feature_selection_XGBoost()
    naive_classifier()
main()