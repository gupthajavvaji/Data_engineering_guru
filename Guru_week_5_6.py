import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("CAC40_stocks_2021_2023.csv")

def read_data(file_name):
    return pd.read_csv(df = pd.read_csv("C:/Users/theco/PycharmProjects/pythonProject/CAC40_stocks_2021_2023.csv"), parse_dates=['Date'])

def clean_data(df):
    # 1. Check for Missing Values
    missing_values = df.isnull().sum()
    print("Missing Values:\n", missing_values)
    ##there are no missing values

    ###standardize casing
    df['Stock'] = df['Stock'].str.upper()

    #check date format consistency
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='raise', format='%Y-%m-%d')
    except ValueError as e:
        print(f"Error: {e}. Please check and fix the date format.")

    return df
###descriptive statistics
grouped_stats = df.groupby('Stock').agg({
    'Volume': ['mean', 'median', 'max', 'min', 'std'],
    'Close': ['mean', 'median', 'max', 'min', 'std', lambda x: x.mode()[0]]
})
for stock, stats in grouped_stats.iterrows():
    print(f"Stock: {stock}")
    print(f"  Volume Mean: {stats['Volume', 'mean']}")
    print(f"  Volume Median: {stats['Volume', 'median']}")
    print(f"  Volume Max: {stats['Volume', 'max']}")
    print(f"  Volume Min: {stats['Volume', 'min']}")
    print(f"  Volume Std: {stats['Volume', 'std']}")
    print(f"  Close Mean: {stats['Close', 'mean']}")
    print(f"  Close Median: {stats['Close', 'median']}")
    print(f"  Close Max: {stats['Close', 'max']}")
    print(f"  Close Min: {stats['Close', 'min']}")
    print(f"  Close Std: {stats['Close', 'std']}")
    print(f"  Close Mode: {stats['Close', '<lambda_0>']}")
    print("\n")


grouped_stats.columns = ['Volume_Mean', 'Volume_Median', 'Volume_Max', 'Volume_Min', 'Volume_Std',
                          'Close_Mean', 'Close_Median', 'Close_Max', 'Close_Min', 'Close_Std', 'Close_Mode']

for stock, stats in grouped_stats.iterrows():
    print(f"Stock: {stock}")
    print(f"  Volume Mean: {stats['Volume_Mean']}")
    print(f"  Volume Median: {stats['Volume_Median']}")
    print(f"  Volume Max: {stats['Volume_Max']}")
    print(f"  Volume Min: {stats['Volume_Min']}")
    print(f"  Volume Std: {stats['Volume_Std']}")
    print(f"  Close Mean: {stats['Close_Mean']}")
    print(f"  Close Median: {stats['Close_Median']}")
    print(f"  Close Max: {stats['Close_Max']}")
    print(f"  Close Min: {stats['Close_Min']}")
    print(f"  Close Std: {stats['Close_Std']}")
    print(f"  Close Mode: {stats['Close_Mode']}")
    print("\n")

print("Grouped Statistics by Stock:")
print(grouped_stats)

df['Date'] = pd.to_datetime(df['Date'])

# Filter data within the specified year 2022
start_date = '2022-01-01'
end_date = '2022-12-31'

def filter_data(df):
    df['Date'] = pd.to_datetime(df['Date'])
    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

df = pd.read_csv("C:/Users/theco/PycharmProjects/pythonProject/CAC40_stocks_2021_2023.csv")
df = clean_data(df)
trimmed_data = filter_data(df)


def calculate_roi(trimmed_data):
    return trimmed_data.groupby('Stock').apply(lambda x: (x['Close'].iloc[-1] - x['Close'].iloc[0]) / x['Close'].iloc[0] * 100).reset_index(name='ROI')


def plot_top_5_roi_stocks_2022(trimmed_data, roi_data):
    top_5_stocks = roi_data.nlargest(5, 'ROI')['Stock']
    top_5_stocks_data_2022 = trimmed_data[(trimmed_data['Stock'].isin(top_5_stocks)) & (trimmed_data['Date'].dt.year == 2022)]
    top_5_stocks_roi = roi_data[roi_data['Stock'].isin(top_5_stocks)]

    plt.figure(figsize=(10, 6))
    for stock in top_5_stocks:
        stock_data = top_5_stocks_data_2022[top_5_stocks_data_2022['Stock'] == stock]
        sns.lineplot(x='Date', y='Close', data=stock_data, label=f'{stock} - ROI: {top_5_stocks_roi[top_5_stocks_roi["Stock"] == stock]["ROI"].values[0]:.2f}%')

    plt.title('Top 5 ROI Stocks in CAC40-2022')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(title='Stock - ROI (%)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_total_value_traded(trimmed_data, roi_data):
    top_5_stocks = roi_data.nlargest(5, 'ROI')['Stock']
    top_5_stocks_data = trimmed_data[trimmed_data['Stock'].isin(top_5_stocks)].copy()
    top_5_stocks_data.loc[:, 'TotalValueTraded'] = top_5_stocks_data['Close'] * top_5_stocks_data['Volume']

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Date', y='TotalValueTraded', hue='Stock', data=top_5_stocks_data, palette='muted')
    plt.title('Total Value Traded for Top 5 ROI Stocks')
    plt.xlabel('Date')
    plt.ylabel('Total Value Traded')
    plt.legend(title='Stock', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_volume_distribution(trimmed_data, roi_data):
    top_5_stocks = roi_data.nlargest(5, 'ROI')['Stock']
    top_5_stocks_data = trimmed_data[trimmed_data['Stock'].isin(top_5_stocks)]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns for 5 plots

    for i, (stock, ax) in enumerate(zip(top_5_stocks, axs.flatten())):
        sns.histplot(top_5_stocks_data[top_5_stocks_data['Stock'] == stock]['Volume'], bins=30, kde=True, label=stock, ax=ax)
        ax.set_title(f'Volume Distribution - {stock}')
        ax.set_xlabel('Volume')
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_daily_open_close_prices(trimmed_data, roi_data):
    highest_roi_stock = roi_data.nlargest(1, 'ROI')['Stock'].iloc[0]
    highest_roi_stock_data = trimmed_data[trimmed_data['Stock'] == highest_roi_stock]

    plt.figure(figsize=(10, 6))
    plt.plot(highest_roi_stock_data['Date'], highest_roi_stock_data['Open'], label='Daily Open Price', marker='o', color='red')
    plt.plot(highest_roi_stock_data['Date'], highest_roi_stock_data['Close'], label='Daily Close Price', marker='o', color='green')
    plt.title(f'Highest ROI Stock Daily Open and Close Prices ({highest_roi_stock})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_average_daily_pricing(trimmed_data, roi_data):
    highest_roi_stock = roi_data.nlargest(1, 'ROI')['Stock'].iloc[0]
    highest_roi_stock_data = trimmed_data[trimmed_data['Stock'] == highest_roi_stock]
    average_daily_prices = highest_roi_stock_data.groupby('Date')['Close'].mean().reset_index()

    coefficients = np.polyfit(np.arange(len(average_daily_prices)), average_daily_prices['Close'], 2)
    polynomial = np.poly1d(coefficients)
    equation = f"{polynomial[2]:.2f}x^2 + {polynomial[1]:.2f}x + {polynomial[0]:.2f}"

    plt.figure(figsize=(8, 6))
    plt.scatter(average_daily_prices['Date'], average_daily_prices['Close'], label=f'{highest_roi_stock} - Best Fit Curve: {equation}')
    plt.plot(average_daily_prices['Date'], polynomial(np.arange(len(average_daily_prices))), color='red', linestyle='--')
    plt.title(f'Highest ROI Stock Average Daily Pricing with Best Fit Curve ({highest_roi_stock})')
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Calculate ROI
roi_data = calculate_roi(trimmed_data)

# Task 1: Plot top 5 ROI stocks in CAC40-2022
plot_top_5_roi_stocks_2022(trimmed_data, roi_data)

# Task 2: Plot total value traded for top 5 ROI stocks
plot_total_value_traded(trimmed_data, roi_data)

# Task 3: Plot top 5 volumn distribution
plot_volume_distribution(trimmed_data, roi_data)

# Task 4: Plot daily open/close prices for the highest ROI stock
plot_daily_open_close_prices(trimmed_data, roi_data)

# Task 5: Plot average daily pricing with trend
plot_average_daily_pricing(trimmed_data, roi_data)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Assuming 'X' contains the features and 'y' contains the target variable
X = df[['Open','High','Low','Close','Adj Close','Volume','Stock']]  # Define your features
y = df['Volume']  # Define your target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertion of string labels to numerical values
label_encoder = LabelEncoder()
df['Stock']= label_encoder.fit_transform(df['Stock'])

label_encoder = LabelEncoder()
X_train['Stock'] = label_encoder.fit_transform(X_train['Stock'])
X_test['Stock'] = label_encoder.transform(X_test['Stock'])

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # Number of neighbors

# Train Model
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Assessing model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Additional evaluation metrics
print(classification_report(y_test, y_pred, zero_division=0))

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Assuming 'data' contains the features for clustering
numerical_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

# Selecting only the numerical columns from the DataFrame
numerical_data = df[numerical_columns]

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform only the numerical columns
scaled_data = scaler.fit_transform(numerical_data)

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)

# Initializing KMeans
kmeans = KMeans(n_clusters=3, n_init=10)  # Set the number of clusters and n_init value

# Fitting the model to the scaled data
kmeans.fit(scaled_data)

# Predicting cluster labels
cluster_labels = kmeans.predict(scaled_data)

# Accessing cluster centers
cluster_centers = kmeans.cluster_centers_

# Visualizing the clusters (for 2D data)
plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=cluster_labels, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', c='red', s=200)
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
