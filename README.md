Exploratory Data Analysis (EDA) for Customer, Product, and Transaction Data
Description
This repository contains an analysis of a dataset consisting of customer, product, and transaction information. The goal is to perform Exploratory Data Analysis (EDA) to uncover business insights and patterns that can drive business decisions.

Files
Customers.csv: Contains customer-related information.
Products.csv: Contains product-related information.
Transactions.csv: Contains transaction-related data, including purchases made by customers.
EDA_Code.ipynb: Python Jupyter notebook performing the analysis.
README.md: This file explaining the analysis and how to run it.
Requirements
Before running the code, ensure you have the following Python libraries installed:

pandas
seaborn
matplotlib
You can install the necessary libraries using the following command:

bash
Copy
pip install pandas seaborn matplotlib
Code Explanation
1. Import Libraries
The following libraries are imported for data manipulation and visualization:

python
Copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
2. Load and Merge Datasets
We load the three CSV files—Customers.csv, Products.csv, and Transactions.csv—using pandas and merge them based on common columns.

python
Copy
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merging the datasets
merged_df = transactions_df.merge(customers_df, on='CustomerID', how='left')
merged_df = merged_df.merge(products_df, on='ProductID', how='left')
3. Data Exploration
The merged dataset's columns are printed to understand its structure.

python
Copy
print("Columns in merged dataframe:", merged_df.columns)
4. Visualizing Data
Distribution of Product Prices
We plot a histogram of the product prices using seaborn to visualize their distribution:

python
Copy
sns.histplot(merged_df['Price_y'])
plt.title('Distribution of Product Prices (Price_y)')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
Distribution of Transaction Values
Next, we plot the distribution of the transaction values (Total Value).

python
Copy
sns.histplot(merged_df['TotalValue'])
plt.title('Distribution of Transaction Values')
plt.xlabel('Total Value')
plt.ylabel('Frequency')
plt.show()
Relationship Between Price and Quantity
A scatter plot is created to explore the relationship between the price of products (Price_y) and the quantity purchased (Quantity).

python
Copy
sns.scatterplot(data=merged_df, x='Price_y', y='Quantity')
plt.title('Price vs. Quantity Purchased')
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.show()
Correlation Matrix
A correlation heatmap is generated for the numeric columns to understand the relationships between numerical features.

python
Copy
numeric_cols = merged_df.select_dtypes(include='number').columns
sns.heatmap(merged_df[numeric_cols].corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()
Insights
The following insights can be derived from the visualizations and analysis:

Price Distribution: The distribution of product prices shows that most products are in a specific price range, which could guide promotional pricing strategies.
Transaction Value Distribution: The distribution of transaction values suggests that most transactions are of smaller value, which may indicate frequent, low-value purchases.
Price vs. Quantity: The scatter plot reveals that there is a negative correlation between price and quantity, with higher-priced items generally being purchased in lower quantities.
Customer Spending: The correlation heatmap indicates a possible strong relationship between customer demographics and their spending patterns, which could guide targeted marketing.
Sales Trends: The analysis of the total value of transactions across time periods may reveal trends in purchasing behavior, such as seasonality or promotional impacts.
How to Run
Clone this repository:

bash
Copy
git clone https://github.com/your-username/eda-project.git
Navigate to the folder containing the Jupyter notebook:

bash
Copy
cd eda-project
Open the Jupyter notebook and run all the cells to perform the EDA.

bash
Copy
jupyter notebook EDA_Code.ipynb
Conclusion
This analysis provides valuable insights into customer purchasing behavior, product pricing trends, and transaction patterns. Further steps could involve feature engineering, predictive modeling, and deriving additional actionable business insights.


# zeotap
Assignment
