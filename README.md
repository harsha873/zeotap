Lookalike Model for Customer Recommendations
Description
This repository contains a lookalike model that recommends similar customers based on their profile and transaction history. The model uses cosine similarity to calculate the similarity between customers and suggests the top 3 most similar customers for each customer.

The analysis is based on a dataset containing customer, product, and transaction information. The model combines both customer attributes and transaction behavior to generate lookalike recommendations.

Files
Customers.csv: Contains customer-related information.
Products.csv: Contains product-related information.
Transactions.csv: Contains transaction-related data, including purchases made by customers.
Lookalike.csv: Contains the lookalike recommendations for customers C0001 to C0020, along with their similarity scores.
Lookalike_Model.ipynb: Python Jupyter notebook implementing the lookalike model.
README.md: This file explaining the project and how to run it.
Requirements
Before running the code, ensure you have the following Python libraries installed:

pandas
scikit-learn
You can install the necessary libraries using the following command:

bash
Copy
pip install pandas scikit-learn
Approach
1. Data Loading and Preprocessing
We load three CSV files — Customers.csv, Products.csv, and Transactions.csv — using pandas and merge them to form a complete dataset that combines customer, product, and transaction information.

python
Copy
customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

# Merge datasets
merged_df = transactions_df.merge(customers_df, on='CustomerID', how='left')
merged_df = merged_df.merge(products_df, on='ProductID', how='left')
2. Feature Engineering
We create aggregated features for each customer:

AvgTransactionValue: Average value of transactions.
TotalQuantity: Total quantity of products purchased.
AvgProductPrice: Average price per product purchased.
TransactionCount: Number of transactions made by the customer.
python
Copy
customer_features = merged_df.groupby('CustomerID').agg({
    'TotalValue': 'mean',                 # Average transaction value
    'Quantity': 'sum',                    # Total quantity purchased
    'Price_y': 'mean',                    # Average price per product
    'ProductName': 'count'                # Number of transactions
}).reset_index()

customer_features.rename(columns={
    'TotalValue': 'AvgTransactionValue',
    'Quantity': 'TotalQuantity',
    'Price_y': 'AvgProductPrice',
    'ProductName': 'TransactionCount'
}, inplace=True)
3. Cosine Similarity Calculation
We use cosine similarity to calculate the similarity scores between customer feature vectors.

python
Copy
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(customer_features.iloc[:, 1:])
4. Lookalike Recommendation
For each customer (C0001 to C0020), we calculate the top 3 lookalike customers based on similarity scores. The get_top_lookalikes() function retrieves the most similar customers.

python
Copy
def get_top_lookalikes(customer_id, customer_ids, similarity_scores, top_n=3):
    customer_idx = customer_ids.index(customer_id)
    similarity_scores_for_customer = similarity_scores[customer_idx]
    
    # Get the indices of the top N most similar customers, excluding the customer itself
    top_indices = similarity_scores_for_customer.argsort()[-top_n-1:-1][::-1]
    
    # Retrieve the customer IDs and similarity scores
    top_customers = [(customer_ids[i], similarity_scores_for_customer[i]) for i in top_indices]
    
    return top_customers
5. Saving the Results
The results are stored in a CSV file (Lookalike.csv) with columns:

CustomerID: The ID of the customer for whom the recommendations are made.
LookalikeCustomerID: The ID of a recommended similar customer.
SimilarityScore: The similarity score between the two customers.
python
Copy
lookalike_df.to_csv('Lookalike.csv', index=False)
Example Output
The Lookalike.csv file contains the following format:

python
Copy
CustomerID,LookalikeCustomerID,SimilarityScore
C0001,C0045,0.95
C0001,C0032,0.93
C0001,C0024,0.91
C0002,C0078,0.89
C0002,C0053,0.87
C0002,C0034,0.86
...
How to Run
Clone this repository:

bash
Copy
git clone https://github.com/your-username/lookalike-model.git
Navigate to the folder containing the Jupyter notebook:

bash
Copy
cd lookalike-model
Open the Jupyter notebook and run all the cells to build the lookalike model.

bash
Copy
jupyter notebook Lookalike_Model.ipynb
The lookalike recommendations will be saved in Lookalike.csv.
Evaluation Criteria
The lookalike model is evaluated based on:

Model Accuracy and Logic: The cosine similarity is a valid measure of similarity for this type of problem. Customers with similar transaction histories and purchasing behavior should be recommended as lookalikes.
Quality of Recommendations: The top 3 lookalike customers are chosen based on the highest similarity scores. These scores reflect how closely the customers' transaction patterns and product preferences align.
Conclusion
The Lookalike Model provides valuable recommendations for similar customers based on purchasing behavior. These insights can help businesses with customer segmentation, targeted marketing, and personalized offerings.

License
This project is licensed under the MIT License - see the LICENSE file for details.

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
