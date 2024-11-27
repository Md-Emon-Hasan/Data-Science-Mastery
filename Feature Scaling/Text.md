### What is Feature Scaling?

Feature scaling is a technique used to normalize or standardize the range of independent variables (features) in your dataset. Many machine learning algorithms, especially those based on distance calculations (like k-nearest neighbors, SVMs, and gradient descent-based models), perform better when the data is scaled. If features in the dataset have different ranges (e.g., one feature ranges from 1 to 1000 and another ranges from 0 to 1), the model may give more importance to features with larger values, leading to biased or incorrect results.

### When to Perform Feature Scaling?

- **Before feeding data to machine learning models** that rely on distance or gradient-based optimizations. Algorithms like k-Nearest Neighbors (k-NN), Support Vector Machines (SVM), and linear regression benefit from scaling.
- **For models sensitive to the scale of features**, like neural networks and logistic regression.
- **When the features have different units of measurement** (e.g., height in cm and weight in kg). Feature scaling ensures that all features contribute equally.

### Types of Feature Scaling

1. **Normalization (Min-Max Scaling)**:
   - This technique scales the data between a specific range (usually [0, 1]). The formula for Min-Max scaling is:
     \[
     X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}
     \]
   - **When to use**: Useful when the data needs to be bounded in a specific range, such as for algorithms that make assumptions about the scale of the data, like neural networks.
   - **Pros**: Retains the relationships between features, which is important for certain models.
   - **Cons**: Sensitive to outliers since they can skew the min and max values.

2. **Standardization (Z-Score Normalization)**:
   - Standardization transforms the data to have a mean of 0 and a standard deviation of 1:
     \[
     X_{\text{scaled}} = \frac{X - \mu}{\sigma}
     \]
     where \( \mu \) is the mean and \( \sigma \) is the standard deviation of the feature.
   - **When to use**: This is more appropriate when the data follows a Gaussian distribution (normal distribution), especially for algorithms that assume data follows a normal distribution, like logistic regression and linear regression.
   - **Pros**: Less sensitive to outliers than Min-Max scaling.
   - **Cons**: Does not bound the data within a specific range, which can be problematic for certain algorithms.

3. **Robust Scaling**:
   - This method scales data using the median and the interquartile range (IQR). It is robust to outliers because it does not rely on extreme values.
   - **When to use**: If the data contains a lot of outliers, robust scaling is preferred.
   - **Pros**: Works well when outliers are present in the dataset.
   - **Cons**: Does not guarantee a specific range like Min-Max scaling.

### How to Perform Feature Scaling

In Python, you can use libraries like **scikit-learn** to easily scale features. Here's how you can apply scaling:

- **Using Min-Max Scaling**:
  ```python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(data)
  ```

- **Using Standardization (Z-Score Normalization)**:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  standardized_data = scaler.fit_transform(data)
  ```

- **Using Robust Scaling**:
  ```python
  from sklearn.preprocessing import RobustScaler
  scaler = RobustScaler()
  robust_scaled_data = scaler.fit_transform(data)
  ```

### Summary

Feature scaling is essential when dealing with machine learning models sensitive to the scale of data. The choice between normalization, standardization, or robust scaling depends on the nature of your dataset and the algorithm you plan to use. Always ensure that scaling is done **after splitting** the data into training and test sets to avoid data leakage.

Sources:
- [scikit-learn Feature Scaling](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature Scaling Explained](https://machinelearningmastery.com/standardize-or-normalize-data-for-machine-learning/)