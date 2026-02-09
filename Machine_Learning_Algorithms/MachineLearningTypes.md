# Machine Learning Types

- Supervised Learning  
- Unsupervised Learning  
- Reinforcement Learning  

---
---
## Supervised Learning

**Definition:**  
Supervised learning is a type of machine learning where the model is trained on labeled data. Each training example has a known input and correct output, allowing the model to learn the relationship between them.

**Example:**  
It is like teaching a child to recognize animals by showing pictures and telling them the name of each animal.

**Example in Machine Learning:**  
After the model is trained, it uses learned patterns to make predictions on new inputs.  
For example, it may predict with **97% confidence** that an image contains a cat.

---
## Types of Supervised Learning

Supervised learning is mainly of two types:
```
                SUPERVISED LEARNING
                       |
           -------------------------
           |                       |
     Classification           Regression
           |                       |
   Predicts categories      Predicts numbers
   (labels/classes)         (continuous values)
   ```
### 1. Classification

**Definition:**  
Used when the output variable is categorical (labels or classes).

**Examples:**  
- Spam or not spam email  
- Cat vs Dog image classification  
- Disease prediction (Yes/No)

**Output:**  
Discrete values (categories)

---

### 2. Regression

**Definition:**  
Used when the output variable is continuous (numeric value).

**Examples:**  
- House price prediction  
- Salary prediction  
- Temperature forecasting  

**Output:**  
Continuous numeric value

---
---
## Classification Algorithms

Classification algorithms are used to predict **categorical outputs** (labels/classes).

### Common Classification Algorithms

1. **Logistic Regression**  
   - Used for binary classification  
   - Simple and fast baseline model  

2. **Decision Tree**  
   - Tree-based model  
   - Easy to understand and visualize  

3. **Random Forest**  
   - Collection of multiple decision trees  
   - More accurate and reduces overfitting  

4. **K-Nearest Neighbors (KNN)**  
   - Classifies based on nearest neighbors  

5. **Support Vector Machine (SVM)**  
   - Finds the best boundary between classes  

6. **Naive Bayes**  
   - Probability-based algorithm  
   - Common in text classification  

7. **Neural Networks**  
   - Used for image, speech, and deep learning tasks  

### Example Use Cases
- Email spam detection  
- Disease prediction  
- Image classification  
- Fraud detection  

---
---
## Logistic Regression

Logistic Regression is a supervised learning algorithm used for **binary classification** problems.  
It predicts the probability that an input belongs to a particular class.

Instead of predicting a straight line like linear regression, logistic regression uses an **S-shaped curve** called the **sigmoid function**.

---

### Sigmoid Function

The sigmoid function converts any value into a range between **0 and 1**.

**Formula:**

```p = 1 / (1 + e^(-z))```

Where:  
```z = b0 + b1x  ```

The output is a probability.

- If probability > 0.5 → Class 1  
- If probability < 0.5 → Class 0  

---

### Sigmoid Curve (Conceptual Diagram)
```
Probability
1.0 | ********
| ****
0.5 |---------Decision Boundary---------
| ****
0.0 |********
Input (z)
```

The curve starts near 0, increases smoothly, and approaches 1.

---

### Decision Boundary

Logistic regression uses a threshold (usually 0.5).

- Above 0.5 → Class 1  
- Below 0.5 → Class 0  

This threshold creates a **decision boundary**.

---

### How Logistic Regression Works

1. Input features are provided  
2. Model calculates weighted sum (z)  
3. Sigmoid function converts to probability  
4. Model compares prediction with actual label  
5. Weights are updated to reduce error  
6. Model learns patterns  

---

### Example

Suppose we predict if a student passes an exam.

Study Hours → Model → Probability → Result  

If probability = 0.82  
Prediction = Pass  

If probability = 0.30  
Prediction = Fail  

---

### Logistic vs Linear Regression

| Linear Regression | Logistic Regression |
|------------------|--------------------|
| Predicts numbers | Predicts probability |
| Output range: any value | Output range: 0 to 1 |
| Straight line | S-shaped curve |
| Used for regression | Used for classification |

---

### Loss Function

Logistic regression uses **log loss (cross-entropy loss)** to measure error.

The goal is to minimize this loss during training.

---

### Key Points

- Used for binary classification  
- Outputs probability  
- Uses sigmoid function  
- Creates decision boundary  
- Works well for simple classification problems  

---
---

## Decision Tree

### Definition
A Decision Tree is a supervised learning algorithm used for **classification** and **regression**.  
It works like a flowchart where the model makes decisions by asking a series of questions about the data until it reaches a final prediction.

Each box is called a **node** and each decision leads to a branch.

---

### Example: Is a Person Fit or Not?

We want to predict whether a person is **Fit** or **Not Fit** based on:

- Exercise (Yes/No)  
- Diet (Healthy/Unhealthy)  
- Sleep Hours  

**Output:** Fit or Not Fit

---

### Decision Tree Diagram
```
            Exercise?
           /        \
        Yes          No
        |            |
    Diet Healthy?   Not Fit
      /     \
    Yes     No
    |        |
  Fit     Sleep > 7?
             /   \
           Yes   No
           |      |
          Fit   Not Fit
```


---

### How It Works

1. The model checks the most important feature first.  
2. It splits the data based on conditions.  
3. Each split creates branches.  
4. The process continues until a final prediction is made.  
5. The last node is called a **leaf node** (final answer).

---

### Step-by-Step Prediction Example

Person data:
- Exercise: Yes  
- Diet: Unhealthy  
- Sleep: 8 hours  

Path:
Exercise → Yes  
Diet → Unhealthy  
Sleep > 7 → Yes  

Final prediction: **Fit**

---

### Important Terms

- **Root Node:** First decision  
- **Internal Node:** Intermediate decision  
- **Leaf Node:** Final output  
- **Branch:** Path between nodes  

---

### Advantages

- Easy to understand  
- Visual representation  
- No complex math  
- Works with both numbers and categories  

---

### Disadvantages

- Can overfit data  
- Sensitive to small changes  
- Large trees become complex  

---

### Real-life Uses

- Health prediction  
- Loan approval  
- Customer churn  
- Fraud detection  

---
---
## Random Forest

### Definition
Random Forest is a supervised learning algorithm used for **classification** and **regression**.  
It works by combining many decision trees and taking the final prediction based on all of them.

Instead of using one tree, Random Forest builds **multiple trees** and combines their results.  
This makes the model more accurate and reduces overfitting.

---

### Simple Idea

Decision Tree = One expert  
Random Forest = Many experts voting  

The final answer is based on majority vote.

---

### Example: Is a Person Fit or Not?

Features:
- Exercise  
- Diet  
- Sleep  
- Age  

Each decision tree gives its prediction.

```
Tree 1 → Fit
Tree 2 → Fit
Tree 3 → Not Fit
Tree 4 → Fit
Tree 5 → Fit
```

Most trees say **Fit** → Final prediction = **Fit**

---

### How Random Forest Works

1. Create many random samples from data  
2. Build a decision tree for each sample  
3. Each tree makes a prediction  
4. Combine all predictions  
5. Final output = majority vote (classification)  
6. For regression → average value  

---

### Diagram
```
          Input Data
              |
    ---------------------
    |    |    |    |   |
  Tree1 Tree2 Tree3 Tree4 Tree5
    |    |    |    |   |
  Fit   Fit  Not  Fit  Fit
    \    |    |    |   /
       Majority Vote
            |
         Final Output
            Fit
```

---

### Why Random Forest Is Better than Decision Tree

| Decision Tree | Random Forest |
|--------------|--------------|
| Uses one tree | Uses many trees |
| Can overfit | Reduces overfitting |
| Less accurate | More accurate |
| Sensitive to data | More stable |

---

### Advantages

- High accuracy  
- Handles large data  
- Reduces overfitting  
- Works for classification & regression  
- Handles missing values well  

---

### Disadvantages

- Slower than decision tree  
- Harder to interpret  
- Uses more memory  

---

### Real-world Uses

- Disease prediction  
- Fraud detection  
- Recommendation systems  
- Stock prediction  
- Customer analysis  

---
---
## K-Nearest Neighbors (KNN)

### Definition
K-Nearest Neighbors (KNN) is a supervised learning algorithm used for **classification** and **regression**.  
It predicts the output by looking at the **K nearest data points** to a new input and using their values.

KNN is called a **lazy learning algorithm** because it does not train a model in advance.  
It stores all data and makes predictions when new data comes.

---

### Simple Idea

If most of your nearest neighbors are fit → you are fit.  
If most neighbors are not fit → you are not fit.

---

### Example: Is a Person Fit or Not?

Features:
- Age  
- Exercise hours  
- Weight  

Output:
Fit or Not Fit

We choose **K = 3**.
```
Nearest neighbors:
Person 1 → Fit
Person 2 → Fit
Person 3 → Not Fit

Majority = Fit
Final prediction = Fit
```

---

### How KNN Works

1. Choose value of K  
2. Calculate distance between new point and all data points  
3. Select K nearest points  
4. Check their labels  
5. Majority vote → final prediction  

---
### KNN Diagram (Fit vs Not Fit)

```
           Fit (o)
        o        o

   Not Fit (x)      Fit (o)
        x      ●      o

   Not Fit (x)      Not Fit (x)
        x        x
```
Legend:
●  = New person (to be predicted)  
o  = Fit  
x  = Not Fit  

Explanation
The model checks the nearest neighbors around the new person (●).
Suppose K = 5

Among the nearest neighbors:
3 are Fit (o)
2 are Not Fit (x)
Majority = Fit
Final Prediction → Fit

### Distance Formula (Euclidean Distance)

```distance = √((x1 − x2)² + (y1 − y2)²)```

KNN finds nearest points using distance.

---

### Choosing K

- Small K → sensitive to noise  
- Large K → smoother decision  
- Common choice → odd number (3, 5, 7)

---

### Advantages

- Simple to understand  
- No training phase  
- Works well for small datasets  
- Easy to implement  

---

### Disadvantages

- Slow for large datasets  
- Needs feature scaling  
- Sensitive to noise  
- Memory heavy  

---

### Real-world Uses

- Recommendation systems  
- Image classification  
- Medical diagnosis  
- Pattern recognition  
---
---
## Support Vector Machine (SVM)

### Definition
Support Vector Machine (SVM) is a supervised learning algorithm used for **classification** and **regression**.  
It works by finding the **best boundary (hyperplane)** that separates different classes of data.

The goal of SVM is to create a line (or plane) that divides the data with the **maximum margin**.

---

### Simple Idea

SVM tries to draw the best line between two classes so that the distance from both classes to the line is maximum.

---

### Example: Fit vs Not Fit

We want to classify whether a person is **Fit** or **Not Fit** based on:

- Exercise hours  
- Weight  

```
Fit (o) and Not Fit (x) points on graph

o        o        o

      ---- Best Boundary ----

x        x        x
```
The line in the middle separates the two groups.

### Key Concepts
Hyperplane: The boundary that separates classes
Margin: Distance between boundary and nearest points
Support Vectors: Closest points to boundary
These points decide the boundary

```
       o      o
  o            o
------------------------- ← Hyperplane
  x            x
       x      x
```
o = Fit
x = Not Fit
Line = Decision boundary


---

### How SVM Works

1. Plot data points  
2. Find best boundary separating classes  
3. Maximize margin  
4. Use support vectors  
5. Predict class for new data  

---

### Types of SVM

- Linear SVM → straight line boundary  
- Non-linear SVM → curved boundary (kernel trick)  

---

### Advantages

- Works well with small datasets  
- Effective in high dimensions  
- Strong mathematical foundation  
- Accurate for classification  

---

### Disadvantages

- Slow for large datasets  
- Hard to tune parameters  
- Not easy to visualize in high dimensions  

---

### Real-world Uses

- Face detection  
- Text classification  
- Bioinformatics  
- Image classification  

---
---
## Naive Bayes

### Definition
Naive Bayes is a supervised learning classification algorithm based on **Bayes' Theorem**.  
It predicts the class of data using probability.

It is called **naive** because it assumes that all features are independent of each other.

---

### Simple Idea

Naive Bayes calculates the probability of each class and chooses the class with the highest probability.

Example:  
If probability of Spam = 0.9  
If probability of Not Spam = 0.1  

Prediction → Spam

---

### Example: Spam Email Detection

Input features:
- Contains "offer"?  
- Contains "free"?  
- Contains "win"?  

Output:
Spam or Not Spam

The model calculates probability using past data and predicts the most likely class.

---

### Bayes Theorem

```P(A|B) = (P(B|A) × P(A)) / P(B)```

Where:
- P(A|B) = probability of class A given B  
- P(B|A) = probability of B given class A  
- P(A) = prior probability  
- P(B) = evidence  

---

### Diagram

```
          New Email
              |
      Check Words/Features
              |
     Calculate Probabilities
        /              \
   Spam (0.85)     Not Spam (0.15)
              |
        Choose Highest
              |
          Final Output
              Spam
```

## How Naive Bayes Works

1. Calculate prior probability of each class  
2. Calculate likelihood of features  
3. Apply Bayes theorem  
4. Compare probabilities  
5. Choose highest probability class  

---

## Types of Naive Bayes

- **Gaussian Naive Bayes** → for continuous data  
- **Multinomial Naive Bayes** → for text data  
- **Bernoulli Naive Bayes** → for binary features  

---

## Advantages

- Very fast  
- Works well for text classification  
- Simple to implement  
- Requires less data  

---

## Disadvantages

- Assumes features are independent  
- Less accurate if features are highly related  

---

## Real-world Uses

- Spam detection  
- Sentiment analysis  
- Document classification  
- Recommendation systems  

---
---
## Neural Network

### Definition
A Neural Network is a supervised learning algorithm inspired by the human brain.  
It consists of layers of interconnected nodes (neurons) that learn patterns from data and make predictions.

Neural networks are widely used in **deep learning**, especially for images, speech, and text.

---

### Basic Structure

A neural network has three main layers:

- Input Layer → receives data  
- Hidden Layer(s) → processes data  
- Output Layer → gives prediction  

---

### Diagram

```
Input Layer   Hidden Layer    Output Layer
  x1  ──┐
  x2  ──┼──►  ( ○  ○  ○ )  ───►   ( ○ )
  x3  ──┘
```
x1, x2, x3 = input features  
Hidden layer = learning patterns  
Output = prediction 

## How It Works

1. Input data enters the network  
2. Each neuron multiplies input by weights  
3. Values pass through activation function  
4. Data moves layer by layer  
5. Output is generated  
6. Error is calculated  
7. Weights are updated  
8. Model improves over time  

---

## Example: Fit or Not Fit

**Inputs:**
- Exercise hours  
- Diet score  
- Sleep hours  

Neural network processes these values and predicts:

**Output → Fit or Not Fit**

---

## Activation Functions

Activation functions decide whether a neuron should activate.

**Common ones:**
- Sigmoid  
- ReLU  
- Tanh  

---

## Types of Neural Networks

- **ANN (Artificial Neural Network)**  
- **CNN (Convolutional Neural Network)** → images  
- **RNN (Recurrent Neural Network)** → sequence data  
- **LSTM** → time series  

---

## Advantages

- Handles complex data  
- Very accurate  
- Learns deep patterns  
- Works well for large datasets  

---

## Disadvantages

- Needs lots of data  
- Requires high computation  
- Hard to interpret  
- Takes time to train  

---

## Real-world Uses

- Image recognition  
- Speech recognition  
- Chatbots  
- Recommendation systems  
- Medical diagnosis  


---
---
---

## Regression

### Definition
Regression is a type of supervised learning used to predict **continuous numeric values**.  
Instead of predicting categories (like Yes/No), regression predicts numbers.

Examples:
- House price prediction  
- Salary prediction  
- Temperature prediction  

---

### Simple Idea

Input data → Model → Predict numeric value

Example:  
Area of house → Model → Price of house

---

### Example

Input:
- Study hours  

Output:
- Exam score  

If a student studies 5 hours → model predicts score = 72

---

### Regression Flow

```
        Input Features
             |
        Train Model
             |
      Learn Patterns
             |
        New Input
             |
       Predict Value
             |
        Numeric Output
```
---
---

## Types of Regression

Regression is a supervised learning method used to predict **continuous numeric values** such as price, salary, or temperature.

### Main Types of Regression

1. **Linear Regression**  
   Predicts output using a straight-line relationship between input and output.

2. **Multiple Linear Regression**  
   Uses multiple input features to predict one numeric output.

3. **Polynomial Regression**  
   Used when the relationship between variables is curved.

4. **Ridge Regression**  
   Adds a penalty to reduce overfitting and improve model stability.

5. **Lasso Regression**  
   Similar to Ridge but can remove less important features.

6. **Elastic Net Regression**  
   Combination of Ridge and Lasso regression.

7. **Decision Tree Regression**  
   Uses a tree structure to predict numeric values.

8. **Random Forest Regression**  
   Uses multiple trees and averages their predictions for better accuracy.

---
---
### 1. Linear Regression (Deep Explanation)

Linear Regression is one of the most important and basic algorithms in machine learning.  
It is used to predict a **continuous numeric value** by finding the best straight line that fits the data.

The goal is to model the relationship between input (X) and output (Y).

---

### Mathematical Equation

```y = mx + b```

Where:  
- y → predicted value  
- x → input feature  
- m → slope (how much y changes when x changes)  
- b → intercept (value of y when x = 0)

For multiple features:

```y = b0 + b1x1 + b2x2 + b3x3 + ...```

---

### Goal of Linear Regression

The model tries to find the **best-fit line** that minimizes the error between predicted values and actual values.

This error is measured using:

**Mean Squared Error (MSE)**

```MSE = average of (actual − predicted)²```

The model adjusts slope and intercept to reduce this error.

---

### How Training Works

1. Start with random line  
2. Predict values  
3. Calculate error  
4. Adjust slope and intercept  
5. Repeat until error is minimum  

This process uses **Gradient Descent**.

---

### Gradient Descent Idea

- If error is high → adjust line  
- Move line closer to data  
- Stop when error is minimum  

The best line = model learned.

---

### Visual Idea
```
Actual data points:   *
                    *
                 *
              *
Best-fit line:  ----------
```
The line passes through the center of points.

## Assumptions of Linear Regression

- Linear relationship between X and Y  
- No strong outliers  
- Data should be independent  
- Constant variance (homoscedasticity)  
- No high correlation between features (for multiple regression)  

---

## Types

- **Simple Linear Regression** → one input  
- **Multiple Linear Regression** → many inputs  

---

## Example

Predict salary using experience:

**Experience (years) → Salary**

If experience increases, salary increases → linear relationship.

---

## Advantages

- Easy to understand  
- Fast training  
- Interpretable  
- Works well for linear data  

---

## Disadvantages

- Cannot capture complex patterns  
- Sensitive to outliers  
- Assumes linear relation  
- Underfits complex data  

---

## Real-world Uses

- House price prediction  
- Sales forecasting  
- Stock trend estimation  
- Demand prediction  
---
---
### 2. Multiple Linear Regression

**Definition:**  
Multiple Linear Regression is an extension of linear regression that uses **more than one input feature** to predict a continuous output value.

Instead of one variable, the model learns from multiple variables together.

---

### Mathematical Equation

```y = b0 + b1x1 + b2x2 + b3x3 + ... + bnxn```

Where:  
- y → predicted value  
- x1, x2, x3 → input features  
- b0 → intercept  
- b1, b2, b3 → coefficients  

Each feature contributes to the final prediction.

---

### Example

Predict house price using:
- Area  
- Number of rooms  
- Location score  
- Age of house  

All features together help predict the final price.

---

### Visual Idea
```
Price
  |
  |      *
  |   *
  | *
  |________________
      Area + Rooms + Location
```
Instead of one line, the model forms a plane in higher dimensions.

## How Training Works

1. Input multiple features  
2. Model assigns weights to each feature  
3. Predict output  
4. Calculate error  
5. Update weights  
6. Repeat until error is minimum  

---

## Why Use Multiple Regression?

Real-world problems depend on many factors.

**Example:**  
Salary depends on:
- Experience  
- Skills  
- Education  
- Location  

Using only one feature gives poor prediction.

---

## Assumptions

- Linear relationship between features and output  
- No strong multicollinearity  
- Independent observations  
- Constant variance  

---

## Advantages

- More accurate than simple regression  
- Uses multiple factors  
- Better real-world modeling  
- Interpretable  

---

## Disadvantages

- Sensitive to correlated features  
- Needs clean data  
- Can overfit if too many features  

---

## Real-world Uses

- House price prediction  
- Salary prediction  
- Sales forecasting  
- Demand prediction  

---
---
### 3. Polynomial Regression

**Definition:**  
Polynomial Regression is a type of regression used when the relationship between input and output is **not a straight line** but a **curve**.

It extends linear regression by adding powers of the input variable (x², x³, etc.) to capture non-linear patterns.

---

### Mathematical Equation

```y = b0 + b1x + b2x² + b3x³ + ... + bnxⁿ```

Even though it models curves, it is still called a form of linear regression because it is linear in the coefficients.

---

### When to Use

Use polynomial regression when data follows a curved trend.

Example:
- Temperature vs time  
- Population growth  
- Sales trend over time  

---

### Visual Idea
```
Value
  |
  |        *
  |     *
  |   *
  |     *
  |        *
  |________________
```
Instead of a straight line, the model fits a curve through the data.

## How It Works

1. Take input feature (x)  
2. Create new features (x², x³, etc.)  
3. Train linear regression on these features  
4. Model learns curved relationship  
5. Predict output  

---

## Example

Predict marks based on study hours.

If studying more initially increases marks quickly and later slowly,  
the pattern becomes curved → polynomial regression fits better.

---

## Degree of Polynomial

- Degree 2 → quadratic curve  
- Degree 3 → cubic curve  
- Higher degree → more complex curve  

Higher degree = more flexible but risk of overfitting.

---

## Advantages

- Captures non-linear patterns  
- More flexible than linear regression  
- Better fit for curved data  

---

## Disadvantages

- Can overfit with high degree  
- Sensitive to outliers  
- Harder to interpret  
- Needs careful degree selection  

---

## Real-world Uses

- Growth prediction  
- Trend analysis  
- Sales forecasting  
- Scientific data modeling  
---
---
### 4. Ridge Regression (L2 Regularization)

**Definition:**  
Ridge Regression is a type of linear regression that adds a **penalty term** to reduce overfitting.  
It is used when the model becomes too complex or when features are highly correlated.

It helps make the model more stable and prevents very large coefficient values.

---

### Why Ridge Regression?

In multiple linear regression:
- Too many features  
- Highly correlated features  
- Large coefficients  

This causes **overfitting**.

Ridge regression solves this by adding a penalty.

---

### Formula

```Loss = MSE + λ(Σw²)```

Where:  
- MSE → Mean Squared Error  
- λ (lambda) → regularization parameter  
- w → coefficients  

The model tries to minimize both:
- prediction error  
- size of coefficients  

---

### Key Idea

Ridge regression **shrinks coefficients** but does not make them zero.

This keeps all features but reduces their impact.

---

### Visual Idea

Without Ridge → very steep line  
With Ridge → smoother line  
```
Data points:   *
             *
          *
       *

Without Ridge → Overfit line  
With Ridge → Smooth line
```

## When to Use

- Many features  
- Multicollinearity present  
- Overfitting problem  
- Want stable model  

---

## Effect of Lambda (λ)

- λ = 0 → same as linear regression  
- Small λ → slight regularization  
- Large λ → strong regularization  
- Too large λ → underfitting  

---

## Advantages

- Reduces overfitting  
- Handles multicollinearity  
- Improves model stability  
- Works well with many features  

---

## Disadvantages

- Does not remove features  
- Need to tune λ  
- Harder to interpret  

---

## Real-world Uses

- House price prediction  
- Stock prediction  
- Medical data modeling  
- Any high-feature dataset  
---
---
### 5. Lasso Regression (L1 Regularization)

**Definition:**  
Lasso Regression is a type of linear regression that adds a **penalty term** to reduce overfitting and can also perform **feature selection**.

It is similar to Ridge Regression but has one major difference:  
it can shrink some coefficients all the way to **zero**, effectively removing those features from the model.

---

### Why Lasso Regression?

When we have:
- Too many features  
- Irrelevant features  
- Overfitting  

Lasso helps by keeping only the most important features.

---

### Formula

```Loss = MSE + λ(Σ|w|)```

Where:  
- MSE → Mean Squared Error  
- λ → regularization parameter  
- |w| → absolute value of coefficients  

This is called **L1 regularization**.

---

### Key Idea

- Shrinks coefficients  
- Some coefficients become **zero**  
- Removes unnecessary features  

So Lasso performs **automatic feature selection**.

---

### Ridge vs Lasso

| Ridge | Lasso |
|------|------|
| Shrinks coefficients | Shrinks + removes |
| Uses L2 penalty | Uses L1 penalty |
| Keeps all features | Removes some features |

---

### Visual Idea
```
Before Lasso:
x1 = 2.3  
x2 = 1.8  
x3 = 0.9  

After Lasso:
x1 = 2.0  
x2 = 0  
x3 = 0.5  

Feature x2 removed.
```

## When to Use

- Too many features  
- Need feature selection  
- High-dimensional data  
- Prevent overfitting  

---

## Effect of Lambda (λ)

- Small λ → little change  
- Large λ → more coefficients become zero  
- Too large λ → underfitting  

---

## Advantages

- Reduces overfitting  
- Removes useless features  
- Simple and effective  
- Improves model interpretability  

---

## Disadvantages

- Can remove useful features if λ too high  
- Needs tuning  
- Not ideal when all features are important  

---

## Real-world Uses

- Feature selection  
- Medical data  
- Finance prediction  
- Text analysis  

---
---
### 6. Elastic Net Regression

**Definition:**  
Elastic Net Regression is a combination of **Ridge Regression** and **Lasso Regression**.  
It uses both L1 and L2 regularization to improve prediction and handle complex datasets.

It is useful when:
- There are many features  
- Features are correlated  
- Need feature selection + stability  

---

### Why Elastic Net?

Ridge:
- Keeps all features  
- Reduces overfitting  

Lasso:
- Removes some features  
- Performs feature selection  

Elastic Net:
- Does both  

---

### Formula

```Loss = MSE + λ1(Σ|w|) + λ2(Σw²)```

Where:  
- λ1 → Lasso penalty (L1)  
- λ2 → Ridge penalty (L2)  

This helps balance:
- Feature selection  
- Model stability  

---

### Key Idea

- Shrinks coefficients  
- Removes some features  
- Handles correlated features better  
- More flexible than Ridge or Lasso alone  

---

### When to Use

- Many features  
- Multicollinearity present  
- Need feature selection  
- Complex datasets  

---

### Visual Idea

```
All features → Model
        ↓
Some coefficients shrink
Some become zero
        ↓
Final stable model
```
## Advantages

- Combines Ridge and Lasso  
- Handles multicollinearity  
- Performs feature selection  
- More stable predictions  

---

## Disadvantages

- Needs tuning of two parameters  
- Slightly more complex  
- Slower than simple regression  

---

## Real-world Uses

- Finance prediction  
- Medical data  
- High-dimensional datasets  
- Stock prediction  

---
---
> **Note:** Decision Tree and Random Forest were already explained in the **classification** section.  
> Below is their **regression version**.

---

## Decision Tree Regression
Uses a tree structure to predict **numeric values** instead of classes.

---

## Random Forest Regression
Uses multiple decision trees and **averages their predictions** to improve accuracy and reduce overfitting.

---
---
---
# Unsupervised Learning

### Definition
Unsupervised learning is a type of machine learning where the model is trained on **unlabeled data**.  
There is no correct output given to the model. Instead, the model tries to find hidden patterns, relationships, or groups in the data on its own.

---

### Key Idea
The model explores the data and discovers structure without being told the answers.

Example:  
Grouping customers based on shopping behavior without knowing their categories.

---

### Difference from Supervised Learning

| Supervised Learning | Unsupervised Learning |
|--------------------|----------------------|
| Uses labeled data | Uses unlabeled data |
| Predicts output | Finds patterns |
| Example: spam detection | Example: customer grouping |

---

### Main Goals

- Find similar groups in data  
- Detect patterns  
- Reduce features  
- Discover hidden structure  

---

### Common Tasks

- Clustering  
- Dimensionality reduction  
- Association rules  

Unsupervised learning is mainly used for **data exploration and pattern discovery**.

---
## Types of Unsupervised Learning

Unsupervised learning mainly has three types:

### 1. Clustering
Clustering groups similar data points together based on their features.

Example:
Grouping customers into different categories based on spending habits.

Common algorithms:
- K-Means  
- Hierarchical Clustering  
- DBSCAN  

---

### 2. Dimensionality Reduction
Dimensionality reduction reduces the number of features while keeping important information.

Example:
Reducing 100 features to 10 important features.

Common algorithms:
- PCA (Principal Component Analysis)  
- t-SNE  

---

### 3. Association Rule Learning
Association rule learning finds relationships between items in data.

Example:
If a customer buys bread, they may also buy butter.

Common algorithms:
- Apriori  
- FP-Growth  

---
---
### 1. Clustering

**Definition:**  
Clustering is an unsupervised learning technique used to group similar data points together based on their features.  
The model automatically finds patterns and forms clusters without any labels.

---

### Simple Idea

Data points that are similar → same group  
Data points that are different → different groups  

---

### Example

Customer segmentation in a mall.

Customers with similar:
- income  
- spending  
- age  

are grouped into the same cluster.

---

### Diagram

```
   ● ● ●        Cluster 1
 ● ●

          ● ● ●       Cluster 2
        ●

  ● ● ●            Cluster 3
```
Each group represents similar data.

## How Clustering Works

1. Take input data  
2. Measure similarity  
3. Group similar points  
4. Form clusters  

No labels are given — model discovers groups itself.

---

## Common Clustering Algorithms

- **K-Means**  
- **Hierarchical Clustering**  
- **DBSCAN**  

---

## Real-world Uses

- Customer segmentation  
- Recommendation systems  
- Image grouping  
- Fraud detection  

---
---
### Common Clustering Algorithms

1. **K-Means Clustering**  
   Groups data into K clusters based on similarity.  
   Most widely used clustering algorithm.

2. **Hierarchical Clustering**  
   Builds a tree of clusters.  
   Shows relationships between data points.

3. **DBSCAN**  
   Groups data based on density.  
   Can find irregular-shaped clusters and detect outliers.

These are the most commonly used clustering algorithms in unsupervised learning.

---
---
## 1. K-Means Clustering

### Definition
K-Means is an unsupervised learning algorithm used to group data into **K clusters** based on similarity.

It divides the data into K groups where each data point belongs to the nearest cluster center.

---

### Simple Idea

Points that are close to each other → same cluster  
Points far apart → different clusters  

---

### How K-Means Works

1. Choose number of clusters (K)  
2. Select random centroids  
3. Assign each point to nearest centroid  
4. Update centroids  
5. Repeat until clusters stop changing  

---

### Diagram

```
Cluster 1        Cluster 2        Cluster 3
 ● ● ●            ● ●              ● ● ●
 ● ●               ●               ●
```
Each cluster has a center called centroid.

## Example

**Customer segmentation:**

Customers grouped by:
- Spending  
- Income  

Cluster 1 → High spenders  
Cluster 2 → Medium  
Cluster 3 → Low  

---

## Choosing K

We choose number of clusters manually.  
Common method: **Elbow Method**

---

## Advantages

- Simple  
- Fast  
- Easy to implement  
- Works well for large data  

---

## Disadvantages

- Need to choose K  
- Sensitive to outliers  
- Works best with circular clusters  

---

## Real-world Uses

- Customer segmentation  
- Market analysis  
- Image compression  
- Recommendation systems  

---
## 2. Hierarchical Clustering

### Definition
Hierarchical Clustering is an unsupervised learning algorithm that groups data into clusters by creating a **tree-like structure** called a dendrogram.

It does not require choosing the number of clusters at the start.

---

### Types of Hierarchical Clustering

1. **Agglomerative (Bottom-Up)**  
   Start with each point as its own cluster → merge closest clusters step by step.

2. **Divisive (Top-Down)**  
   Start with all data in one cluster → split into smaller clusters.

Most commonly used: **Agglomerative**.

---

### How It Works

1. Treat each data point as a cluster  
2. Find closest clusters  
3. Merge them  
4. Repeat until all points are in one cluster  
5. Use dendrogram to decide number of clusters  

---

### Diagram (Dendrogram)

```
        _________
       |         |
     ___       ___
    |   |     |   |
   A    B    C    D
```
Cutting the tree at a certain level gives clusters.

## Example

Grouping students based on:
- Marks  
- Attendance  
- Behavior  

Similar students are merged into clusters.

---

## Advantages

- No need to choose K initially  
- Shows hierarchy  
- Good for small datasets  
- Easy to visualize  

---

## Disadvantages

- Slow for large datasets  
- Hard to scale  
- Sensitive to noise  

---

## Real-world Uses

- Gene analysis  
- Document clustering  
- Customer segmentation  
- Social network analysis  

---
## 3. DBSCAN (Density-Based Clustering)

### Definition
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised learning algorithm that groups data points based on **density**.

It forms clusters where data points are closely packed together and marks points in low-density areas as **outliers**.

---

### Key Idea

- Dense area → cluster  
- Sparse area → noise/outlier  

Unlike K-Means, DBSCAN can find clusters of **any shape**.

---

### Important Terms

- **Core Point:** has many neighbors nearby  
- **Border Point:** near a cluster  
- **Noise Point:** outlier  

---

### How DBSCAN Works

1. Choose two parameters:
   - eps → distance radius  
   - minPts → minimum points in area  

2. Pick a point  
3. Find neighbors within eps  
4. If enough neighbors → form cluster  
5. Expand cluster  
6. Points with no cluster → noise  

---

### Diagram

```
Cluster 1        Cluster 2

 ● ● ●            ● ● ●
 ● ●               ●
 ●

      x       x                      x  ← Noise (outliers)
```

## Example

**Detect fraud transactions:**

- Normal transactions → clusters  
- Fraud transactions → outliers  

---

## Advantages

- Finds clusters of any shape  
- Detects outliers  
- No need to choose number of clusters  
- Works well with noisy data  

---

## Disadvantages

- Hard to choose parameters  
- Not good for varying density  
- Slower on very large data  

---

## Real-world Uses

- Fraud detection  
- Anomaly detection  
- GPS location clustering  
- Image processing  


---
---
---
# Reinforcement Learning

### Definition
Reinforcement Learning (RL) is a type of machine learning where an **agent learns by interacting with an environment** and receiving rewards or penalties.

The goal is to learn the best actions to maximize total reward over time.

It works like learning through trial and error.

---

### Simple Idea

Agent → takes action → gets reward → learns → improves

Like training a dog:
- Good action → reward  
- Bad action → punishment  
- Dog learns correct behavior  

---

### Key Components

1. **Agent** → learner/decision maker  
2. **Environment** → where agent acts  
3. **Action** → what agent does  
4. **Reward** → feedback  
5. **State** → current situation  

---

### Flow Diagram

```
        Agent
          |
       Takes Action
          |
      Environment
          |
       Reward/Penalty
          |
       Learns Policy
          |
        Improves
```

## Example

**Game playing:**

- Agent = player  
- Environment = game  
- Action = move  
- Reward = score  

If agent wins → reward  
If agent loses → penalty  

Over time, agent learns best moves.

---

## Real-life Example

**Self-driving car:**

- Agent → car  
- Environment → road  
- Action → turn / accelerate  
- Reward → safe driving  

---

## Types of Reinforcement Learning

- **Q-Learning**  
  Learns value of actions  

- **Deep Q Learning (DQN)**  
  Uses neural networks  

- **Policy Gradient**  
  Learns directly from rewards  

---

## Advantages

- Learns from experience  
- Good for dynamic environments  
- No labeled data needed  
- Improves over time  

---

## Disadvantages

- Needs lots of training  
- Slow learning  
- Complex tuning  
- High computation  

---

## Real-world Uses

- Game AI (Chess, AlphaGo)  
- Robotics  
- Self-driving cars  
- Recommendation systems  
- Stock trading  



