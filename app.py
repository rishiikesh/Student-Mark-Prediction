from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

file_path = "C:/Users/rishi/OneDrive/Desktop/Creations/attendance_data.xlsx"  
df = pd.read_excel(file_path)

# Assuming you want to predict 'marks' based on all features
X = df[['Attendance', 'Marks_Sem1', 'Marks_Sem2', 'Marks_Sem3', 'Marks_Sem4', 'Marks_Sem5', 'Hours_Studied']]
y = df['marks']
roll_numbers = df['roll_number']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, roll_numbers_train, roll_numbers_test = train_test_split(
    X, y, roll_numbers, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor()

linear_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Function to evaluate model metrics
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    return mae, mse, r_squared

# Function to get feature importance
def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        data = {'Feature': feature_names, 'Importance': importance}
        df_importance = pd.DataFrame(data)
        return px.bar(df_importance, x='Feature', y='Importance', title='Feature Importance')
    else:
        return None

# Function to predict marks for a given roll number
def predict_marks_for_roll_number(roll_number, models, X, roll_numbers):
    if roll_number in roll_numbers.values:
        features = X[roll_numbers == roll_number]

        linear_prediction = models[0].predict(features)[0]
        decision_tree_prediction = models[1].predict(features)[0]
        random_forest_prediction = models[2].predict(features)[0]

        return linear_prediction, decision_tree_prediction, random_forest_prediction
    else:
        return None

# Function to generate main plot
def get_prediction_plots(roll_number, predictions, model_names):
    num_plots = len(model_names)

    # Create a bar chart for each model's prediction
    fig = px.bar(x=model_names, y=predictions, labels={'x': 'Models', 'y': 'Predicted Marks'})
    fig.update_layout(title=f'Predicted Marks for Roll Number {roll_number}', barmode='group')
    plot_div = fig.to_html(full_html=False)

    return plot_div

# Function to generate feature plots
def get_feature_plots(X):
    num_plots = X.shape[1]

    # Create a histogram for each feature
    fig = px.histogram(X, x=X.columns, nbins=30, marginal='box', labels={'x': 'Features', 'y': 'Count'})
    fig.update_layout(title='Distribution of Features', height=800)
    plot_div = fig.to_html(full_html=False)

    return plot_div

# Function to render the student overview page
@app.route('/student/<int:roll_number>')
def student_overview(roll_number):
    if roll_number in roll_numbers.values:
        student_data = df[df['roll_number'] == roll_number]
        return render_template('student_overview.html', roll_number=roll_number, student_data=student_data)
    else:
        return render_template('not_found.html', roll_number=roll_number)

# Function to render individual model prediction page
@app.route('/model/<int:roll_number>/<model_name>')
def individual_model_prediction(roll_number, model_name):
    models_dict = {'linear': linear_model, 'decision_tree': decision_tree_model, 'random_forest': random_forest_model}
    
    if roll_number in roll_numbers.values and model_name in models_dict:
        model = models_dict[model_name]
        features = X[roll_numbers == roll_number]

        prediction = model.predict(features)[0]

        return render_template('individual_model_prediction.html', 
                               roll_number=roll_number,
                               model_name=model_name.capitalize(),
                               prediction=prediction)
    else:
        return render_template('not_found.html', roll_number=roll_number)

# Function to render the home page
@app.route('/')
def home():
    return render_template('index.html')

# Function to handle the prediction form submission
@app.route('/predict', methods=['POST'])
def predict():
    roll_number = int(request.form['roll_number'])

    predictions = predict_marks_for_roll_number(roll_number, [linear_model, decision_tree_model, random_forest_model], X, roll_numbers)

    if predictions:
        linear_prediction, decision_tree_prediction, random_forest_prediction = predictions
        model_names = ['Linear Regression', 'Decision Tree', 'Random Forest']

        # Model evaluation metrics
        mae, mse, r_squared = evaluate_model(y_test, random_forest_model.predict(X_test))

        # Feature importance for Random Forest
        feature_importance_plot = get_feature_importance(random_forest_model, X.columns)

        # Generate the main plot
        main_plot = get_prediction_plots(roll_number, [linear_prediction, decision_tree_prediction, random_forest_prediction], model_names)

        # Generate feature plots
        feature_plots = get_feature_plots(X)

        return render_template('result.html', 
                               roll_number=roll_number,
                               linear_prediction=linear_prediction,
                               decision_tree_prediction=decision_tree_prediction,
                               random_forest_prediction=random_forest_prediction,
                               mae=mae,
                               mse=mse,
                               r_squared=r_squared,
                               feature_importance_plot=feature_importance_plot,
                               main_plot=main_plot,
                               feature_plots=feature_plots)
    else:
        return render_template('not_found.html', roll_number=roll_number)

if __name__ == '__main__':
    app.run(debug=True)
