from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

app = Flask(__name__)

# Generate synthetic data with study hours and attendance
file_path = "C:/Users/rishi/OneDrive/Desktop/Creations/attendance_data.xlsx"  # Replace with the actual file path
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

def predict_marks_for_roll_number(roll_number, models, X, roll_numbers):
    if roll_number in roll_numbers.values:
        features = X[roll_numbers == roll_number]

        linear_prediction = models[0].predict(features)[0]
        decision_tree_prediction = models[1].predict(features)[0]
        random_forest_prediction = models[2].predict(features)[0]

        return linear_prediction, decision_tree_prediction, random_forest_prediction
    else:
        return None

def get_prediction_plots(roll_number, predictions, model_names):
    num_plots = len(model_names)

    # Create a bar chart for each model's prediction
    fig = px.bar(x=model_names, y=predictions, labels={'x': 'Models', 'y': 'Predicted Marks'})
    fig.update_layout(title=f'Predicted Marks for Roll Number {roll_number}', barmode='group')
    plot_div = fig.to_html(full_html=False)

    return plot_div

def get_feature_plots(X):
    num_plots = X.shape[1]

    # Create a histogram for each feature
    fig = px.histogram(X, x=X.columns, nbins=30, marginal='box', labels={'x': 'Features', 'y': 'Count'})
    fig.update_layout(title='Distribution of Features', height=800)
    plot_div = fig.to_html(full_html=False)

    return plot_div

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    roll_number = int(request.form['roll_number'])

    predictions = predict_marks_for_roll_number(roll_number, [linear_model, decision_tree_model, random_forest_model], X, roll_numbers)

    if predictions:
        linear_prediction, decision_tree_prediction, random_forest_prediction = predictions
        model_names = ['Linear Regression', 'Decision Tree', 'Random Forest']

        # Generate the main plot
        main_plot = get_prediction_plots(roll_number, [linear_prediction, decision_tree_prediction, random_forest_prediction], model_names)

        # Generate feature plots
        feature_plots = get_feature_plots(X)

        return render_template('result.html', 
                               roll_number=roll_number,
                               linear_prediction=linear_prediction,
                               decision_tree_prediction=decision_tree_prediction,
                               random_forest_prediction=random_forest_prediction,
                               main_plot=main_plot,
                               feature_plots=feature_plots)
    else:
        return render_template('not_found.html', roll_number=roll_number)

if __name__ == '__main__':
    app.run(debug=True)
