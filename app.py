import pandas as pd
from flask import Flask,render_template,request
from joblib import load
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split

app = Flask(__name__)

dataset = 'Dataset.csv'

df = pd.read_csv(dataset)

x = df.drop('Class' , axis = 1 )
y = df['Class']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)

model_name_1 = "G:\\Projects\\venn_project\\LogisticRegression_model.joblib"
labeled_model = load(model_name_1)
# Load Random Forest Classifier model
model_name_2 = "G:\\Projects\\venn_project\\RandomForestClassifier_model.joblib"
cluster_model = load(model_name_2)


def plot_graph(predictions, model_name):
    plt.figure(figsize=(8, 6))
    plt.hist(predictions, bins=10, edgecolor='black')
    plt.title(f'{model_name} Predictions Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Convert plot to image for rendering in Flask
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url


@app.route('/')
def home():
    lr_preds = labeled_model.predict(x_test)
    rf_preds = cluster_model.predict(x_test)
    
    lr_accuracy = accuracy_score(y_test, lr_preds)
    rf_accuracy = accuracy_score(y_test, rf_preds)
    
    lr_classification_report = classification_report(y_test, lr_preds)
    rf_classification_report = classification_report(y_test, rf_preds)
    
    lr_plot = plot_graph(lr_preds, 'Logistic Regression')
    rf_plot = plot_graph(rf_preds, 'Random Forest Classifier')
    
    return render_template('index.html', lr_accuracy=lr_accuracy, lr_report=lr_classification_report,
                           rf_accuracy=rf_accuracy, rf_report=rf_classification_report,
                           lr_plot=lr_plot, rf_plot=rf_plot)

if __name__ == '__main__':
    app.run(debug=True)