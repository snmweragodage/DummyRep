# app.py (with password support + forgot password feature)
import os
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from torchvision import transforms
from datetime import datetime
import random
import string

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///saved_results.db'
db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    age = db.Column(db.Integer)
    dob = db.Column(db.String(20))
    password = db.Column(db.String(120))

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    filename = db.Column(db.String(120))
    result = db.Column(db.String(50))
    cancer_type = db.Column(db.String(50))
    risk = db.Column(db.String(50))
    percentage = db.Column(db.String(10))
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.drop_all()
    db.create_all()
    print(">>> Database DROPPED and RECREATED with new schema.")
    print(">>> Database created at:", os.path.abspath("saved_results.db"))

# CNN model definition
class SkinCancerCNN(nn.Module):
    def __init__(self):
        super(SkinCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = SkinCancerCNN().to(device)
model2 = SkinCancerCNN().to(device)
model1.load_state_dict(torch.load('model_stage1.pt', map_location=device))
model2.load_state_dict(torch.load('model_stage2.pt', map_location=device))
model1.eval()
model2.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            session['username'] = username
            return redirect(url_for('home'))
        return "Invalid username or password."
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        dob = request.form['dob']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            return "Username exists."

        user = User(username=username, email=email, phone=phone, gender=gender, age=age, dob=dob, password=password)
        db.session.add(user)
        db.session.commit()
        session['username'] = username
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        user = User.query.filter_by(username=username).first()
        if not user:
            return "No account found with that username."

        new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        user.password = new_password
        db.session.commit()

        return f"Your new temporary password is: <strong>{new_password}</strong><br><a href='/'>Go to Login</a>"

    return render_template('forgot_password.html')

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        return redirect(url_for('predict', filename=file.filename))
    return render_template('upload.html')

@app.route('/predict/<filename>')
def predict(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    cancer_output = model1(image)
    cancer_prob = cancer_output.item()
    cancer_pred = (cancer_output > 0.5).float()
    result, cancer_type, risk, percentage = 'Non-Cancerous', 'N/A', 'Low', f"{(1 - cancer_prob) * 100:.0f}%"
    if cancer_pred.item() == 1:
        subtype_output = model2(image)
        subtype_pred = (subtype_output > 0.5).float()
        result = 'Cancerous'
        cancer_type = 'Melanoma' if subtype_pred.item() == 1 else 'Non-Melanoma'
        risk = 'High' if cancer_type == 'Melanoma' else 'Medium'
        percentage = f"{cancer_prob * 100:.0f}%"
    r = Result(username=session['username'], filename=filename, result=result, cancer_type=cancer_type, risk=risk, percentage=percentage)
    db.session.add(r)
    db.session.commit()
    return render_template('result.html', result=result, cancer_type=cancer_type, risk=risk, img_path='static/uploads/' + filename, percentage=percentage)

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    results = Result.query.filter_by(username=session['username']).order_by(Result.upload_time.desc()).all()
    return render_template('history.html', results=results)

@app.route('/profile')
def profile():
    user = User.query.filter_by(username=session['username']).first()
    return render_template('user.html', user=user)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
