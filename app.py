from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
from dotenv import load_dotenv
import os
import google.generativeai as genai
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

# Load disease prediction models
diabetes_model = pickle.load(open('models/diabetes1.pkl', 'rb'))
heart_model = pickle.load(open('models/HeartDiseaseModel.pkl', 'rb'))
parkinsons_model = pickle.load(open('models/parkinsons1.pkl', 'rb'))
breastcancer_model = pickle.load(open('models/svm_model1.pkl', 'rb'))



# Load the SVC model
svc = pickle.load(open('models/svc.pkl', 'rb'))

scaler = pickle.load(open('models/scaler.pkl', 'rb'))
with open('models/scalerforapark.pkl', 'rb') as scaler_file:
    scalerforpark = pickle.load(scaler_file) 
scalerforbc = pickle.load(open('models/scalerforbc.pkl','rb'))


load_dotenv()  # take environment variables from .env.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the GenerativeModel and chat
model = genai.GenerativeModel('gemini-pro')
chat_history = []

# Load symptom-related datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Symptoms dictionary and diseases list
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def get_gemini_response(question):
    response = model.generate_content(question)
    return response.text

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout


# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions, medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')


@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            user_input = [float(request.form.get('Pregnancies')),
                          float(request.form.get('Glucose')),
                          float(request.form.get('BloodPressure')),
                          float(request.form.get('SkinThickness')),
                          float(request.form.get('Insulin')),
                          float(request.form.get('BMI')),
                          float(request.form.get('DiabetesPedigreeFunction')),
                          float(request.form.get('Age'))]

            if any(np.isnan(user_input)) or any(np.isinf(user_input)):
                raise ValueError("Invalid input values. Please enter valid numerical values.")
            
            np.array(user_input).reshape(1, -1)
            scaled_input_values = scaler.transform([user_input])

            # Predict using the diabetes model
            diab_prediction = diabetes_model.predict(scaled_input_values)

            if diab_prediction[0] == 1:
                result = 'DigiHealth indicates a positive sign of Diabetes disease. Kindly visit a nearby hospital for treatment.'
            else:
                result = 'Congratulations!!! DigiHealth predicts you are not suffering from Diabetes disease.'

            return render_template('result.html', result=result)

        except (ValueError, TypeError) as e:
            print("Problematic input:", request.form)
            return render_template('error.html', error=str(e))

    return render_template('diabetes.html')


@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease():
    if request.method == 'POST':
        try:
            user_input = [float(request.form.get('age')),
                          float(request.form.get('sex')),
                          float(request.form.get('cp')),
                          float(request.form.get('trestbps')),
                          float(request.form.get('chol')),
                          float(request.form.get('fbs')),
                          float(request.form.get('restecg')),
                          float(request.form.get('thalach')),
                          float(request.form.get('exang')),
                          float(request.form.get('oldpeak')),
                          float(request.form.get('slope')),
                          float(request.form.get('ca')),
                          float(request.form.get('thal'))]

            if any(np.isnan(user_input)):
                raise ValueError("Invalid input values. Please enter valid numerical values.")

            heart_prediction = heart_model.predict(np.array(user_input).reshape(1, -1))

            if heart_prediction[0] == 1:
                result = 'DigiHealth indicates a positive sign of Heart disease. Kindly visit a nearby hospital for treatment.'
            else:
                result = 'Congratulations!!! DigiHealth predicts you are not suffering from Heart disease.'

            return render_template('result.html', result=result)

        except ValueError as e:
            return render_template('error.html', error=str(e))

    return render_template('heart_disease.html')


@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    if request.method == 'POST':
        try:
            user_input = [float(request.form.get('fo')),
                          float(request.form.get('fhi')),
                          float(request.form.get('flo')),
                          float(request.form.get('Jitter_percent')),
                          float(request.form.get('Jitter_Abs')),
                          float(request.form.get('RAP')),
                          float(request.form.get('PPQ')),
                          float(request.form.get('DDP')),
                          float(request.form.get('Shimmer')),
                          float(request.form.get('Shimmer_dB')),
                          float(request.form.get('APQ3')),
                          float(request.form.get('APQ5')),
                          float(request.form.get('APQ')),
                          float(request.form.get('DDA')),
                          float(request.form.get('NHR')),
                          float(request.form.get('HNR')),
                          float(request.form.get('RPDE')),
                          float(request.form.get('DFA')),
                          float(request.form.get('spread1')),
                          float(request.form.get('spread2')),
                          float(request.form.get('D2')),
                          float(request.form.get('PPE'))]

            if any(np.isnan(user_input)):
                raise ValueError("Invalid input values. Please enter valid numerical values.")
            
            user_input = np.array(user_input).reshape(1, -1)
            std_input_data = scalerforpark.transform(user_input)

            parkinsons_prediction = parkinsons_model.predict(std_input_data)

            if parkinsons_prediction[0] == 1:
                result = 'DigiHealth indicates a positive sign of Parkinsons disease. Kindly visit a nearby hospital for treatment.'
            else:
                result = 'Congratulations!!! DigiHealth predicts you are not suffering from Parkinsons disease.'

            return render_template('result.html', result=result)

        except ValueError as e:
            return render_template('error.html', error=str(e))

    return render_template('parkinsons.html')


@app.route('/breast_cancer', methods=['GET', 'POST'])
def breast_cancer():
    if request.method == 'POST':
        try:
            # Retrieve input values from the form
            user_input = [request.form.get(field) for field in [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean',
                'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                'smoothness_se', 'compactness_se', 'concavity_se',
                'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
                'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                'smoothness_worst', 'compactness_worst', 'concavity_worst',
                'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
            ]]
            
            # Convert input values to floats
            user_input = [float(value) if value is not None else 0.0 for value in user_input]

            # Check for NaN or infinite values
            if any(np.isnan(user_input)) or any(np.isinf(user_input)):
                raise ValueError("Invalid input values. Please enter valid numerical values.")

            # Make predictions using the breast cancer model
            np.asarray(user_input).reshape(1,-1)
            
            user_input = scalerforbc.transform([user_input])
            cancer_prediction = breastcancer_model.predict(user_input)

            if cancer_prediction[0] == 1:
                result = 'DigiHealth indicates a positive sign of Breast Cancer disease (the tumor is malignant (cancerous)). Kindly visit a nearby hospital for treatment.'
            else:
                result = 'DigiHealth predicts you are suffering from Breast Cancer disease (The tumor is benign (non-cancerous)).'

            return render_template('result.html', result=result)

        except (ValueError, TypeError) as e:
            return render_template('error.html', error=str(e))

    return render_template('breast_cancer.html')


@app.route('/gemini')
def gemini():
    return render_template('gemini.html', chat_history=chat_history)

@app.route('/gemini', methods=['POST'])
def ask_question():
    user_input = request.form['user_input']
    bot_response = get_gemini_response(user_input)
    chat_history.append(('User', user_input))
    chat_history.append(('Bot', bot_response))
    return render_template('gemini.html', chat_history=chat_history)

if __name__ == '__main__':
    app.run(debug=True)
