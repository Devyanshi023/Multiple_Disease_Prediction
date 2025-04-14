import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import random

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Getting the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/xgboost_diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction', 'Heart Disease Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart'],
                           default_index=0)

# ----------------- Diabetes Page -----------------
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Input Columns
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex = st.selectbox(
            'Sex',
            options=[0, 1],
            index=1,  # Default to Male
            format_func=lambda x: 'Female' if x == 0 else 'Male'
        )

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure')

    with col1:
        BMI = st.text_input('BMI')

    # Diabetes Pedigree Function input
    with col2:
        DiabetesPedigreeFunction = st.text_input(
            'Diabetes Pedigree Function',
            value=st.session_state.get('dpf_value', '')
        )

    with col3:
        Age = st.text_input('Age')

    # Pregnancies input (only if female)
    if Sex == 0:  # Female
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')
    else:
        Pregnancies = '0'  # Males default to 0

    # Predict button and result
    diab_diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            user_input = [
                float(Glucose),
                float(BloodPressure),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age),
                float(Pregnancies)
            ]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = '🔴 The person is diabetic'
            else:
                diab_diagnosis = '🟢 The person is not diabetic'

        except ValueError:
            diab_diagnosis = "⚠️ Please enter valid numeric values."

    st.success(diab_diagnosis)

    # --- Split Bottom Section into 2 Columns ---
    exp_col, dpf_col = st.columns([2, 1])

    # Column 1: Feature Explanations
    with exp_col:
        st.markdown("### 🧾 Explanation of Diabetes Features")
        st.markdown("""
        - **Glucose**: Plasma glucose concentration (normal fasting level is 70–100 mg/dL).
        - **BloodPressure**: Diastolic blood pressure in mm Hg (normal is < 80 mm Hg).
        - **BMI**: Body Mass Index = weight (kg) / height (m²), healthy range is 18.5–24.9.
        - **DiabetesPedigreeFunction**: A score indicating family history of diabetes.
        """)

    # Column 2: DPF Estimator
    with dpf_col:
        with st.expander("🧬 Estimate Your DPF (Family History)"):
            st.write("If you're unsure of your **Diabetes Pedigree Function**, use this guide to estimate it:")
            parent = st.checkbox("Parent(s) with diabetes")
            siblings = st.checkbox("Sibling(s) with diabetes")
            extended = st.checkbox("Grandparents, Aunts, or Uncles")
            multi_gen = st.checkbox("Multiple generations with diabetes")

            dpf_estimate = 0.1  # base
            if parent: dpf_estimate += 0.3
            if siblings: dpf_estimate += 0.2
            if extended: dpf_estimate += 0.1
            if multi_gen: dpf_estimate += 0.4

            dpf_estimate = round(min(dpf_estimate, 1.0), 3)
            st.markdown(f"**Estimated DPF: `{dpf_estimate}`**")

            if st.button("Use this DPF estimate"):
                st.session_state['dpf_value'] = dpf_estimate

    # --- Health Tips ---
    st.markdown("### 🍎 Healthy Lifestyle Tips for Diabetes")
    diabetes_tips = [
        "Eat a balanced diet rich in fiber and low in sugar.",
        "Monitor your blood glucose levels regularly.",
        "Stay physically active—aim for 150 minutes/week of moderate exercise.",
        "Avoid smoking and limit alcohol consumption.",
        "Stay well hydrated and reduce stress.",
        "Control portion sizes to avoid blood sugar spikes.",
        "Don’t skip meals—especially breakfast!"
    ]
    st.markdown("- " + "\n- ".join(random.sample(diabetes_tips, 3)))


# ----------------- Heart Disease Page -----------------
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1 = male, 0 = female)')
    with col3:
        cp = st.text_input('Chest Pain Type (0-3)')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = yes, 0 = no)')

    with col1:
        thal = st.text_input('Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable)')
    
    heart_diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(sex), float(exang), float(thal), float(cp),
                          float(thalach), float(age), float(trestbps)]

            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'
        except:
            heart_diagnosis = 'Please enter valid numerical inputs'

    st.success(heart_diagnosis)
    
    # Feature Explanation Section for Heart Disease
    st.markdown("### 🧾 Explanation of Heart Disease Features")
    st.markdown("""
    - **Sex**: 1 = Male, 0 = Female.
    - **cp (Chest Pain type)**: 
    - 0 = A common kind of chest pain that usually happens during physical activity or stress. It's often related to heart problems. 
    - 1 = Chest pain that feels different from the usual — might not always happen during activity and can be harder to link directly to the heart. 
    - 2 = Chest pain that's most likely not caused by the heart. It could be due to things like muscle strain or indigestion.
    - 3 = No chest pain.
    - **trestbps (Resting BP)**: Systolic blood pressure in mm Hg (normal is < 120 mm Hg).
    - **thalach (Max Heart Rate)**: Maximum heart rate achieved during exercise (normal ~120-200 bpm).
    - **exang (Exercise Induced Pain in chest)**: 1 = Yes (pain), 0 = No (no pain).
    - **thal (Thalassemia)**: 
    - 1 = Normal  
    - 2 = Fixed defect Fixed defect (old heart damage)
    - 3 = Reversible defect (possible heart issue under stress)
    """)

    # Heart Health Tips (randomized)
    st.markdown("### ❤️ Healthy Heart Tips")
    import random
    heart_tips = [
        "Exercise at least 30 minutes a day, 5 days a week.",
        "Limit saturated fats and sugar in your diet.",
        "Quit smoking and avoid secondhand smoke.",
        "Manage your stress levels through relaxation techniques.",
        "Get regular health screenings for blood pressure and cholesterol.",
        "Stay hydrated and get 7–8 hours of sleep daily.",
        "Maintain a healthy weight to reduce heart disease risk."
    ]
    st.markdown("- " + "\n- ".join(random.sample(heart_tips, 3)))
