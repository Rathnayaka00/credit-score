import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

model_path = 'D:\All Projects\Intern\ML\To Upload\Task 1\model\model.pkl' 
with open(model_path, 'rb') as file:
    model = pickle.load(file)


st.title('Credit Score Prediction')


annual_income = st.number_input('Annual Income', min_value=0, value=50000)
num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=0, value=2)
num_credit_card = st.number_input('Number of Credit Cards', min_value=0, value=3)
interest_rate = st.number_input('Interest Rate (%)', min_value=0.0, max_value=100.0, value=15.0)
num_of_loan = st.number_input('Number of Loans', min_value=0, value=1)
delay_from_due_date = st.number_input('Delay from Due Date (days)', min_value=0, value=5)
num_of_delayed_payment = st.number_input('Number of Delayed Payments', min_value=0, value=1)
credit_mix = st.selectbox('Credit Mix', ['Good', 'Fair', 'Bad'], index=0)  # Example categories
outstanding_debt = st.number_input('Outstanding Debt', min_value=0, value=10000)
credit_history_age = st.number_input('Credit History Age (years)', min_value=0, value=5)
monthly_balance = st.number_input('Monthly Balance', min_value=0, value=2000)


credit_mix_encoder = LabelEncoder()
credit_mix_encoded = credit_mix_encoder.fit_transform([credit_mix])[0]


input_data = pd.DataFrame({
    'Annual_Income': [annual_income],
    'Num_Bank_Accounts': [num_bank_accounts],
    'Num_Credit_Card': [num_credit_card],
    'Interest_Rate': [interest_rate],
    'Num_of_Loan': [num_of_loan],
    'Delay_from_due_date': [delay_from_due_date],
    'Num_of_Delayed_Payment': [num_of_delayed_payment],
    'Credit_Mix': [credit_mix_encoded],
    'Outstanding_Debt': [outstanding_debt],
    'Credit_History_Age': [credit_history_age],
    'Monthly_Balance': [monthly_balance]
})


if st.button('Predict Credit Score'):
    try:
        prediction = model.predict(input_data)
        st.write(f"Predicted Credit Score: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")
