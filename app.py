# app.py
import streamlit as st
import pandas as pd
from processing import analyze_dataset, analysis_to_string, verify_with_llm

st.title('IntuiqAI - Dataset Analysis')

uploaded_file = st.file_uploader("Upload CSV dataset", type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('Dataset loaded successfully.')

    st.subheader('Dataset Preview')
    st.dataframe(df.head())

    analysis = analyze_dataset(df)
    analysis_str = analysis_to_string(df, analysis)

    st.subheader('Initial Analysis')
    analysis_df = pd.DataFrame.from_dict(analysis, orient='index')
    st.table(analysis_df)

    with st.spinner('Verifying analysis with LLM...'):
        verified = verify_with_llm(analysis_str)

    st.subheader('Verified Analysis by LLM')
    st.markdown(verified)

    with st.form(key='approval_form'):
        st.write('Review the verified analysis and suggestions above.')
        approve = st.checkbox('Approve the analysis and suggested changes')
        submit = st.form_submit_button('Submit Decision')

        if submit:
            if approve:
                st.success('Analysis and changes approved. (In future phases, this would apply transformations.)')
            else:
                st.warning('Analysis rejected. You can upload a new file or modify the dataset and try again.')