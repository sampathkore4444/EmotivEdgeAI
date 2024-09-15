import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from langdetect import detect
import pandas as pd
import concurrent.futures

st.set_page_config(page_title="AI Powered Emotivedge")

# Load the classifier globally to save resources
# classifier = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis") #This model is best

# Cache the model and processor to ensure they are loaded only once
@st.cache_resource
def load_model():
    classifier = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis") #This model is best
    return classifier

# Load model and processor only once
classifier = load_model()



# st.title('Sentiment Analysis App')
st.title('Emotivedge Powered by AI')

def analyze_text(text):
    result = classifier(text)[0]
    return result['label'], result['score']

def analyze_text_concurrent(texts):
    # Create a ThreadPoolExecutor to manage threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all texts to the executor and create future objects
        future_to_index = {executor.submit(analyze_text, text): i for i, text in enumerate(texts)}
        results = []
        # Collect results as they complete, ensuring they remain in order
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f'Text at index {index} generated exception: {exc}')
            else:
                results.append((index, result))
        # Sort results by the original indices
        results.sort()
        # Extract just the results in their original order
        sorted_results = [result for _, result in results]
    return sorted_results

def analyze_dataframe(df, text_column):
    sentiments = analyze_text_concurrent(df[text_column].tolist())
    df['Sentiment_Label'], df['Sentiment_Score'] = zip(*sentiments)
    return df

user_input = st.text_area('Enter employee feedback here:', '')

uploaded_file = st.file_uploader("Upload Excel file containing employee feedback", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Uploaded DataFrame:")
    st.write(df)
    text_column = st.selectbox("Select the column containing feedback data", df.columns)
    if st.button('Analyze Feedback in Excel File'):
        result_df = analyze_dataframe(df, text_column)
        st.write("Sentiment Analysis Results:")
        st.write(result_df)

        # Visualization of Sentiment Distribution
        # st.write("Sentiment Distribution:")
        st.write("Emotion Distribution:")
        fig, ax = plt.subplots()
        sns.countplot(x='Emotion_Label', data=result_df, ax=ax)
        plt.close(fig)  # Close the figure after rendering
        st.pyplot(fig)

        # Interactive filtering
        st.subheader('Interactive Filtering')
        emotion_filter = st.selectbox('Filter by Emotion', options=['','All', 'POS', 'NEG', 'NEU'])
        print("sentiment_filter=",emotion_filter)

        if emotion_filter != 'All':
            result_df = result_df[result_df['Sentiment_Label'] == emotion_filter]
       
        st.write("Filtered Results:")
        st.write(result_df)

        # Download option
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results as CSV", csv, "sentiment_analysis_results.csv", "text/csv", key='download-csv')

if st.button('Analyze Employee Emotion for Input Text'):
    if user_input:
        st.write('<script>document.title="";</script>', unsafe_allow_html=True)
        label, score = analyze_text(user_input)
        # st.write('Sentiment Analysis Results using BERT:')
        st.write('Employee Emotion Analysis Results:')
        # st.write(f"Label: {label}")
        label = 'POSITIVE ' if label=='POS' else 'NEGATIVE'
        st.write(f"Emotion: {label}")
        st.write(f"Score: {score}")

        # Summary statistics
        st.write('Text Summary Statistics:')
        word_count = len(user_input.split())
        char_count = len(user_input)
        st.write(f'Word Count: {word_count}')
        st.write(f'Character Count: {char_count}')

        # Language detection
        st.write('Language Detection:')
        lang = detect(user_input)
        st.write(f'Detected Language: {lang}')

st.write('Note: Enter employee feedback in the box above or upload Excel file to analyze the sentiments.')
