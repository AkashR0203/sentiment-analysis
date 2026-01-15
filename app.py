import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Page title and description
st.title("🧠 Sentiment Analysis Web App")
st.markdown("Analyze the sentiment of English text: **Positive**, **Negative**, or **Neutral**. Built with VADER for accurate results.")

# Single text analysis section
st.header("Single Text Analysis")
text = st.text_area("Enter your text here (in English):", height=150)

if st.button("Analyze Text"):
    if text.strip():
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "Positive 😊"
            color = "success"
        elif compound <= -0.05:
            sentiment = "Negative 😞"
            color = "error"
        else:
            sentiment = "Neutral 😐"
            color = "warning"
        
        st.markdown(f"**Sentiment: {sentiment}**")
        with st.expander("View Detailed Scores"):
            st.json(scores)
    else:
        st.warning("Please enter some text to analyze!")

# Batch analysis section
st.header("Batch Analysis from CSV")
st.info("Upload a CSV file with a column named **'review'** containing the text to analyze.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        if 'review' not in df.columns:
            st.error("CSV must contain a column named 'review'!")
        else:
            st.success("File uploaded successfully!")
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Analyze sentiments
            sentiments = []
            for review in df['review']:
                review_str = str(review) if pd.notna(review) else ""
                scores = analyzer.polarity_scores(review_str)
                compound = scores['compound']
                if compound >= 0.05:
                    sent = "Positive"
                elif compound <= -0.05:
                    sent = "Negative"
                else:
                    sent = "Neutral"
                sentiments.append(sent)
            
            df['sentiment'] = sentiments
            
            # Display results
            st.write("Analysis Results:")
            st.dataframe(df)
            
            # Visualization: Sentiment counts bar chart
            sentiment_counts = df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)
            
            # Download button
            csv_data = df.to_csv(index=False).encode()
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )
            
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Footer
st.markdown("---")
st.caption("Built by Akash Rao 🚀 | Powered by Streamlit & VADER Sentiment")