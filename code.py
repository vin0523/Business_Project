import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("📊 Social Media Trends Dashboard with Automated Insights")

# Upload CSV
uploaded_file = st.file_uploader("Upload your social_media_trends.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data Cleaning
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Engagement'] = pd.to_numeric(df['Engagement'], errors='coerce')
    df = df.dropna(subset=['Engagement'])
    df = df.drop_duplicates()

    # Show raw data
    st.subheader("📄 Preview of Cleaned Data")
    st.dataframe(df.head())

    # Generate Insights
    def generate_insights(df):
        top_hashtag = df.groupby('Hashtag')['Engagement'].mean().idxmax()
        top_platform = df.groupby('Platform')['Engagement'].mean().idxmax()
        best_day = df.groupby(df['Date'].dt.day_name())['Engagement'].mean().idxmax()
        return f"""
        ### 🔍 Automated Insights
        - 📈 Highest average engagement hashtag: `#{top_hashtag}`
        - 🌐 Top-performing platform: **{top_platform}**
        - 🗓️ Best day to post: **{best_day}**
        """

    st.markdown(generate_insights(df))

    # Daily Engagement Trend
    st.subheader("📅 Average Engagement Over Time")
    daily_engagement = df.groupby(df['Date'].dt.date)['Engagement'].mean()
    st.line_chart(daily_engagement)

    # Top Hashtags
    st.subheader("🏷️ Top 10 Hashtags by Average Engagement")
    top_hashtags = df.groupby('Hashtag')['Engagement'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_hashtags.values, y=top_hashtags.index, ax=ax)
    st.pyplot(fig)

    # Platform Distribution
    st.subheader("🌐 Platform Distribution")
    platform_counts = df['Platform'].value_counts()
    fig2, ax2 = plt.subplots()
    sns.barplot(x=platform_counts.index, y=platform_counts.values, ax=ax2)
    st.pyplot(fig2)

    # Engagement by Post Type
    st.subheader("📝 Engagement by Post Type")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x='Post_Type', y='Engagement', data=df, ax=ax3)
    st.pyplot(fig3)

    # Feature Engineering
    df['weekday'] = df['Date'].dt.weekday
    df['month'] = df['Date'].dt.month

    # Machine Learning Section
    st.subheader("🤖 Engagement Prediction using Random Forest")
    X = pd.get_dummies(df[['Hashtag', 'Platform', 'Post_Type', 'weekday', 'month']], drop_first=True)
    y = df['Engagement']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Manually calculate RMSE to avoid 'squared' argument issue
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    st.markdown(f"- RMSE: `{rmse:.2f}`\n- R^2 Score: `{r2:.2f}`")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True).tail(10)
    fig4, ax4 = plt.subplots()
    importances.plot(kind='barh', ax=ax4)
    ax4.set_title("Top 10 Feature Importances")
    st.pyplot(fig4)

else:
    st.info("Please upload a CSV file to get started.")
