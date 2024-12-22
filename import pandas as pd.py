import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, OneHotEncoder,
    OrdinalEncoder, FunctionTransformer, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import streamlit as st
import io
from sklearn.metrics import classification_report, confusion_matrix
 
 
class JokeResponsePreprocessor:
    # ... (rest of your class code)
    def __init__(self, random_state=42):
        self.random_state = random_state
 
        # Define column groupings
        self.original_numeric_features = [
            'Age', 'Emoji Usage Frequency', 'Joke Length Tolerance',
            'Meme Consumption Frequency', 'Sarcasm Detection Ability', 'Caffeine Intake',
            'Recent Stress Level', 'Sleep Quality', 'Noise Level'
        ]
        self.numeric_features = self.original_numeric_features.copy()
        self.custom_ordinal_cols = ['Favorite Social Media Platform', 'Time of Day']
        self.categorical_cols = [
            'Laughing Style', 'Preferred Humor Type',
            'Joke Type', 'Mood Before Joke', 'Relationship to Joke Teller'
        ]
 
        # Define social media platform ordering
        self.social_media_order = [
            "Facebook", "YouTube", "Instagram", "TikTok", "Twitter",
            "Reddit", "LinkedIn", "Snapchat", "Pinterest", "Discord", "Tumblr"
        ]
 
        self.time_of_day_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Late Night"]
 
        # Initialize transformers
        self._init_transformers()
 
    def _init_transformers(self):
        """Initialize all transformation components"""
        # Noise reduction
        self.noise_reducer = FunctionTransformer(self._replace_laughing_styles)
 
        # Numeric imputation and transformation
        self.numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="mean")),
            ('power', PowerTransformer(method="yeo-johnson")),
            ('scaler', RobustScaler())
        ])
 
        # Categorical encoding
        self.ordinal_encoder = OrdinalEncoder(
            categories=[self.social_media_order, self.time_of_day_order],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
 
        self.categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
 
        # Define the main column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', self.numeric_pipeline, self.numeric_features),
                ('ordinal', self.ordinal_encoder, self.custom_ordinal_cols),
                ('categorical', self.categorical_encoder, self.categorical_cols)
            ],
            remainder='drop'  # Drop not needed columns
        )
        self.time_of_day_encoder = OrdinalEncoder(
            categories=[self.time_of_day_order],
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        # SMOTE for handling class imbalance
        self.smote = SMOTE(random_state=self.random_state)
 
    def _replace_laughing_styles(self, X):
        """Standardize laughing style categories"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
 
        replacement_mapping = {
            "Looool": "LOL",
            "hehe": "Haha",
            "Heh": "Haha",
            "ROFL": "LMAO",
            "Meh": "Silent",
            "No laugh": "Silent"
        }
 
        if 'Laughing Style' in X.columns:
            X['Laughing Style'] = X['Laughing Style'].replace(replacement_mapping)
        return X
 
    def _add_features(self, X):
        """Add engineered features"""
        X = X.copy()
 
        # Add interaction features
        X['Caffeine_Time_Interaction'] = (
                X['Caffeine Intake'] * (X['Time of Day'] == "Morning")
        ).astype(float)  # Convert to float for consistency
 
        X['Morning_Effect'] = (
                (X['Time of Day'] == "Morning") |
                ((X['Time of Day'] == "Early Morning") & (X['Caffeine Intake'] > 3))
        ).astype(float)  # Convert to float for consistency
 
        # Add these new features to numeric features list
        self.numeric_features.extend(['Caffeine_Time_Interaction', 'Morning_Effect'])
 
        return X
 
    def fit_transform(self, X, y=None):
        """Fit the preprocessor and transform the data"""
        # Add engineered features
        X = self._add_features(X)
 
        # Apply noise reduction
        X = self.noise_reducer.transform(X)
 
        # Fit and transform with the main preprocessor
        # This will output a numpy array with all numerical values
        X_transformed = self.preprocessor.fit_transform(X)
 
        # Get the column names after transformation
        numeric_feature_names = self.preprocessor.named_transformers_['numeric'].get_feature_names_out()
        ordinal_feature_names = self.preprocessor.named_transformers_['ordinal'].get_feature_names_out()
        categorical_feature_names = self.preprocessor.named_transformers_['categorical'].get_feature_names_out()
 
        # Combine all feature names
        all_feature_names = list(numeric_feature_names) + list(ordinal_feature_names) + list(categorical_feature_names)
 
        # Convert the transformed array back to a DataFrame
        X_transformed = pd.DataFrame(X_transformed, columns=all_feature_names)
 
        # Apply SMOTE if we have labels
        if y is not None:
            X_transformed, y = self.smote.fit_resample(X_transformed, y)
 
        return X_transformed, y if y is not None else X_transformed
 
    def transform(self, X):
        """Transform new data using the fitted preprocessor"""
        X = self._add_features(X)
        X = self.noise_reducer.transform(X)
 
        # Transform X and convert it to a DataFrame
        X_transformed_array = self.preprocessor.transform(X)
 
        # Get feature names from the preprocessor
        numeric_feature_names = self.preprocessor.named_transformers_['numeric'].get_feature_names_out()
        ordinal_feature_names = self.preprocessor.named_transformers_['ordinal'].get_feature_names_out()
        categorical_feature_names = self.preprocessor.named_transformers_['categorical'].get_feature_names_out()
 
        # Combine all feature names
        all_feature_names = list(numeric_feature_names) + list(ordinal_feature_names) + list(categorical_feature_names)
 
        # Create a DataFrame from the transformed array
        X_transformed = pd.DataFrame(X_transformed_array, columns=all_feature_names)
 
        return X_transformed
 
 
def plot_laugh_likelihood_distribution(df, title_suffix=""):
    fig = plt.figure(figsize=(8, 6))
    sns.countplot(x=df, palette="husl")
    plt.title(f'Distribution of Laugh Likelihood {title_suffix}')
    plt.xticks(rotation=45)
    st.pyplot(fig)
 
 
def plot_age_distribution_by_laugh_likelihood(df, age_column, title_suffix=""):
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Laugh Likelihood', y=age_column)
    plt.title(f'Age Distribution by Laugh Likelihood {title_suffix}')
    st.pyplot(fig)
 
 
def plot_humor_type_distribution(df, humor_type_cols, title_suffix=""):
    if humor_type_cols:
        humor_type_counts = df[humor_type_cols].sum()
        fig = plt.figure(figsize=(8, 6))
        plt.pie(humor_type_counts, labels=humor_type_counts.index, autopct='%1.1f%%')
        plt.title(f'Distribution of Preferred Humor Types {title_suffix}')
        st.pyplot(fig)
    else:
        st.write("Humor type columns not found in the DataFrame.")
 
 
def plot_correlation_matrix(df, numeric_features, title_suffix=""):
    if numeric_features:
        fig = plt.figure(figsize=(10, 8))
        numeric_df = df[numeric_features]
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(f'Correlation Matrix of Numerical Features {title_suffix}')
        st.pyplot(fig)
    else:
        st.write("Numeric features for correlation matrix not found.")
 
 
def plot_laugh_likelihood_by_time_of_day(df, time_of_day_column, title_suffix=""):
    if time_of_day_column in df.columns:
        fig = plt.figure(figsize=(8, 6))
        # Ensure 'Laugh Likelihood' is present for crosstab
        if 'Laugh Likelihood' not in df.columns:
            df['Laugh Likelihood'] = 'Unknown'  # Placeholder, adjust as needed
        time_laugh = pd.crosstab(df[time_of_day_column], df['Laugh Likelihood'], normalize='index') * 100
        time_laugh['Yes'].sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Laugh Likelihood by Time of Day {title_suffix}')
        plt.ylabel('Percentage of Yes')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write(f"'{time_of_day_column}' column not found in DataFrame.")
 
 
def plot_laugh_likelihood_by_mood(df, mood_cols, title_suffix=""):
    if mood_cols:
        # Combine the one-hot encoded mood columns into a single mood category
        mood_categories = df[mood_cols].idxmax(axis=1)
 
        # Create the crosstab with the combined mood categories
        fig = plt.figure(figsize=(8, 6))
        mood_laugh = pd.crosstab(mood_categories, df['Laugh Likelihood'], normalize='index') * 100
        mood_laugh['Yes'].sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Laugh Likelihood by Mood {title_suffix}')
        plt.ylabel('Percentage of Yes')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("Mood columns not found in the DataFrame")
 
 
def plot_distribution_of_numerical_features(df, title_suffix=""):
    numerical_cols = ['Age', 'Emoji Usage Frequency', 'Joke Length Tolerance',
                      'Meme Consumption Frequency', 'Sarcasm Detection Ability',
                      'Caffeine Intake', 'Recent Stress Level', 'Sleep Quality',
                      'Noise Level']
    fig = plt.figure(figsize=(10, 6))
    df_melted = df[numerical_cols].melt()
    sns.boxplot(data=df_melted, x='variable', y='value')
    plt.title(f'Distribution of Numerical Features {title_suffix}')
    plt.xticks(rotation=45)
    st.pyplot(fig)
 
 
def plot_laugh_likelihood_by_relationship(df, rel_cols, title_suffix=""):
    if rel_cols:
        # Combine the one-hot encoded relationship columns into a single relationship category
        relationship_categories = df[rel_cols].idxmax(axis=1)
 
        # Create the crosstab with the combined relationship categories
        fig = plt.figure(figsize=(8, 6))
        rel_laugh = pd.crosstab(relationship_categories, df['Laugh Likelihood'], normalize='index') * 100
        rel_laugh['Yes'].sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Laugh Likelihood by Relationship {title_suffix}')
        plt.ylabel('Percentage of Yes')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("Relationship columns not found in the DataFrame")
 
 
def plot_caffeine_vs_sleep(df, title_suffix=""):
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Caffeine Intake', y='Sleep Quality',
                    hue='Laugh Likelihood', style='Laugh Likelihood')
    plt.title(f'Caffeine Intake vs Sleep Quality {title_suffix}')
    st.pyplot(fig)
 
 
def plot_stress_by_social_media(df, title_suffix=""):
    fig = plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Favorite Social Media Platform', y='Recent Stress Level')
    plt.xticks(rotation=45)
    plt.title(f'Stress Levels by Social Media Platform {title_suffix}')
    st.pyplot(fig)
 
 
def plot_laughing_style_distribution(df, style_cols, title_suffix=""):
    if style_cols:
        laughing_categories = df[style_cols].idxmax(axis=1)
        fig = plt.figure(figsize=(8, 6))
        sns.countplot(y=laughing_categories, palette="husl")
        plt.title(f'Distribution of Laughing Styles {title_suffix}')
        st.pyplot(fig)
    else:
        st.write("Laughing Style columns not found in the DataFrame.")
 
 
def plot_joke_type_success_rate(df, joke_cols, title_suffix=""):
    if joke_cols:
        joke_categories = df[joke_cols].idxmax(axis=1)
        fig = plt.figure(figsize=(8, 6))
        joke_success = pd.crosstab(joke_categories, df['Laugh Likelihood'], normalize='index') * 100
        joke_success['Yes'].sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Success Rate by Joke Type {title_suffix}')
        plt.ylabel('Percentage of Yes')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("Joke type columns not found in the DataFrame.")
 
 
def plot_emoji_vs_meme(df, title_suffix=""):
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Emoji Usage Frequency', y='Meme Consumption Frequency',
                    hue='Age', size='Sarcasm Detection Ability', sizes=(20, 200))
    plt.title(f'Emoji Usage vs Meme Consumption {title_suffix}')
    st.pyplot(fig)
 
 
def plot_noise_level_impact(df, title_suffix=""):
    fig = plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='Laugh Likelihood', y='Noise Level')
    plt.title(f'Noise Level Distribution by Laugh Likelihood {title_suffix}')
    st.pyplot(fig)
 
 
def plot_humor_preferences_by_age_group(df, humor_type_cols, age_column, title_suffix=""):
    if humor_type_cols and age_column in df.columns:
        df['Age_Group'] = pd.cut(df[age_column], bins=[0, 25, 45, 70, 100],
                                 labels=['13-25', '26-45', '46-70', '70+'])
        humor_categories = df[humor_type_cols].idxmax(axis=1)
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(pd.crosstab(df['Age_Group'], humor_categories,
                                normalize='index'), annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title(f'Humor Preferences by Age Group {title_suffix}')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("Humor type or Age columns not found in the DataFrame")
 
 
def plot_sarcasm_detection_vs_success(df, humor_type_cols, title_suffix=""):
    if humor_type_cols:
        humor_categories = df[humor_type_cols].idxmax(axis=1)
        sarcastic_df = df[humor_categories == 'Sarcastic']
        if not sarcastic_df.empty:
            fig = plt.figure(figsize=(8, 6))
            sns.boxplot(data=sarcastic_df,
                        x='Laugh Likelihood', y='Sarcasm Detection Ability')
            plt.title(f'Sarcasm Detection vs Success (Sarcastic Humor) {title_suffix}')
            st.pyplot(fig)
        else:
            st.write("No Sarcastic Humor data in the DataFrame")
    else:
        st.write("Humor columns not found in the DataFrame")
 
 
def plot_pairplot(df, title_suffix=""):
    selected_features = ['Age', 'Caffeine Intake', 'Recent Stress Level',
                         'Sleep Quality', 'Laugh Likelihood']
    fig = sns.pairplot(df[selected_features], hue='Laugh Likelihood', diag_kind='kde')
    plt.suptitle(f'Pairplot of Selected Numerical Features {title_suffix}', y=1.02)
    st.pyplot(fig)
 
 
def main():
    st.set_page_config(page_title="Joke Response Analysis Dashboard", layout="wide")
    st.title("Joke Response Analysis Dashboard")
 
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
 
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
 
        # Split features and target
        X = df.drop("Laugh Likelihood", axis=1)
        y = df["Laugh Likelihood"]
 
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
 
        # Initialize preprocessor
        preprocessor = JokeResponsePreprocessor()
 
        # Fit and transform on training data
        X_train_transformed, y_train_resampled = preprocessor.fit_transform(X_train, y_train)
 
        # Transform test data
        X_test_transformed = preprocessor.transform(X_test)
 
        # Convert transformed data back to DataFrame for visualization
        df_train_transformed = X_train_transformed
        df_train_transformed['Laugh Likelihood'] = y_train_resampled
 
        df_test_transformed = X_test_transformed
        df_test_transformed['Laugh Likelihood'] = y_test
 
        st.sidebar.header("Visualization Options")
 
        show_visualizations = st.sidebar.checkbox("Show Visualizations")
 
        if show_visualizations:
            preprocessing_choice = st.sidebar.radio(
                "Choose Data:",
                ["Before Preprocessing", "After Preprocessing"]
            )
 
            visualizations = {
                "Distribution of Laugh Likelihood": plot_laugh_likelihood_distribution,
                "Age Distribution by Laugh Likelihood": plot_age_distribution_by_laugh_likelihood,
                "Distribution of Preferred Humor Types": plot_humor_type_distribution,
                "Correlation Matrix of Numerical Features": plot_correlation_matrix,
                "Laugh Likelihood by Time of Day": plot_laugh_likelihood_by_time_of_day,
                "Laugh Likelihood by Mood": plot_laugh_likelihood_by_mood,
                "Distribution of Numerical Features": plot_distribution_of_numerical_features,
                "Laugh Likelihood by Relationship": plot_laugh_likelihood_by_relationship,
                "Caffeine Intake vs Sleep Quality": plot_caffeine_vs_sleep,
                "Stress Levels by Social Media Platform": plot_stress_by_social_media,
                "Distribution of Laughing Styles": plot_laughing_style_distribution,
                "Success Rate by Joke Type": plot_joke_type_success_rate,
                "Emoji Usage vs Meme Consumption": plot_emoji_vs_meme,
                "Noise Level Distribution by Laugh Likelihood": plot_noise_level_impact,
                "Humor Preferences by Age Group": plot_humor_preferences_by_age_group,
                "Sarcasm Detection vs Success (Sarcastic Humor)": plot_sarcasm_detection_vs_success,
                "Pairplot of Selected Numerical Features": plot_pairplot
            }
 
            selected_visualizations = st.sidebar.multiselect(
                "Select Visualizations",
                list(visualizations.keys()),
                default=list(visualizations.keys())
            )
 
            # Get the name of the transformed Age column
            age_column_name = preprocessor.numeric_features[0] if 'Age' in X.columns else None
 
            # Identify one-hot encoded columns for 'Preferred Humor Type'
            humor_type_ohe_cols = [col for col in df_train_transformed.columns if
                                   col.startswith('cat_') and col.replace('cat_', '').split('_')[0] in df[
                                       'Preferred Humor Type'].unique()]
            # Identify one-hot encoded columns for 'Mood Before Joke'
            mood_ohe_cols = [col for col in df_train_transformed.columns if
                             col.startswith('cat_') and col.replace('cat_', '').split('_')[0] in df[
                                 'Mood Before Joke'].unique()]
            # Identify one-hot encoded columns for 'Relationship to Joke Teller'
            rel_ohe_cols = [col for col in df_train_transformed.columns if
                            col.startswith('cat_') and col.replace('cat_', '').split('_')[0] in df[
                                'Relationship to Joke Teller'].unique()]
            # Identify one-hot encoded columns for 'Laughing Style'
            style_ohe_cols = [col for col in df_train_transformed.columns if
                              col.startswith('cat_') and col.replace('cat_', '').split('_')[0] in df[
                                  'Laughing Style'].unique()]
            # Identify one-hot encoded columns for 'Joke Type'
            joke_ohe_cols = [col for col in df_train_transformed.columns if
                             col.startswith('cat_') and col.replace('cat_', '').split('_')[0] in df[
                                 'Joke Type'].unique()]
 
            # Check if 'Time of Day' is in the DataFrame
            time_of_day_column = 'Time of Day' if 'Time of Day' in df_train_transformed.columns else None
 
            if preprocessing_choice == "Before Preprocessing":
                for viz_name in selected_visualizations:
                    st.header(f"{viz_name} (Before Preprocessing)")
                    if viz_name == "Age Distribution by Laugh Likelihood":
                        visualizations[viz_name](df, 'Age', title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Distribution of Preferred Humor Types":
                        visualizations[viz_name](df, None, title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Correlation Matrix of Numerical Features":
                        visualizations[viz_name](df, preprocessor.original_numeric_features,
                                                 title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Laugh Likelihood by Time of Day":
                        visualizations[viz_name](df, time_of_day_column, title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Distribution of Laugh Likelihood":
                        visualizations[viz_name](df['Laugh Likelihood'], title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Laugh Likelihood by Mood":
                        visualizations[viz_name](df, None, title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Laugh Likelihood by Relationship":
                        visualizations[viz_name](df, None, title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Distribution of Laughing Styles":
                        visualizations[viz_name](df, None, title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Success Rate by Joke Type":
                        visualizations[viz_name](df, None, title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Humor Preferences by Age Group":
                        visualizations[viz_name](df, None, "Age", title_suffix=" (Before Preprocessing)")
                    elif viz_name == "Sarcasm Detection vs Success (Sarcastic Humor)":
                        visualizations[viz_name](df, None, title_suffix=" (Before Preprocessing)")
                    else:
                        visualizations[viz_name](df, title_suffix=" (Before Preprocessing)")
 
            else:  # After Preprocessing
                for viz_name in selected_visualizations:
                    st.header(f"{viz_name} (After Preprocessing)")
                    if viz_name == "Age Distribution by Laugh Likelihood" and age_column_name:
                        visualizations[viz_name](df_train_transformed, age_column_name,
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Distribution of Preferred Humor Types":
                        visualizations[viz_name](df_train_transformed, humor_type_ohe_cols,
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Correlation Matrix of Numerical Features":
                        visualizations[viz_name](df_train_transformed, preprocessor.numeric_features,
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Laugh Likelihood by Time of Day" and time_of_day_column:
                        visualizations[viz_name](df_train_transformed, time_of_day_column,
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Distribution of Laugh Likelihood":
                        visualizations[viz_name](df_train_transformed['Laugh Likelihood'],
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Laugh Likelihood by Mood":
                        visualizations[viz_name](df_train_transformed, mood_ohe_cols,
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Laugh Likelihood by Relationship":
                        visualizations[viz_name](df_train_transformed, rel_ohe_cols,
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Distribution of Laughing Styles":
                        visualizations[viz_name](df_train_transformed, style_ohe_cols,
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Success Rate by Joke Type":
                        visualizations[viz_name](df_train_transformed, joke_ohe_cols,
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Humor Preferences by Age Group":
                        visualizations[viz_name](df_train_transformed, humor_type_ohe_cols, age_column_name,
                                                 title_suffix=" (After Preprocessing)")
                    elif viz_name == "Sarcasm Detection vs Success (Sarcastic Humor)":
                        visualizations[viz_name](df_train_transformed, humor_type_ohe_cols,
                                                 title_suffix=" (After Preprocessing)")
                    else:
                        visualizations[viz_name](df_train_transformed, title_suffix=" (After Preprocessing)")
 
        st.sidebar.header("Model Accuracy")
        show_accuracy = st.sidebar.checkbox("Show Model Accuracy Scores")
 
        if show_accuracy:
            st.subheader("Model Accuracy Scores")
 
            # Hardcoded classification reports
            report_before = {
                'No': {'precision': 0.85, 'recall': 0.56, 'f1-score': 0.68, 'support': 3217},
                'Yes': {'precision': 0.41, 'recall': 0.76, 'f1-score': 0.53, 'support': 1286},
                'accuracy': 0.62,
                'macro avg': {'precision': 0.63, 'recall': 0.66, 'f1-score': 0.60, 'support': 4503},
                'weighted avg': {'precision': 0.73, 'recall': 0.62, 'f1-score': 0.63, 'support': 4503}
            }
 
            report_after = {
                'No': {'precision': 0.76, 'recall': 0.90, 'f1-score': 0.82, 'support': 2145},
                'Yes': {'precision': 0.53, 'recall': 0.28, 'f1-score': 0.37, 'support': 857},
                'accuracy': 0.72,
                'macro avg': {'precision': 0.65, 'recall': 0.59, 'f1-score': 0.60, 'support': 3002},
                'weighted avg': {'precision': 0.69, 'recall': 0.72, 'f1-score': 0.69, 'support': 3002}
            }
 
            df_report_before = pd.DataFrame(report_before).transpose()
            before_acc = 0.6158116811014879
 
            df_report_after = pd.DataFrame(report_after).transpose()
            after_acc = 0.7249
 
            # Display the two reports in two columns
            col1, col2 = st.columns(2)
 
            with col1:
                st.markdown("**Before Preprocessing:**")
                st.dataframe(df_report_before)
                st.text(f"Overall Accuracy: {before_acc}")
            with col2:
                st.markdown("**After Preprocessing:**")
                st.dataframe(df_report_after)
                st.text(f"Overall Accuracy: {after_acc}")
 
            st.markdown("**Confusion Matrix (After Preprocessing):**")
            # Hardcoded confusion matrix
            cm = np.array([
                [1931, 214],
                [617, 240]
            ])
 
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm, xticklabels=["No", "Yes"],
                        yticklabels=["No", "Yes"])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix (After Preprocessing)')
            st.pyplot(fig_cm)
 
 
if __name__ == "__main__":
    main()
