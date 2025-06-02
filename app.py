import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import json

input_folder = "./z_my_data/test_prompt_data_judged"

# Load files in folder
all_data = []
for filename in os.listdir(input_folder):
    if filename.endswith(".jsonl"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                all_data.append(
                    {
                        "category": data["category"],
                        "name": data["name"],
                        "template": data["template"],
                        "context": data["context"],
                        "completion": data["completion"],
                        "sentiment": data["sentiment"],
                        "bias_score": data["bias_score"],
                        "lowercase": data["additional_metadata"][
                            "lowercase_conversion"
                        ],
                        "word1": data["word1"],
                        "word2": data["word2"],
                    }
                )

df = pd.DataFrame(all_data)

# Debug information
st.sidebar.write(f"Total records loaded: {len(df)}")

st.title("Data Explorer")

# add "All" option
templates = ["All"] + list(df["template"].unique())
template = st.sidebar.selectbox("Filter by Template", templates)

selected_categories = st.sidebar.multiselect(
    "Filter by Category",
    options=sorted(df["category"].unique()),
    help="Leave empty to include all categories",
)

name_options = sorted(df["name"].unique(), key=lambda x: x.lower())
selected_names = st.sidebar.multiselect(
    "Name", options=name_options, help="Leave empty to include all names"
)

# Add word filters
word1_options = sorted(df["word1"].unique(), key=lambda x: x.lower())
selected_word1 = st.sidebar.multiselect(
    "Word 1", options=word1_options, help="Leave empty to include all words"
)

word2_options = sorted(df["word2"].unique(), key=lambda x: x.lower())
selected_word2 = st.sidebar.multiselect(
    "Word 2", options=word2_options, help="Leave empty to include all words"
)

# Add lowercase filter
lowercase_options = ["All", "Yes", "No"]
selected_lowercase = st.sidebar.selectbox(
    "Lowercase", options=lowercase_options, help="Filter by lowercase conversion"
)

# Add sorting option
sort_by = st.sidebar.selectbox(
    "Sort by",
    options=["Sentiment", "Bias Score"],
    help="Choose which metric to sort the plot by",
)

# Filter if not "All"
if template != "All":
    filtered_df = df[df["template"] == template]
else:
    filtered_df = df

# Filter by selected category
if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

# Filter by selected name
if selected_names:
    filtered_df = filtered_df[filtered_df["name"].isin(selected_names)]

# Filter by selected word1
if selected_word1:
    filtered_df = filtered_df[filtered_df["word1"].isin(selected_word1)]

# Filter by selected word2
if selected_word2:
    filtered_df = filtered_df[filtered_df["word2"].isin(selected_word2)]

# Filter by lowercase
if selected_lowercase != "All":
    filtered_df = filtered_df[filtered_df["lowercase"] == selected_lowercase.lower()]

# Debug information
st.sidebar.write(f"Records after filtering: {len(filtered_df)}")

# Calculate averages per name
avg_metrics = filtered_df.groupby("name", as_index=False).agg(
    {"sentiment": "mean", "bias_score": "mean"}
)

# Sort based on user selection
if sort_by == "Sentiment":
    avg_metrics = avg_metrics.sort_values(by="sentiment", ascending=False)
else:  # Bias Score
    avg_metrics = avg_metrics.sort_values(by="bias_score", ascending=False)

# Debug information
st.sidebar.write(f"Attributes: {len(avg_metrics)}")

# Create the figure
fig = go.Figure()

# Add sentiment bars
fig.add_trace(
    go.Bar(
        x=avg_metrics["name"],
        y=avg_metrics["sentiment"],
        name="Sentiment",
        marker_color="green",
        opacity=0.7,
    )
)

# Add bias score bars
fig.add_trace(
    go.Bar(
        x=avg_metrics["name"],
        y=avg_metrics["bias_score"],
        name="Bias Score",
        marker_color="blue",
        opacity=0.7,
    )
)

# Update layout
fig.update_layout(
    title="Average Sentiment and Bias Score per Attribute",
    xaxis_title="Name",
    yaxis_title="Score",
    yaxis_range=[-1, 1],
    barmode="overlay",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
)

st.plotly_chart(fig, use_container_width=True)

# Create distribution plot
st.subheader("Sentiment Distribution by Template")

# Create the distribution plot
fig_dist = px.box(
    filtered_df,
    x="template",
    y="sentiment",
    title=f'Sentiment Distribution by Template{" for selected words" if (selected_word1 or selected_word2) else ""}',
    labels={"template": "Template", "sentiment": "Sentiment"},
)

# Update layout
fig_dist.update_layout(
    xaxis_title="Template",
    yaxis_title="Sentiment",
    yaxis_range=[-1, 1],
    showlegend=False,
)

st.plotly_chart(fig_dist, use_container_width=True)

# Add a checkbox to show raw data
if st.checkbox("Show Raw Data"):
    # Calculate statistics for each template-word1-word2 combination
    stats_df = (
        filtered_df.groupby(["template", "word1", "word2"])
        .agg({"sentiment": ["median", "std", "count"]})
        .reset_index()
    )

    # Flatten column names
    stats_df.columns = [
        "template",
        "word1",
        "word2",
        "median_sentiment",
        "std_sentiment",
        "count",
    ]

    # Merge statistics back to the filtered data
    display_df = filtered_df.merge(
        stats_df, on=["template", "word1", "word2"], how="left"
    )

    # Select and reorder columns for display
    display_columns = [
        "name",
        "category",
        "context",
        "template",
        "word1",
        "word2",
        "completion",
        "sentiment",
        "bias_score",
        "lowercase",
        "median_sentiment",
        "std_sentiment",
        "count",
    ]

    # Format the statistics columns
    display_df["median_sentiment"] = display_df["median_sentiment"].round(3)
    display_df["std_sentiment"] = display_df["std_sentiment"].round(3)

    st.dataframe(display_df[display_columns])

# # Add statistics
# st.subheader("Statistics")
# st.write(f"Total number of examples: {len(filtered_df)}")
# st.write(f"Number of unique names: {len(filtered_df['name'].unique())}")
# st.write(
#     f"Average sentiment across all examples: {filtered_df['sentiment'].mean():.3f}"
# )
# st.write(
#     f"Average bias score across all examples: {filtered_df['bias_score'].mean():.3f}"
# )


# streamlit run app.py
