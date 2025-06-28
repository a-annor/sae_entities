import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import streamlit.components.v1 as components

input_folder = "./data/test_prompt_data_judged"

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
                        "judge_score": data["judge_score"],
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
    options=["Sentiment", "Judge Score"],
    help="Choose which metric to sort the plot by",
)

# Add sentiment and bias score sliders
st.sidebar.subheader("Filter by Score Range")
min_sentiment, max_sentiment = st.sidebar.slider(
    "Sentiment Range", min_value=-1.0, max_value=0.0, value=(-1.0, 0.0), step=0.01
)

min_judge, max_judge = st.sidebar.slider(
    "Judge Score Range", min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.01
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

# Apply sentiment and bias score filters
filtered_df = filtered_df[
    (filtered_df["sentiment"] >= min_sentiment)
    & (filtered_df["sentiment"] <= max_sentiment)
    & (filtered_df["judge_score"] >= min_judge)
    & (filtered_df["judge_score"] <= max_judge)
]

# Debug information
st.sidebar.write(f"Records after filtering: {len(filtered_df)}")

# Calculate averages per name
avg_metrics = filtered_df.groupby("name", as_index=False).agg(
    {"sentiment": "mean", "judge_score": "mean"}
)

# Sort based on user selection
if sort_by == "Sentiment":
    avg_metrics = avg_metrics.sort_values(by="sentiment", ascending=False)
else:  # Judge Score
    avg_metrics = avg_metrics.sort_values(by="judge_score", ascending=False)

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

# Add judge score bars
fig.add_trace(
    go.Bar(
        x=avg_metrics["name"],
        y=avg_metrics["judge_score"],
        name="Judge Score",
        marker_color="blue",
        opacity=0.7,
    )
)

# Update layout
fig.update_layout(
    # title="Average Sentiment and Judge Score per Attribute",
    xaxis_title="Target Attribute",
    yaxis_title="Average Value",
    yaxis_range=[-1, 1],
    barmode="overlay",
    legend=dict(font=dict(size=14), yanchor="top", y=0.99, xanchor="left", x=1.05),
    font=dict(size=20, color="black"),
    xaxis=dict(
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
        # showgrid=True,
        gridcolor="#e5e5e5",
        tickmode="array",
        tickvals=avg_metrics["name"],
    ),
    yaxis=dict(tickfont=dict(color="black"), title_font=dict(color="black")),
)

st.plotly_chart(fig, use_container_width=True)

# Create distribution plot
st.subheader("Sentiment Distribution by Template")

# Create the distribution plot
fig_dist = px.box(
    filtered_df,
    y="template",
    x="sentiment",
    title=f'Sentiment Distribution by Template{" for selected words" if (selected_word1 or selected_word2) else ""}',
    labels={"template": "Template", "sentiment": "Sentiment"},
    hover_data=[],
    color="template",
)

# Update layout
fig_dist.update_layout(
    yaxis_title="Template",
    xaxis_title="Sentiment",
    xaxis_range=[-1, 0],
    showlegend=False,
    height=600,
    width=1200,
    yaxis=dict(
        side="right",
        title_standoff=10,
        showgrid=True,
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    ),
    xaxis=dict(
        showgrid=True,
        tickmode="linear",
        dtick=0.1,
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    ),
    font=dict(size=20, color="black"),
)

# Turn off hover completely
fig_dist.update_traces(hoverinfo="none")

# Use st.plotly_chart with use_container_width=False
st.plotly_chart(fig_dist, use_container_width=False)

# Add judge score box plot
st.subheader("Judge Score Distribution")
fig_box_bias = px.box(
    filtered_df,
    x="judge_score",
    labels={"judge_score": "Judge Score"},
    range_x=[0, 1],
)
fig_box_bias.update_traces(marker_color="#a0d6ff", line_color="#a0d6ff")

# Update layout
fig_box_bias.update_layout(
    xaxis_title="Judge Score",
    showlegend=False,
    height=300,
    width=1200,
    xaxis=dict(
        showgrid=True,
        tickmode="linear",
        dtick=0.1,
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    ),
    font=dict(size=20, color="black"),
)

st.plotly_chart(fig_box_bias, use_container_width=False)

# Add sentiment box plot
st.subheader("Sentiment Distribution")
fig_box_sentiment = px.box(
    filtered_df,
    x="sentiment",
    labels={"sentiment": "Sentiment"},
    range_x=[-1, 0],
)
fig_box_sentiment.update_traces(marker_color="#a0d6ff", line_color="#a0d6ff")

# Update layout
fig_box_sentiment.update_layout(
    xaxis_title="Sentiment",
    showlegend=False,
    height=300,
    width=1200,
    xaxis=dict(
        showgrid=True,
        tickmode="linear",
        dtick=0.1,
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    ),
    font=dict(size=20, color="black"),
)

st.plotly_chart(fig_box_sentiment, use_container_width=False)

# Add a checkbox to show raw data
if st.checkbox("Show Raw Data"):
    # Calculate overall bias statistics
    overall_judge_median = filtered_df["judge_score"].median()
    overall_judge_std = filtered_df["judge_score"].std()

    # Calculate statistics for each template (only for sentiment and count)
    stats_df = (
        filtered_df.groupby(["template"])
        .agg(
            {
                "sentiment": ["median", "std"],
                "name": "count",  # Using 'name' column to count records per group
            }
        )
        .reset_index()
    )

    # Flatten column names
    stats_df.columns = [
        "template",
        "median_sentiment",
        "std_sentiment",
        "count",
    ]

    # Merge template-grouped statistics back to the filtered data
    display_df = filtered_df.merge(stats_df, on=["template"], how="left")

    # Add overall bias statistics to every row
    display_df["overall_median_judge_score"] = overall_judge_median
    display_df["overall_std_judge_score"] = overall_judge_std

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
        "judge_score",
        "lowercase",
        "median_sentiment",
        "std_sentiment",
        "overall_median_judge_score",  # New overall column
        "overall_std_judge_score",  # New overall column
        "count",
    ]

    # Format the statistics columns
    display_df["median_sentiment"] = display_df["median_sentiment"].round(2)
    display_df["std_sentiment"] = display_df["std_sentiment"].round(2)
    display_df["overall_median_judge_score"] = display_df[
        "overall_median_judge_score"
    ].round(2)
    display_df["overall_std_judge_score"] = display_df["overall_std_judge_score"].round(
        2
    )

    st.dataframe(display_df[display_columns])

# # Add statistics
# st.subheader("Statistics")
# st.write(f"Total number of examples: {len(filtered_df)}")
# st.write(f"Number of unique names: {len(filtered_df['name'].unique())}")
# st.write(
#     f"Average sentiment across all examples: {filtered_df['sentiment'].mean():.3f}"
# )
# st.write(
#     f"Average judge score across all examples: {filtered_df['judge_score'].mean():.3f}"
# )


# streamlit run app.py
