import streamlit as st
import pandas as pd
import ast
from collections import defaultdict
import plotly.express as px

# pip install streamlit pandas plotly
# streamlit run app.py

# Title of the application
st.title("Analysis of Event Sub-Sequences")

# Description section with larger equations
st.subheader("Metrics Description")

st.markdown("**Support**")
st.latex(r"\text{support}(A) = \frac{\#\text{ of sequences with } A}{\#\text{ of total sequences}}")

st.markdown("**Confidence**")
st.latex(r"\text{confidence}(A \rightarrow I) = \frac{\#\text{ of sequences with } A \text{ and } I}{\#\text{ of sequences with } A}")

st.markdown("**Lift**")
st.latex(r"\text{lift}(A \rightarrow I) = \frac{\text{confidence}(A \rightarrow I)}{\text{global proportion of } I}")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Loading the data
@st.cache_data
def load_data():
    df = pd.read_csv('sncb_data_challenge.csv', sep=';')

    # Function to parse list representations in string format
    def parse_sequence(seq_str):
        return ast.literal_eval(seq_str)

    # Parse the sequences in the DataFrame
    sequence_columns = ['vehicles_sequence', 'events_sequence', 'seconds_to_incident_sequence',
                        'train_kph_sequence', 'dj_ac_state_sequence', 'dj_dc_state_sequence']

    for col in sequence_columns:
        df[col] = df[col].apply(parse_sequence)

    # Ensure 'incident_type' is treated as a string
    df['incident_type'] = df['incident_type'].astype(str)

    return df






df = load_data()

# Selection of parameters
min_subseq_length = st.sidebar.slider('Minimum sub-sequence length', 1, 10, 1)
max_subseq_length = st.sidebar.slider('Maximum sub-sequence length', 1, 20, 5)
min_support_count = st.sidebar.slider('Minimum support count (total occurrences)', 1, 50, 3)
min_lift = st.sidebar.slider('Minimum lift threshold', 1.0, 10.0, 3.0)
min_confidence = st.sidebar.slider('Minimum confidence threshold', 0.0, 1.0, 1.0)

# Selection of incident types
incident_types = df['incident_type'].unique()
selected_incidents = st.sidebar.multiselect('Select incident types', incident_types, default=incident_types)

# Add a checkbox to include only events before the incident
include_only_events_before_incident = st.sidebar.checkbox('Include only events before the incident', value=True)

# Button to run the analysis
if st.sidebar.button('Run Analysis'):

    # Data structures to count sub-sequences per incident type
    subseq_counts_per_incident = defaultdict(lambda: defaultdict(int))
    total_subseq_counts = defaultdict(int)

    # Total number of sequences (to calculate support)
    total_sequences = len(df)

    # Number of sequences per incident type (to calculate proportions)
    incident_type_counts = df['incident_type'].value_counts().to_dict()

    # Iterate over the data to extract and count sub-sequences
    for idx, row in df.iterrows():
        events_sequence = row['events_sequence']
        seconds_sequence = row['seconds_to_incident_sequence']
        incident_type = row['incident_type']

        # Adjust based on the checkbox
        if include_only_events_before_incident:
            # Filter events to keep only those occurring before the incident
            events_to_consider = [event for event, time in zip(events_sequence, seconds_sequence) if time < 0]
        else:
            # Take all events
            events_to_consider = events_sequence

        # Generate sub-sequences of all possible lengths
        seq_length = len(events_to_consider)
        for subseq_len in range(min_subseq_length, min(max_subseq_length, seq_length) + 1):
            for i in range(seq_length - subseq_len + 1):
                subseq = tuple(events_to_consider[i:i + subseq_len])
                # Count the occurrence of the sub-sequence for the incident type
                subseq_counts_per_incident[subseq][incident_type] += 1
                # Count the total number of occurrences of the sub-sequence
                total_subseq_counts[subseq] += 1

    # Filter sub-sequences by minimum support count (total occurrences)
    frequent_subseqs = {subseq: count for subseq, count in total_subseq_counts.items() if count >= min_support_count}

    # Calculate metrics for each sub-sequence
    subseq_metrics = []

    for subseq in frequent_subseqs:
        incident_counts = subseq_counts_per_incident[subseq]
        total_occurrences = total_subseq_counts[subseq]
        for incident_type, count in incident_counts.items():
            support = count / total_sequences
            confidence = count / total_occurrences
            # Proportion of sequences leading to this incident type
            incident_proportion = incident_type_counts[incident_type] / total_sequences
            lift = confidence / incident_proportion if incident_proportion > 0 else 0
            subseq_metrics.append({
                'subsequence': subseq,
                'incident_type': incident_type,
                'length': len(subseq),
                'support': support,
                'confidence': confidence,
                'lift': lift,
                'occurrences': count,
                'total_occurrences': total_occurrences
            })

    # Convert metrics to DataFrame for easier analysis
    metrics_df = pd.DataFrame(subseq_metrics)

    # Filter strongly associated sub-sequences based on the defined thresholds
    metrics_df = metrics_df[
        (metrics_df['lift'] >= min_lift) &
        (metrics_df['confidence'] >= min_confidence) &
        (metrics_df['incident_type'].isin(selected_incidents))
    ]

    # Sort sub-sequences by 'incident_type', 'lift', 'confidence', and 'length' in descending order
    metrics_df = metrics_df.sort_values(by=['incident_type', 'lift', 'confidence', 'length'],
                                        ascending=[True, False, False, False])

    # Reset index
    metrics_df = metrics_df.reset_index(drop=True)

    # Convert 'subsequence' column to a string for better readability
    metrics_df['subsequence'] = metrics_df['subsequence'].apply(lambda x: ' -> '.join(map(str, x)))

    # Display results
    st.subheader("Results of Highly Associated Sub-Sequences")
    st.write(f"Number of sub-sequences found: {len(metrics_df)}")

    # Display the DataFrame
    st.dataframe(metrics_df)

    # Graphical visualization
    st.subheader("Visualizations")

    # Interactive bar chart of sub-sequences by lift
    fig_lift = px.bar(metrics_df, x='subsequence', y='lift', color='incident_type',
                      title='Lift of Sub-Sequences by Incident Type',
                      labels={'lift': 'Lift', 'subsequence': 'Sub-Sequence', 'incident_type': 'Incident Type'},
                      height=600)
    st.plotly_chart(fig_lift, use_container_width=True)

    # Interactive bar chart of sub-sequences by confidence
    fig_confidence = px.bar(metrics_df, x='subsequence', y='confidence', color='incident_type',
                            title='Confidence of Sub-Sequences by Incident Type',
                            labels={'confidence': 'Confidence', 'subsequence': 'Sub-Sequence', 'incident_type': 'Incident Type'},
                            height=600)
    st.plotly_chart(fig_confidence, use_container_width=True)

    # Interactive bar chart of sub-sequences by support
    fig_support = px.bar(metrics_df, x='subsequence', y='support', color='incident_type',
                         title='Support of Sub-Sequences by Incident Type',
                         labels={'support': 'Support', 'subsequence': 'Sub-Sequence', 'incident_type': 'Incident Type'},
                         height=600)
    st.plotly_chart(fig_support, use_container_width=True)

    # Option to download the results
    st.subheader("Download Results")
    csv = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name='highly_associated_subsequences.csv',
        mime='text/csv',
    )

else:
    st.write("Please select the parameters and click on 'Run Analysis' to display the results.")
