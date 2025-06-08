import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Salary Insights Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("clean_data.csv")
    
    # Create all binned columns at load time
    df['Age Group'] = pd.cut(df['Age'], 
                           bins=[10, 20, 30, 40, 50, 60, 70],
                           labels=['10-20', '20-30', '30-40', '40-50', '50-60', '60+'])
    
    df['Salary Bin'] = pd.cut(df['Salary'],
                            bins=[0, 50000, 100000, 150000, 200000, 250000, 300000],
                            labels=['<50K', '50-100K', '100-150K', '150-200K', '200-250K', '250K+'])
    
    df['Experience Group'] = pd.cut(
        df['Years of Experience'],
        bins=[-0.1, 5, 10, 15, 20, 30, 50],
        labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30+"],
        include_lowest=True
    )
    
    return df

df = load_data()

# Sidebar filters
with st.sidebar:
    st.title("Filters")
    
    selected_genders = st.multiselect(
        "Gender",
        options=sorted(df['Gender'].unique()),
        default=sorted(df['Gender'].unique())
    )
    
    selected_education = st.multiselect(
        "Education Level",
        options=sorted(df['Education Level'].unique()),
        default=sorted(df['Education Level'].unique())
    )
    
    min_exp, max_exp = st.slider(
        "Years of Experience",
        min_value=float(df['Years of Experience'].min()),
        max_value=float(df['Years of Experience'].max()),
        value=(0.0, float(df['Years of Experience'].max()))
    )
    
    min_age, max_age = st.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(20, 70)
    )
    
    min_salary, max_salary = st.slider(
        "Salary Range (USD)",
        min_value=int(df['Salary'].min()),
        max_value=int(df['Salary'].max()),
        value=(30000, 250000)
    )

# Apply filters and create a frozen copy for metrics
@st.cache_data
def get_filtered_data(df, genders, education, min_exp, max_exp, min_age, max_age, min_salary, max_salary):
    filtered = df[
        (df['Gender'].isin(genders)) &
        (df['Education Level'].isin(education)) &
        (df['Years of Experience'].between(min_exp, max_exp)) &
        (df['Age'].between(min_age, max_age)) &
        (df['Salary'].between(min_salary, max_salary))
    ].copy()
    
    # Create all dynamic columns here before returning
    age_bin_size = 5  # Default value
    age_bins = list(range(20, 71, age_bin_size))
    if age_bins[-1] != 70:
        age_bins.append(70)
    age_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins)-1)]
    filtered["Dynamic Age Group"] = pd.cut(
        filtered["Age"],
        bins=age_bins,
        labels=age_labels
    )
    
    return filtered

filtered_df = get_filtered_data(
    df, selected_genders, selected_education, 
    min_exp, max_exp, min_age, max_age, min_salary, max_salary)

# Dashboard header
st.image('header.jpg')
st.title("Salary Analysis Dashboard")
st.markdown("""Created by Wan Ammar & Tuan Faqih""")
st.markdown("""Discover the Truth Behind Paychecks — Slice the Data Your Way!
Use the sidebar filters to customize your view.""")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Employees", f"{len(filtered_df):,}")
col2.metric("Avg Salary", f"${filtered_df['Salary'].mean():,.0f}")
col3.metric("Median Salary", f"${filtered_df['Salary'].median():,.0f}")
col4.metric("Salary Range", f"${filtered_df['Salary'].min():,.0f} - ${filtered_df['Salary'].max():,.0f}")

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Gender Analysis", 
    "Experience Impact", 
    "Education Comparison", 
    "Job Title View", 
    "Age Trends"
])

## TAB 1: Gender Analysis
with tab1:
    st.header("Salary Distribution by Gender")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Box Plot Analysis")
        show_outliers = st.checkbox("Show outliers", value=True, key="outliers")
        
        fig_box = px.box(
            filtered_df,
            x="Gender",
            y="Salary",
            color="Gender",
            points="outliers" if show_outliers else False,
            title="Salary Distribution by Gender"
        )
        fig_box.update_layout(
            yaxis_title="Salary (USD)",
            xaxis_title="Gender",
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        st.subheader("Bar Chart Comparison")
        metric = st.radio(
            "Select metric:",
            ["Mean", "Median"],
            horizontal=True,
            key="gender_metric"
        )
        
        if metric == "Mean":
            gender_data = filtered_df.groupby("Gender")["Salary"].mean().reset_index()
        else:
            gender_data = filtered_df.groupby("Gender")["Salary"].median().reset_index()
        
        fig_bar = px.bar(
            gender_data,
            x="Gender",
            y="Salary",
            color="Gender",
            text_auto=".2s",
            title=f"{metric} Salary by Gender"
        )
        fig_bar.update_layout(
            yaxis_title=f"{metric} Salary (USD)",
            xaxis_title="Gender",
            showlegend=False
        )
        fig_bar.update_traces(
            texttemplate="$%{y:,.0f}",
            textposition="outside"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Gender pay gap calculation
    if len(selected_genders) == 2:
        male_avg = filtered_df[filtered_df["Gender"] == "Male"]["Salary"].mean()
        female_avg = filtered_df[filtered_df["Gender"] == "Female"]["Salary"].mean()
        pay_gap = male_avg - female_avg
        pay_gap_pct = (pay_gap / female_avg) * 100
        
        st.subheader("Gender Pay Gap Analysis")
        st.write(f"Average difference: ${pay_gap:,.0f}")
        st.write(f"Percentage difference: {pay_gap_pct:.1f}%")

## TAB 2: Experience Impact
with tab2:
    st.header("Salary vs. Years of Experience")
    
    fig_scatter = px.scatter(
        filtered_df,
        x="Years of Experience",
        y="Salary",
        color="Gender",
        trendline="lowess",
        hover_data=["Job Title", "Education Level"],
        title="Salary Growth with Experience"
    )
    fig_scatter.update_layout(
        yaxis_title="Salary (USD)",
        xaxis_title="Years of Experience"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.subheader("Experience Group Analysis")
    
    exp_bins = [-0.1, 5, 10, 15, 20, 30, 50]
    filtered_df["Experience Group"] = pd.cut(
        filtered_df["Years of Experience"],
        bins=exp_bins,
        labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30+"],
        include_lowest=True 
    )
    
    exp_stats = filtered_df.groupby("Experience Group")["Salary"].agg(["mean", "median", "count"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_exp_bar = px.bar(
            exp_stats,
            x=exp_stats.index,
            y="mean",
            error_y=filtered_df.groupby("Experience Group")["Salary"].std(),
            title="Average Salary by Experience Group"
        )
        fig_exp_bar.update_layout(
            yaxis_title="Average Salary (USD)",
            xaxis_title="Years of Experience"
        )
        st.plotly_chart(fig_exp_bar, use_container_width=True)
    
    with col2:
        st.dataframe(
            exp_stats.style.format({
                "mean": "${:,.0f}",
                "median": "${:,.0f}",
                "count": "{:,}"
            })
        )

## TAB 3: Education Comparison
with tab3:
    st.header("Average Salary by Education Level")
    
    edu_stats = filtered_df.groupby("Education Level")["Salary"].mean().sort_values(ascending=False)
    
    fig_edu = px.bar(
        edu_stats.reset_index(),
        x="Salary",
        y="Education Level",
        orientation="h",
        color="Salary",
        color_continuous_scale="Bluered",
        title="Average Salary by Education Level (Sorted)"
    )
    fig_edu.update_layout(
        xaxis_title="Average Salary (USD)",
        yaxis_title="Education Level",
        yaxis={"categoryorder": "total ascending"}
    )
    st.plotly_chart(fig_edu, use_container_width=True)
    
    if len(selected_education) >= 2:
        highest_edu = edu_stats.index[0]
        lowest_edu = edu_stats.index[-1]
        premium = edu_stats.iloc[0] - edu_stats.iloc[-1]
        premium_pct = (premium / edu_stats.iloc[-1]) * 100
        
        st.subheader("Education Premium")
        st.write(f"{highest_edu} earns ${premium:,.0f} more than {lowest_edu}")
        st.write(f"That's a {premium_pct:.1f}% difference")

## TAB 4: Job Title View
with tab4:
    st.header("Salary Distribution by Job Title")

    # Compute top 10 highest-paying roles by average salary
    avg_salary_by_job = (
        filtered_df.groupby("Job Title")["Salary"]
        .mean()
        .sort_values(ascending=False)
    )
    top_jobs = avg_salary_by_job.head(10).index.tolist()

    # Job title selection with searchable multiselect
    selected_jobs = st.multiselect(
        "Search or select job titles to view:",
        options=sorted(filtered_df["Job Title"].unique()),
        default=top_jobs,
        key="job_select"
    )

    if selected_jobs:
        job_filtered = filtered_df[filtered_df["Job Title"].isin(selected_jobs)]

        # Recalculate average salary just for selected jobs
        avg_selected = (
            job_filtered.groupby("Job Title")["Salary"]
            .mean()
            .sort_values(ascending=True)  # so high-salary jobs show at the top
            .reset_index()
        )

        # Plot horizontal bar chart
        fig_job = px.bar(
            avg_selected,
            x="Salary",
            y="Job Title",
            orientation="h",
            title="Salary Distribution by Job Title",
            color="Salary",
            color_continuous_scale="Blues"
        )

        fig_job.update_layout(
            xaxis_title="Average Salary (USD)",
            yaxis_title="Job Title",
            yaxis=dict(tickfont=dict(size=11)),
            coloraxis_showscale=False,
            plot_bgcolor='white'
        )

        st.plotly_chart(fig_job, use_container_width=True)
    else:
        st.warning("Please select at least one job title to display.")

## TAB 5: Age Trends
with tab5:
    st.header("Salary Trends by Age Group")
    
    age_bin_size = st.select_slider(
        "Age group size:",
        options=[2, 5, 10],
        value=5,
        key="age_bin"
    )
    
    age_bins = list(range(20, 71, age_bin_size))
    if age_bins[-1] != 70:
        age_bins.append(70)
    
    age_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins)-1)]
    filtered_df["Dynamic Age Group"] = pd.cut(
        filtered_df["Age"],
        bins=age_bins,
        labels=age_labels
    )
    
    viz_type = st.radio(
        "Visualization type:",
        ["Line Chart", "Heatmap"],
        horizontal=True,
        key="viz_type"
    )
    
    if viz_type == "Line Chart":
        age_data = filtered_df.groupby(["Dynamic Age Group", "Gender"])["Salary"].mean().reset_index()
        
        fig_age = px.line(
            age_data,
            x="Dynamic Age Group",
            y="Salary",
            color="Gender",
            markers=True,
            title=f"Average Salary by Age Group ({age_bin_size}-year bins)"
        )
        fig_age.update_layout(
            yaxis_title="Average Salary (USD)",
            xaxis_title="Age Group"
        )
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        heat_data = filtered_df.groupby(["Dynamic Age Group", "Salary Bin"]).size().unstack().fillna(0)
        
        fig_heat = px.imshow(
            heat_data,
            labels=dict(x="Salary Range", y="Age Group", color="Count"),
            color_continuous_scale="Viridis",
            aspect="auto",
            title=f"Salary Distribution by Age Group ({age_bin_size}-year bins)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# Display clean data
with st.expander("View Data Table"):
    st.subheader(f"Cleaned Dataset ({len(filtered_df):,} rows)")
    st.dataframe(filtered_df)

# Data export
with st.sidebar:
    st.download_button(
        "Download Filtered Data",
        filtered_df.to_csv(index=False),
        "filtered_salary_data.csv",
        "text/csv",
        key='download-csv'
    )
