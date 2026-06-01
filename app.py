import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Palo Alto Networks Workforce Attrition Dashboard",
    layout="wide"
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA_PATH = "Palo Alto Networks.csv"

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data
def load_data(uploaded_file=None):

    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(DATA_PATH)

    except FileNotFoundError:
        st.error(
            "Dataset not found.\n\n"
            "Upload the CSV file using the sidebar or place "
            "'Palo Alto Networks.csv' in the repository root."
        )
        st.stop()

    except Exception as e:
        st.error(f"Unable to load dataset.\n\nError: {e}")
        st.stop()

    df.columns = df.columns.str.strip()

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    required_columns = [
        "Attrition",
        "Department",
        "JobRole",
        "Age",
        "YearsAtCompany",
        "TotalWorkingYears",
        "DistanceFromHome",
        "OverTime",
        "BusinessTravel",
        "Gender"
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        st.error(
            "Dataset is missing required columns:\n\n"
            + ", ".join(missing)
        )
        st.stop()

    df["Attrition"] = pd.to_numeric(
        df["Attrition"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "Attrition",
            "Department",
            "JobRole"
        ]
    )

    df["Attrition"] = df["Attrition"].astype(int)

    # -----------------------------
    # Derived Features
    # -----------------------------

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56+"
        ]
    )

    df["TenureBand"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-1, 1, 3, 5, 10, 100],
        labels=[
            "0-1 years",
            "2-3 years",
            "4-5 years",
            "6-10 years",
            "11+ years"
        ]
    )

    df["CareerStage"] = pd.cut(
        df["TotalWorkingYears"],
        bins=[-1, 5, 10, 20, 100],
        labels=[
            "Early Career",
            "Developing",
            "Experienced",
            "Senior"
        ]
    )

    df["DistanceBand"] = pd.cut(
        df["DistanceFromHome"],
        bins=[-1, 5, 10, 20, 100],
        labels=[
            "0-5",
            "6-10",
            "11-20",
            "21+"
        ]
    )

    df["WorkloadSegment"] = np.select(
        [
            (df["OverTime"] == "Yes")
            & (df["BusinessTravel"] == "Travel_Frequently"),

            df["OverTime"] == "Yes",

            df["BusinessTravel"] == "Travel_Frequently"
        ],
        [
            "Overtime + Travel",
            "Overtime Only",
            "Travel Only"
        ],
        default="Lower Workload"
    )

    return df


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def attrition_summary(df, group_cols):

    summary = (
        df.groupby(group_cols, observed=False)
        .agg(
            Employees=("Attrition", "size"),
            Exits=("Attrition", "sum"),
            AttritionRate=("Attrition", "mean")
        )
        .reset_index()
    )

    summary["AttritionRate"] *= 100

    return summary.sort_values(
        ["AttritionRate", "Exits"],
        ascending=False
    )


def format_rate_table(df):

    temp = df.copy()

    if "AttritionRate" in temp.columns:
        temp["AttritionRate"] = temp["AttritionRate"].round(2)

    return temp


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "Workforce Attrition Patterns and Risk Hotspot Analysis"
)

st.caption(
    "Palo Alto Networks Diagnostic HR Analytics Dashboard"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_data(uploaded_file)

# --------------------------------------------------
# FILTERS
# --------------------------------------------------

with st.sidebar:

    st.header("Filters")

    departments = st.multiselect(
        "Department",
        sorted(df["Department"].unique()),
        default=sorted(df["Department"].unique())
    )

    roles = st.multiselect(
        "Job Role",
        sorted(df["JobRole"].unique()),
        default=sorted(df["JobRole"].unique())
    )

    tenure_range = st.slider(
        "Years at Company",
        int(df["YearsAtCompany"].min()),
        int(df["YearsAtCompany"].max()),
        (
            int(df["YearsAtCompany"].min()),
            int(df["YearsAtCompany"].max())
        )
    )

    overtime = st.multiselect(
        "OverTime",
        sorted(df["OverTime"].unique()),
        default=sorted(df["OverTime"].unique())
    )

    travel = st.multiselect(
        "Business Travel",
        sorted(df["BusinessTravel"].unique()),
        default=sorted(df["BusinessTravel"].unique())
    )

    gender = st.multiselect(
        "Gender",
        sorted(df["Gender"].unique()),
        default=sorted(df["Gender"].unique())
    )

# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered_df = df[
    df["Department"].isin(departments)
    & df["JobRole"].isin(roles)
    & df["YearsAtCompany"].between(
        tenure_range[0],
        tenure_range[1]
    )
    & df["OverTime"].isin(overtime)
    & df["BusinessTravel"].isin(travel)
    & df["Gender"].isin(gender)
]

if filtered_df.empty:
    st.warning(
        "No records match the selected filters."
    )
    st.stop()

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

employees = len(filtered_df)

exited = int(
    filtered_df["Attrition"].sum()
)

retained = employees - exited

attrition_rate = (
    filtered_df["Attrition"].mean() * 100
)

workload_rate = (
    filtered_df.loc[
        (
            filtered_df["OverTime"] == "Yes"
        )
        |
        (
            filtered_df["BusinessTravel"]
            == "Travel_Frequently"
        ),
        "Attrition"
    ].mean()
    * 100
)

st.subheader("Attrition Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Employees", f"{employees:,}")
c2.metric("Attrition Rate", f"{attrition_rate:.2f}%")
c3.metric("Exited", f"{exited:,}")
c4.metric("Retained", f"{retained:,}")
c5.metric("Workload Index", f"{workload_rate:.2f}%")

# --------------------------------------------------
# CHARTS
# --------------------------------------------------

left, right = st.columns(2)

with left:

    st.write(
        "Retained vs Exited Employees"
    )

    chart_df = pd.DataFrame(
        {
            "Employees": [
                retained,
                exited
            ]
        },
        index=[
            "Retained",
            "Exited"
        ]
    )

    st.bar_chart(chart_df)

with right:

    st.write(
        "Attrition by Workload Segment"
    )

    workload_summary = attrition_summary(
        filtered_df,
        ["WorkloadSegment"]
    )

    st.bar_chart(
        workload_summary.set_index(
            "WorkloadSegment"
        )["AttritionRate"]
    )

# --------------------------------------------------
# DEPARTMENT ANALYSIS
# --------------------------------------------------

st.subheader(
    "Department & Role Analysis"
)

dept_summary = attrition_summary(
    filtered_df,
    ["Department"]
)

role_summary = attrition_summary(
    filtered_df,
    ["JobRole"]
)

dept_role_summary = attrition_summary(
    filtered_df,
    ["Department", "JobRole"]
)

heatmap_df = dept_role_summary.pivot(
    index="JobRole",
    columns="Department",
    values="AttritionRate"
).fillna(0)

left, right = st.columns([1, 2])

with left:

    st.write(
        "Department Attrition Rate"
    )

    st.dataframe(
        format_rate_table(dept_summary),
        use_container_width=True
    )

    st.bar_chart(
        dept_summary.set_index(
            "Department"
        )["AttritionRate"]
    )

with right:

    st.write(
        "Role by Department Attrition"
    )

    st.dataframe(
        heatmap_df.round(2),
        use_container_width=True
    )

# --------------------------------------------------
# TOP RISK ROLES
# --------------------------------------------------

st.subheader(
    "Highest Risk Job Roles"
)

st.dataframe(
    format_rate_table(
        role_summary.head(10)
    ),
    use_container_width=True
)

# --------------------------------------------------
# RISK HOTSPOTS
# --------------------------------------------------

st.subheader(
    "Risk Hotspots"
)

risk_table = dept_role_summary.copy()

risk_table = risk_table[
    risk_table["Employees"] >= 20
]

if not risk_table.empty:

    max_exits = max(
        risk_table["Exits"].max(),
        1
    )

    risk_table["RiskScore"] = (
        risk_table["AttritionRate"] * 0.7
        +
        (
            risk_table["Exits"]
            / max_exits
        ) * 30
    )

    risk_table = risk_table.sort_values(
        "RiskScore",
        ascending=False
    ).head(15)

    st.dataframe(
        risk_table,
        use_container_width=True
    )

# --------------------------------------------------
# RECOMMENDATIONS
# --------------------------------------------------

st.subheader(
    "Retention Recommendations"
)

st.markdown(
"""
1. Strengthen first-year employee retention initiatives.

2. Reduce excessive overtime and travel burden.

3. Prioritize high-risk departments and roles.

4. Improve work-life balance and job satisfaction.

5. Develop career-growth pathways for early-career employees.

6. Continuously monitor attrition hotspots.
"""
)
