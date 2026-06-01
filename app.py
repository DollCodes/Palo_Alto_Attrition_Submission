import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Palo Alto Networks Workforce Attrition Dashboard",
    layout="wide",
)

DATA_PATH = "Palo Alto Networks.csv"


@st.cache_data
def load_data(uploaded_file=None):
    data = pd.read_csv(uploaded_file if uploaded_file is not None else DATA_PATH)
    data = data.copy()
    data.columns = [column.strip() for column in data.columns]

    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].astype(str).str.strip()

    data["Attrition"] = pd.to_numeric(data["Attrition"], errors="coerce")
    data = data.dropna(subset=["Attrition", "Department", "JobRole"])
    data["Attrition"] = data["Attrition"].astype(int)

    data["AgeGroup"] = pd.cut(
        data["Age"],
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
        include_lowest=True,
    )
    data["TenureBand"] = pd.cut(
        data["YearsAtCompany"],
        bins=[-1, 1, 3, 5, 10, 60],
        labels=["0-1 years", "2-3 years", "4-5 years", "6-10 years", "11+ years"],
    )
    data["CareerStage"] = pd.cut(
        data["TotalWorkingYears"],
        bins=[-1, 5, 10, 20, 60],
        labels=[
            "Early career (0-5 yrs)",
            "Developing (6-10 yrs)",
            "Experienced (11-20 yrs)",
            "Senior (21+ yrs)",
        ],
    )
    data["DistanceBand"] = pd.cut(
        data["DistanceFromHome"],
        bins=[-1, 5, 10, 20, 100],
        labels=["0-5", "6-10", "11-20", "21+"],
    )
    data["WorkloadSegment"] = np.select(
        [
            (data["OverTime"] == "Yes") & (data["BusinessTravel"] == "Travel_Frequently"),
            data["OverTime"] == "Yes",
            data["BusinessTravel"] == "Travel_Frequently",
        ],
        ["Overtime + frequent travel", "Overtime only", "Frequent travel only"],
        default="Lower workload exposure",
    )
    return data


def attrition_summary(data, group_columns):
    summary = (
        data.groupby(group_columns, observed=False)
        .agg(
            Employees=("Attrition", "size"),
            Exits=("Attrition", "sum"),
            AttritionRate=("Attrition", "mean"),
        )
        .reset_index()
    )
    summary["AttritionRate"] = summary["AttritionRate"] * 100
    return summary.sort_values(["AttritionRate", "Exits"], ascending=False)


def format_rate_table(data):
    table = data.copy()
    table["AttritionRate"] = table["AttritionRate"].map(lambda value: f"{value:.2f}%")
    return table


st.title("Workforce Attrition Patterns and Risk Hotspot Analysis")
st.caption("Palo Alto Networks diagnostic HR analytics dashboard")

with st.sidebar:
    st.header("Data Source")
    uploaded_file = st.file_uploader("Upload alternate CSV", type=["csv"])

df = load_data(uploaded_file)

with st.sidebar:
    st.header("Filters")
    departments = st.multiselect(
        "Department",
        sorted(df["Department"].unique()),
        default=sorted(df["Department"].unique()),
    )
    roles = st.multiselect(
        "Job role",
        sorted(df["JobRole"].unique()),
        default=sorted(df["JobRole"].unique()),
    )
    tenure_range = st.slider(
        "Years at company",
        int(df["YearsAtCompany"].min()),
        int(df["YearsAtCompany"].max()),
        (int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max())),
    )
    overtime = st.multiselect(
        "Overtime",
        sorted(df["OverTime"].unique()),
        default=sorted(df["OverTime"].unique()),
    )
    travel = st.multiselect(
        "Business travel",
        sorted(df["BusinessTravel"].unique()),
        default=sorted(df["BusinessTravel"].unique()),
    )
    gender = st.multiselect(
        "Gender",
        sorted(df["Gender"].unique()),
        default=sorted(df["Gender"].unique()),
    )

filtered_df = df[
    df["Department"].isin(departments)
    & df["JobRole"].isin(roles)
    & df["YearsAtCompany"].between(tenure_range[0], tenure_range[1])
    & df["OverTime"].isin(overtime)
    & df["BusinessTravel"].isin(travel)
    & df["Gender"].isin(gender)
]

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

employees = len(filtered_df)
exited = int(filtered_df["Attrition"].sum())
retained = employees - exited
attrition_rate = filtered_df["Attrition"].mean() * 100
early_tenure_rate = filtered_df.loc[
    filtered_df["YearsAtCompany"] <= 1, "Attrition"
].mean() * 100
workload_rate = filtered_df.loc[
    (filtered_df["OverTime"] == "Yes")
    | (filtered_df["BusinessTravel"] == "Travel_Frequently"),
    "Attrition",
].mean() * 100

st.subheader("Attrition Overview Dashboard")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Employees", f"{employees:,}")
kpi2.metric("Attrition rate", f"{attrition_rate:.2f}%")
kpi3.metric("Exited", f"{exited:,}")
kpi4.metric("Retained", f"{retained:,}")
kpi5.metric("Workload attrition index", f"{workload_rate:.2f}%")

overview_left, overview_right = st.columns(2)
with overview_left:
    st.write("Retained vs exited employee distribution")
    st.bar_chart(pd.DataFrame({"Employees": [retained, exited]}, index=["Retained", "Exited"]))

with overview_right:
    st.write("Attrition by workload exposure")
    workload_summary = attrition_summary(filtered_df, ["WorkloadSegment"])
    st.bar_chart(workload_summary.set_index("WorkloadSegment")["AttritionRate"])

st.subheader("Department and Role Heatmaps")
dept_summary = attrition_summary(filtered_df, ["Department"])
role_summary = attrition_summary(filtered_df, ["JobRole"])
dept_role_summary = attrition_summary(filtered_df, ["Department", "JobRole"])
dept_role_heatmap = dept_role_summary.pivot(
    index="JobRole", columns="Department", values="AttritionRate"
).fillna(0)

left, right = st.columns([1, 2])
with left:
    st.write("Department attrition rate")
    st.dataframe(format_rate_table(dept_summary), hide_index=True, use_container_width=True)
    st.bar_chart(dept_summary.set_index("Department")["AttritionRate"])

with right:
    st.write("Role by department attrition intensity")
    st.dataframe(
        dept_role_heatmap.style.format("{:.2f}%").background_gradient(cmap="Reds", axis=None),
        use_container_width=True,
    )

st.write("Highest-risk job roles")
st.dataframe(format_rate_table(role_summary.head(10)), hide_index=True, use_container_width=True)

st.subheader("Demographic Attrition Explorer")
age_tab, status_tab, education_tab, satisfaction_tab = st.tabs(
    ["Age", "Gender and marital status", "Education", "Satisfaction"]
)

with age_tab:
    age_summary = attrition_summary(filtered_df, ["AgeGroup"])
    st.bar_chart(age_summary.set_index("AgeGroup")["AttritionRate"])
    st.dataframe(format_rate_table(age_summary), hide_index=True, use_container_width=True)

with status_tab:
    gender_summary = attrition_summary(filtered_df, ["Gender"])
    marital_summary = attrition_summary(filtered_df, ["MaritalStatus"])
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Gender")
        st.bar_chart(gender_summary.set_index("Gender")["AttritionRate"])
        st.dataframe(format_rate_table(gender_summary), hide_index=True, use_container_width=True)
    with col_b:
        st.write("Marital status")
        st.bar_chart(marital_summary.set_index("MaritalStatus")["AttritionRate"])
        st.dataframe(format_rate_table(marital_summary), hide_index=True, use_container_width=True)

with education_tab:
    education_summary = attrition_summary(filtered_df, ["Education"])
    field_summary = attrition_summary(filtered_df, ["EducationField"])
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Education level")
        st.bar_chart(education_summary.set_index("Education")["AttritionRate"])
        st.dataframe(format_rate_table(education_summary), hide_index=True, use_container_width=True)
    with col_b:
        st.write("Education field")
        st.bar_chart(field_summary.set_index("EducationField")["AttritionRate"])
        st.dataframe(format_rate_table(field_summary), hide_index=True, use_container_width=True)

with satisfaction_tab:
    for label, column in [
        ("Job satisfaction", "JobSatisfaction"),
        ("Environment satisfaction", "EnvironmentSatisfaction"),
        ("Work-life balance", "WorkLifeBalance"),
    ]:
        summary = attrition_summary(filtered_df, [column])
        st.write(label)
        st.bar_chart(summary.set_index(column)["AttritionRate"])
        st.dataframe(format_rate_table(summary), hide_index=True, use_container_width=True)

st.subheader("Tenure and Workload Analysis")
tenure_left, tenure_right = st.columns(2)
with tenure_left:
    tenure_summary = attrition_summary(filtered_df, ["TenureBand"])
    career_summary = attrition_summary(filtered_df, ["CareerStage"])
    st.write("Attrition by tenure bucket")
    st.bar_chart(tenure_summary.set_index("TenureBand")["AttritionRate"])
    st.dataframe(format_rate_table(tenure_summary), hide_index=True, use_container_width=True)
    st.write("Early-career vs experienced attrition")
    st.bar_chart(career_summary.set_index("CareerStage")["AttritionRate"])

with tenure_right:
    overtime_summary = attrition_summary(filtered_df, ["OverTime"])
    travel_summary = attrition_summary(filtered_df, ["BusinessTravel"])
    distance_summary = attrition_summary(filtered_df, ["DistanceBand"])
    st.write("Overtime impact")
    st.bar_chart(overtime_summary.set_index("OverTime")["AttritionRate"])
    st.write("Business travel impact")
    st.bar_chart(travel_summary.set_index("BusinessTravel")["AttritionRate"])
    st.write("Distance from home impact")
    st.bar_chart(distance_summary.set_index("DistanceBand")["AttritionRate"])

st.subheader("Risk Hotspot Table")
risk_table = dept_role_summary[dept_role_summary["Employees"] >= 20].copy()
risk_table["RiskScore"] = (
    risk_table["AttritionRate"] * 0.7
    + (risk_table["Exits"] / risk_table["Exits"].max()) * 30
)
risk_table = risk_table.sort_values("RiskScore", ascending=False).head(15)
st.dataframe(format_rate_table(risk_table), hide_index=True, use_container_width=True)

st.subheader("Actionable Retention Recommendations")
st.markdown(
    """
1. Build a first-year retention program because 0-1 year tenure attrition is materially above baseline.
2. Reduce avoidable overtime and frequent-travel pressure, especially where both occur together.
3. Prioritize Sales Representatives, Laboratory Technicians, and Human Resources roles for targeted retention reviews.
4. Improve work-life balance, job satisfaction, and environment satisfaction in high-risk groups.
5. Monitor entry-level and early-career employees as a distinct workforce risk segment.
6. Review stock option and long-term incentive eligibility for high-risk talent pools.
"""
)
