import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import plotly.express as px

# App Title
# 
st.set_page_config(page_title="Mam Sim", layout="wide")
st.title("Mam Sim")
patient_dict = pickle.load(open("data/patient_dict_day.pkl", "rb"))
expected_dict = pickle.load(open("data/expected_day_dict_day.pkl", "rb"))
# clip expected_dict
for day in expected_dict:
    expected_dict[day] = [x for x in expected_dict[day] if x >=0 and x<= 1500]

# # Sidebar Configuration
# st.sidebar.header("Options")
# option = st.sidebar.selectbox(
#     "Select an option:",
#     ["Home", "Data Visualization", "About"]
# )

# Days of the week
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# graph histogram to show number of patients each day
st.header("Expected number of patients each day of week:")

# Create 7 columns for each day
columns = st.columns(7)
# Plot and display distribution for each day
for col, day in zip(columns, days_of_week):
    with col:
        day_data = patient_dict[day]
        # Create a distribution plot for the day
        fig = px.histogram(
            x=day_data, 
            title=day, 
            nbins=10, 
        )
        fig.update_layout(
            title=dict(font=dict(size=14)),
            margin=dict(l=10, r=10, t=30, b=10),
            height=200,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Calculate statistics
        median = np.median(day_data)
        q1 = np.percentile(day_data, 25)
        q3 = np.percentile(day_data, 75)
        min_val = np.min(day_data)
        max_val = np.max(day_data)

        st.write(f"**Median:** {median}")
        st.write(f"**Q1:** {q1}")
        st.write(f"**Q3:** {q3}")
        st.write(f"**Min:** {min_val}")
        st.write(f"**Max:** {max_val}")

st.header("Expected days from the visit to the examination:")

# Create 7 columns for each day
columns = st.columns(7)
# Plot and display distribution for each day
for col, day in zip(columns, days_of_week):
    with col:
        day_data = expected_dict[day]
        # Create a distribution plot for the day
        fig = px.histogram(
            x=day_data, 
            title=day, 
            nbins=10, 
        )
        fig.update_layout(
            title=dict(font=dict(size=14)),
            margin=dict(l=10, r=10, t=30, b=10),
            height=200,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Calculate statistics
        median = np.median(day_data)
        q1 = np.percentile(day_data, 25)
        q3 = np.percentile(day_data, 75)
        min_val = np.min(day_data)
        max_val = np.max(day_data)

        st.write(f"**Median:** {median}")
        st.write(f"**Q1:** {q1}")
        st.write(f"**Q3:** {q3}")
        st.write(f"**Min:** {min_val}")
        st.write(f"**Max:** {max_val}")




### Choose the number of throughput each day
st.header("Throughput per Day")

# Default slot values
slot_values = [0] * len(days_of_week)  # Initialize with zeros

# Create a row of buttons
button_cols = st.columns(5)
# Initialize session state for slot selection
if "slot_values" not in st.session_state:
    st.session_state.slot_values = [114, 88, 97, 72, 45, 40, 66]  # Default to "Min" values

with button_cols[0]:
    if st.button("Prefill with Min"):
        st.session_state.slot_values = [114, 88, 97, 72, 45, 40, 66]
with button_cols[1]:
    if st.button("Prefill with Max"):
        st.session_state.slot_values = [143, 145, 137, 119, 120, 47, 83]
with button_cols[2]:
    if st.button("Prefill with Median"):
        st.session_state.slot_values = [128, 132, 121, 109, 107, 44, 78]
with button_cols[3]:
    if st.button("Prefill with Q1"):
        st.session_state.slot_values = [122, 123, 113, 105, 101, 43, 76]
with button_cols[4]:
    if st.button("Prefill with Q3"):
        st.session_state.slot_values = [134, 137, 126, 113, 111, 44, 80]

# Create columns for inputs for each day
columns = st.columns(len(days_of_week))
slots = {}

# Add number inputs
for col, day, default_value in zip(columns, days_of_week, st.session_state.slot_values):
    with col:
        slots[day] = st.number_input(f"{day[:3]}", min_value=0, step=1, value=default_value)

col1, col2, col3, col4 = st.columns(4)
with col1:
    num_days = st.number_input(
        "Number of Days to Simulate",
        min_value=365,
        step=1,
        value=365  # Default value
    )

with col2:
    window_finding = st.number_input(
        "Window for Finding Available Slots",
        1, 60, 15
    )
with col3:
    burn_in = st.number_input(
        "Burn-in Period",
        min_value=1,
        max_value=2000,
        value=1
    )
with col4:
    num_run = st.number_input(
        "Number of Runs",
        min_value=1,
        step=1,
        value=1
    )

appt_threshold = window_finding  # Threshold for late appointments





import random


# number of run 

appointment_schedule_list = []
late_patients_list = []
wait_times_list = []
available_slots_list = []

if st.button("Run Simulation"):
    st.write("Running simulation...")

    for run in range(num_run):
        available_slots = list(slots.values()) * 10000  # Initialize available slots
        # Initialize variables
        appointment_schedule = []  # Tracks appointment days
        wait_times = []  # Tracks wait times for all patients
        placeholder = st.empty()
        progress_bar = st.progress(0)
        for day in range(num_days+burn_in):  # Simulate for 5 years
            current_day_index = day % 7  # Get the current day of the week
            current_day = days_of_week[current_day_index]  # Get the current day

            patients_today = random.choice(patient_dict[current_day])  # Get the number of patients for the day
            progress_bar.progress((day + 1) / (num_days+burn_in))
            if day == burn_in:
                burn_in_patient_count = len(appointment_schedule)
            for patient in range(patients_today):
                expected_day = random.choice(expected_dict[current_day])
                next_day = int(day + expected_day)
                with placeholder:
                    st.write(f"Day {day} Patient {patient} expects to visit on day {next_day} with available slots {available_slots[next_day]}")
                
                if available_slots[next_day] > 0:
                    appointment_schedule.append(expected_day)
                    available_slots[next_day] -= 1
                    wait_times.append(0)
                else:
                    # find the next available slot within window_finding days
                    found = False
                    for i in range(1, window_finding):
                        if available_slots[next_day + i] > 0:
                            appointment_schedule.append(expected_day + i)
                            available_slots[next_day + i] -= 1
                            found = True
                            break
                        if available_slots[next_day - i] > 0:
                            appointment_schedule.append(expected_day - i)
                            available_slots[next_day - i] -= 1
                            found = True
                            break
                        wait_times.append(0)
                    if not found:
                        # find the next available slot until found 
                        i = 15
                        while not found:
                            if available_slots[next_day + i] > 0:
                                available_slots[next_day + i] -= 1
                                found = True
                            i += 1

                        wait_times.append(i)
                        appointment_schedule.append(expected_day + i)

        appointment_schedule_list.append(appointment_schedule[burn_in_patient_count:])
        late_patients_list.append(len([v for v in wait_times[burn_in_patient_count:] if v > appt_threshold]))
        wait_times_list.append(wait_times[burn_in_patient_count:])
        available_slots_list.append(available_slots[burn_in:burn_in+num_days])

    # Calculate statistics
    total_patients = [len(appointment_schedule) for appointment_schedule in appointment_schedule_list]
    late_percentage = [(late_patients / total_patients) * 100 for late_patients, total_patients in zip(late_patients_list, total_patients)]
    avg_wait_time = [np.mean([v for v in wait_times if v > 0]) for wait_times in wait_times_list]


    def get_stats(data):
        stats = {
            "Metric": ["Mean", "Median", "Standard Deviation", "Variance", "Minimum", "Maximum", "Q1 (25th percentile)", "Q3 (75th percentile)"],
            "Value": [
                np.mean(data),
                np.median(data),
                np.std(data),
                np.var(data),
                np.min(data),
                np.max(data),
                np.percentile(data, 25),
                np.percentile(data, 75),
            ],
        }
        # Create a DataFrame for display
        stats_df = pd.DataFrame(stats)
        return stats_df

  

    # Display results
    st.header(f"Simulation Results (Based on {num_run} runs)")

    st.write("Total Patients: ")
    stats_df = get_stats(total_patients)
    # 0 decimal places
    stats_df["Value"] = stats_df["Value"].apply(lambda x: f"{x:.0f}")
    st.table(stats_df)
    st.write(f"Late Patients: ")
    stats_df =get_stats(late_patients_list)
    # 0 decimal places 
    stats_df["Value"] = stats_df["Value"].apply(lambda x: f"{x:.0f}")
    st.table(stats_df)
    st.write(f"Late Patients Percentage: ")
    stats_df =get_stats(late_percentage)
    # 2 decimal places and add % sign
    stats_df["Value"] = stats_df["Value"].apply(lambda x: f"{x:.2f}%")
    st.table(stats_df)
    st.write(f"Average Wait Time (days):")
    stats_df =get_stats(avg_wait_time)
    # 2 decimal places
    stats_df["Value"] = stats_df["Value"].apply(lambda x: f"{x:.2f}")
    st.table(stats_df)

    for choose_run in  list(range(num_run)):
        appointment_schedule = appointment_schedule_list[choose_run]
        late_patients = late_patients_list[choose_run]
        wait_times = wait_times_list[choose_run]
        available_slots = available_slots_list[choose_run]

        cola, colb, colc, cold, cole, colf = st.columns(6)
        with cola:
            # Create a histogram for appointment days
            fig = px.histogram(
                x=appointment_schedule, 
                title="Appointment Days", 
                nbins=num_days, 
            )
            fig.update_layout(
                title=dict(font=dict(size=14)),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        with colb:
            # Create a line chart for available slots
            fig = px.line(
                y=available_slots, 
                title="Available Slots", 
            )
            fig.update_layout(
                title=dict(font=dict(size=14)),
                margin=dict(l=10, r=10, t=30, b=10),
            )

            st.plotly_chart(fig, use_container_width=True)
        with colc:
            # sum of available slots
            fig = px.line(
                y=np.cumsum(available_slots), 
                title="Cumulative Available Slots", 
            )

            fig.update_layout(
                title=dict(font=dict(size=14)),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        with cold:
            # Create a histogram for wait times
            fig = px.histogram(
                x=[v for v in wait_times if v >0], 
                title="Wait Times", 
                nbins=30, 
            )
            fig.update_layout(
                title=dict(font=dict(size=14)),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        with cole:
            # accumulate late patients
            accumulated_late_patients = np.cumsum([1 if v > 0 else 0 for v in wait_times])
            fig = px.line(
                y=accumulated_late_patients, 
                title="Accumulated Late Patients", 
            )

            fig.update_layout(
                title=dict(font=dict(size=14)),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        with colf:
            # Accumulate waiting times
            accumulated_wait_times = np.cumsum(wait_times)
            fig = px.line(
                y=accumulated_wait_times, 
                title="Accumulated Wait Times", 
            )

            fig.update_layout(
                title=dict(font=dict(size=14)),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        



        





