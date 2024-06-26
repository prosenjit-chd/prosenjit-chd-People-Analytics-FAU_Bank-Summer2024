import pandas as pd
from pulp import *
from io import StringIO

def optimize_teller_staffing(csv_filepath):
    # Read the CSV data into a DataFrame
    data = pd.read_csv(csv_filepath, index_col=0)

    # Define the service rate: number of customers a single employee can handle per hour
    service_rate = 8

    # Calculate the required number of employees for each hour (ceiling division)
    data['Required_Employees'] = (data['Avg_Customer_Number'] / service_rate).apply(lambda x: -(-x // 1))

    # Define decision variables for the number of employees per shift
    num_employees_shift1 = LpVariable("Num_Employees_Shift1", lowBound=0, cat='Integer')
    num_employees_shift2 = LpVariable("Num_Employees_Shift2", lowBound=0, cat='Integer')

    # Create a linear programming problem to minimize the total number of employees
    problem = LpProblem("Teller_Staffing_Optimization", LpMinimize)

    # Objective function: Minimize the sum of employees in both shifts
    problem += num_employees_shift1 + num_employees_shift2, "Minimize_Total_Employees"

    # Constraints: Ensure sufficient employees are available for each hour's demand
    for hour, row in data.iterrows():
        if row['Shift 1'] == 'X':
            problem += num_employees_shift1 >= row['Required_Employees'], f"Shift1_Hour_{hour}"
        if row['Shift 2'] == 'X':
            problem += num_employees_shift2 >= row['Required_Employees'], f"Shift2_Hour_{hour}"

    # Solve the optimization problem
    problem.solve()

    # Output the results
    print("Status:", LpStatus[problem.status])
    print("Total Employees Needed:", value(problem.objective))
    print(f"Shift 1: Employees Needed = {int(num_employees_shift1.value())}")
    print(f"Shift 2: Employees Needed = {int(num_employees_shift2.value())}")

    # Print detailed solver information
    print(f"Total time (CPU seconds): {problem.solutionCpuTime:.2f} (Wallclock seconds): {problem.solutionCpuTime:.2f}")

if __name__ == "__main__":
    csv_filepath = "fau_bank_shifts.csv"
    optimize_teller_staffing(csv_filepath)
