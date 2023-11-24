import csv

# File path
file_path = './moral_scenarios_test.csv'

# Function to parse the file
def parse_csv(file_path):
    parsed_data = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Extracting the scenario prompt and answers
            prompt_base = row[0].split("?")[0] + "? "
            scenario_1 = row[0].split("|")[1].strip()
            scenario_2 = row[0].split("|")[2].split(".")[0].strip() + "."
            answer_key = row[-1]

            # print(scenario_1[-11:])
            if scenario_1[-11:] == " Scenario 2":
                # print("uhoh")
                scenario_1 = scenario_1[:-11]

            # Mapping answer keys to actual answers
            answer_map = {
                "A": ["Wrong", "Wrong"],
                "B": ["Wrong", "Not wrong"],
                "C": ["Not wrong", "Wrong"],
                "D": ["Not wrong", "Not wrong"]
            }

            answers = answer_map.get(answer_key, ["Unknown", "Unknown"])

            # Creating the list of dictionaries
            parsed_data.append({"prompt": scenario_1, "answer": answers[0]})
            parsed_data.append({"prompt": scenario_2, "answer": answers[1]})

    return parsed_data

# Parse the CSV file
parsed_scenarios = parse_csv(file_path)
print(parsed_scenarios[:8])
