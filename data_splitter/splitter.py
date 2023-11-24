import csv

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

with open("moral_csv.csv", 'w') as file:
    # File path
    fp = './moral_scenarios_test.csv'
    p_data = parse_csv(fp)

    file.write("statement" + "," + "label" + "\n")
    for i in p_data:
        file.write(i["prompt"] + "," + str(1 if i["answer"] == "Not wrong" else 0) + "\n")
