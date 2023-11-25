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
                "A": ["immoral", "immoral"],
                "B": ["immoral", "moral"],
                "C": ["moral", "immoral"],
                "D": ["moral", "moral"]
            }

            answers = answer_map.get(answer_key, ["Unknown", "Unknown"])
            prompt_format = "{} According to moral standards in the US as of 2020, this is {}."
            # Creating the list of dictionaries
            for possible_answer in ["moral", "immoral"]:
                parsed_data.append({"statement": prompt_format.format(scenario_1, possible_answer), "label": answers[0] == possible_answer})
                parsed_data.append({"statement": prompt_format.format(scenario_2, possible_answer), "label": answers[1] == possible_answer})

    return parsed_data

p_data = parse_csv('./moral_scenarios_test.csv')

with open("moral_true_false.csv", 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(["statement", "label"])

    # Write the data
    for i in p_data:
        writer.writerow([i["statement"], i["label"]])
