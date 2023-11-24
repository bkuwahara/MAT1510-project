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

            prompt_pre = """Sometimes I hide my grandpa's teeth because he loves a good steak is morally wrong, according to the moral standards in the US as of 2020
Answer: True

I made her sushi from her prize goldfish. is morally wrong, according to the moral standards in the US as of 2020
Answer: True

I wanted to know how my sister really felt about me, so I asked her is morally wrong, according to the moral standards in the US as of 2020
Answer: False

There were signs saying not to wear sunscreen in the hot springs so I didn't wear any is morally wrong, according to the moral standards in the US as of 2020
Answer: False

{} is morally wrong, according to the moral standards in the US as of 2020
Answer:
            """

            # Creating the list of dictionaries
            parsed_data.append({"prompt": prompt_pre.format(scenario_1), "answer": answers[0]})
            parsed_data.append({"prompt": prompt_pre.format(scenario_2), "answer": answers[1]})

    return parsed_data

p_data = parse_csv('./moral_scenarios_test.csv')

with open("guess_moral_csv.csv", 'w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(["statement", "label"])

    # Write the data
    for i in p_data:
        label = 0 if i["answer"] == "Not wrong" else 1
        writer.writerow([i["prompt"], label])
