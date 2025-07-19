import json

def parse_breed_data(input_file="dog_care_guide_120breeds.txt", output_file="dog_breeds_data.json"):
    """
    Parses dog breed information from a text file and saves it as a JSON file.
    """
    breeds_data = {}
    current_breed = None
    
    # Define the categories to look for, ensuring they match the file exactly
    categories = ["기본특징", "건강상 유의점", "털 관리", "운동/활동", "성격", "기타"]
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue # Skip empty lines

            # Check if it's a new breed (starts with a number and a breed name)
            # Example: "3-1. 푸들 (Poodle)"
            if line.startswith(tuple(str(i) for i in range(1, 100))) and ". " in line:
                parts = line.split(". ", 1) # Split by ". " to get the breed name
                if len(parts) > 1:
                    breed_full_name = parts[1].strip()
                    current_breed = breed_full_name
                    breeds_data[current_breed] = {"name": breed_full_name}
                continue

            # Check if it's a category line
            # Example: "- 기본특징: ..."
            found_category = False
            for cat in categories:
                if line.startswith("- " + cat + ":"):
                    content = line[len("- " + cat + ":"):].strip()
                    if current_breed: # Ensure we have a breed to assign to
                        breeds_data[current_breed][cat] = content
                    found_category = True
                    break # Found a category, move to next line
            
            # If it's not a new breed or a category, it's content for the current category
            # This part is removed because each category content is now assumed to be on a single line
            # and directly extracted when the category line is found.

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(breeds_data, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully parsed data from '{input_file}' and saved to '{output_file}'")

if __name__ == "__main__":
    parse_breed_data()
