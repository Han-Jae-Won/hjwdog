import re

def update_specific_breed_data(file_path="dog_care_guide_120breeds.txt"):
    """
    Updates specific breed information in the dog_care_guide_120breeds.txt file.
    """
    detailed_breed_info = {
        "푸들 (Poodle)": {
            "기본특징": "푸들은 우아하고 지능적인 견종으로, 곱슬거리는 털과 활발한 성격이 특징입니다. 토이, 미니어처, 스탠더드 등 다양한 크기가 있습니다.",
            "건강상 유의점": "슬개골 탈구, 진행성 망막 위축증, 애디슨병 등에 취약할 수 있으므로 정기적인 검진이 중요합니다.",
            "털 관리": "털 빠짐이 적지만, 털이 쉽게 엉키므로 매일 빗질하고 4~6주마다 미용이 필요합니다.",
            "운동/활동": "지능이 높아 정신적 자극과 함께 충분한 산책 및 놀이 활동이 필요합니다.",
            "성격": "매우 영리하고 훈련 능력이 뛰어나며, 사람과 교감하는 것을 좋아합니다. 활발하고 장난기가 많습니다.",
            "기타": "알레르기가 있는 사람에게도 비교적 적합하며, 다양한 훈련에 잘 반응합니다."
        },
        "말티즈 (Maltese)": {
            "기본특징": "말티즈는 작고 우아한 외모에 길고 하얀 털이 특징인 견종입니다. 애교 많고 활발한 성격으로 실내견으로 인기가 많습니다.",
            "건강상 유의점": "슬개골 탈구, 눈물 자국, 심장 질환 등에 주의해야 합니다.",
            "털 관리": "털 빠짐은 적지만, 길고 가는 털이 쉽게 엉키므로 매일 빗질하고 눈물 자국 관리가 필요합니다.",
            "운동/활동": "실내에서 충분한 활동으로도 만족하지만, 짧은 산책이나 놀이를 통해 에너지를 발산하는 것이 좋습니다.",
            "성격": "애교가 많고 사람을 잘 따르며, 활발하고 장난기가 많습니다. 때로는 고집이 있을 수 있습니다.",
            "기타": "작은 체구로 아파트나 주택 등 실내 환경에 잘 적응합니다."
        },
        "골든 리트리버 (Golden Retriever)": {
            "기본특징": "골든 리트리버는 온순하고 친근한 성격의 대형견으로, 황금색 털과 부드러운 인상이 특징입니다.",
            "건강상 유의점": "고관절 이형성증, 팔꿈치 이형성증, 암 등에 취약할 수 있습니다.",
            "털 관리": "털 빠짐이 많은 편이므로 주 2~3회 빗질이 필요하며, 특히 털갈이 시기에는 더욱 신경 써야 합니다.",
            "운동/활동": "활동량이 매우 많아 매일 충분한 산책, 달리기, 수영 등 격렬한 운동이 필요합니다.",
            "성격": "매우 온순하고 인내심이 강하며, 사람과 다른 동물들에게 친근합니다. 훈련 능력이 뛰어납니다.",
            "기타": "안내견, 치료견 등으로 많이 활동하며, 가족견으로도 매우 적합합니다."
        }
    }

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    current_breed_name = None
    in_breed_section = False

    for line in lines:
        stripped_line = line.strip()

        # Check for breed header
        breed_match = re.match(r"^\d+-\d+\.\s*(.+)", stripped_line)
        if breed_match:
            current_breed_name = breed_match.group(1).strip()
            in_breed_section = current_breed_name in detailed_breed_info
            new_lines.append(line) # Always keep the breed header
            continue

        if in_breed_section:
            # Check for category line
            for category, content in detailed_breed_info[current_breed_name].items():
                if stripped_line.startswith(f"- {category}:"):
                    new_lines.append(f"- {category}: {content}\n")
                    break
            else: # If no category matched, it's a generic line, skip it
                pass
        else:
            new_lines.append(line) # Keep lines outside of specific breed sections

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    
    print(f"Successfully updated specific breed data in '{file_path}'.")

if __name__ == "__main__":
    update_specific_breed_data()
