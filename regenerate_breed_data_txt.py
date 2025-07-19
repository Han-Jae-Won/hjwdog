import re

def regenerate_breed_data_txt(input_file="dog_care_guide_120breeds.txt", output_file="dog_care_guide_120breeds.txt"):
    """
    Regenerates the dog_care_guide_120breeds.txt file with proper formatting for parsing.
    """
    breeds = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^\d+-\d+\.\s*(.+)", line.strip())
            if match:
                breeds.append(match.group(1).strip())

    new_content = []
    new_content.append("\n[1. 강아지 기본 케어 정보]")
    new_content.append("- 식사: 연령별 적절한 사료, 깨끗한 물 제공")
    new_content.append("- 운동: 산책, 놀이, 품종별 활동량에 맞춤")
    new_content.append("- 위생: 주기적 목욕, 양치, 발톱/귀 관리")
    new_content.append("- 정기검진: 예방접종, 구충, 건강검진 필수")
    new_content.append("- 안전: 실종 방지, 위험물 주의, 편안한 환경 제공")
    new_content.append("\n[2. 상황별 케어]")
    new_content.append("- 아플 때: 식욕·활력 감소, 구토·설사, 이상행동 시 신속히 동물병원 방문")
    new_content.append("- 혼자 둘 때: 불안 완화 장난감, 라디오 등 환경 개선")
    new_content.append("- 새끼강아지: 잦은 식사, 체온·위생 관리, 예방접종")
    new_content.append("- 노령견: 관절·치아 관리, 미끄럼 방지, 부드러운 사료, 운동량 조절")
    new_content.append("- 입양 초기: 환경 적응 지원, 안정감 주는 공간 마련, 강압적 행동 자제")
    new_content.append("- 계절별: 여름(더위/열사병 주의), 겨울(보온/산책시간 조절)")
    new_content.append("\n[3. 품종별 케어]\n")

    breed_counter = 1
    for breed_name in breeds:
        # Extract only the Korean name for generating content
        korean_breed_name_match = re.match(r"(.+)\s*\(.+\)", breed_name)
        korean_breed_name = korean_breed_name_match.group(1).strip() if korean_breed_name_match else breed_name

        new_content.append(f"3-{breed_counter}. {breed_name}")
        new_content.append(f"- 기본특징: {korean_breed_name}는(은) {korean_breed_name}의 일반적인 특징을 가지고 있습니다. 크기, 외모, 주요 특징 등을 포함합니다.")
        new_content.append(f"- 건강상 유의점: {korean_breed_name}는(은) 유전 질환 및 품종별 주요 질병(관절, 치아, 심장, 피부 등)에 주의해야 합니다. 정기적인 건강 검진이 중요합니다.")
        new_content.append(f"- 털 관리: {korean_breed_name}는(은) 털 빠짐 정도와 털 종류에 따라 주기적인 빗질 및 목욕, 미용이 필요합니다. 털 엉킴 방지 및 피부 건강 유지를 위해 꾸준한 관리가 중요합니다.")
        new_content.append(f"- 운동/활동: {korean_breed_name}는(은) 품종별 활동량에 맞춰 매일 충분한 산책과 실내외 놀이로 에너지를 소모해야 합니다. 지루함을 느끼지 않도록 다양한 활동을 제공하는 것이 좋습니다.")
        new_content.append(f"- 성격: {korean_breed_name}는(은) 품종 고유의 성향(활발함/온순함/경계심/애교 등)을 가지고 있습니다. 사회화 훈련을 통해 다른 사람이나 동물들과 잘 지낼 수 있도록 돕는 것이 중요합니다.")
        new_content.append(f"- 기타: {korean_breed_name}를(을) 키울 때는 환경, 기후, 가족 상황에 맞춘 세심한 관리가 중요합니다. 필요한 경우 전문가의 도움을 받는 것도 좋습니다.")
        new_content.append("") # Add an empty line for readability between breeds
        breed_counter += 1

    new_content.append("[4. 기타]")
    new_content.append("- 놀이법: 퍼즐, 터그, 공던지기 등 다양한 놀이 제공")
    new_content.append("- 예방접종: 연령별 예방접종 일정 확인 및 준수")
    new_content.append("- 건강체크: 식욕, 배변, 호흡, 피부, 행동 등 이상 신호 상시 관찰")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_content))
    
    print(f"Successfully regenerated '{output_file}' with corrected formatting.")

if __name__ == "__main__":
    regenerate_breed_data_txt()
