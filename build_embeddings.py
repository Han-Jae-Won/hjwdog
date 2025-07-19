import numpy as np
import json
from sentence_transformers import SentenceTransformer
import faiss

def build_and_save_rag_embeddings(data_file="dog_care_guide_120breeds.txt", model_name='jhgan/ko-sroberta-multitask'):
    """
    Reads a text file, generates embeddings for each line, and saves the embeddings and texts for RAG.
    """
    print(f"Loading model for RAG: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Reading data from: {data_file} for RAG...")
    with open(data_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Found {len(texts)} lines to process for RAG.")
    print("Generating RAG embeddings... (This may take a while)")
    
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    np.save("faq_embeddings.npy", embeddings)
    with open("faq_texts.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    print("Successfully saved RAG embeddings to 'faq_embeddings.npy'")
    print("Successfully saved RAG texts to 'faq_texts.json'")

def build_and_save_breed_embeddings(breed_data_file="dog_breeds_data.json", model_name='jhgan/ko-sroberta-multitask'):
    """
    Reads breed data JSON, generates embeddings for combined features, and saves them.
    """
    print(f"Loading model for breed features: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Reading breed data from: {breed_data_file}...")
    try:
        with open(breed_data_file, "r", encoding="utf-8") as f:
            breed_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {breed_data_file} not found. Please ensure it's generated.")
        return

    breed_feature_texts = []
    breed_names = []
    
    # Combine all relevant features into a single text for each breed
    for breed_name, info in breed_data.items():
        combined_text = f"품종명: {info.get('name', '')}. "
        combined_text += f"기본특징: {info.get('기본특징', '')}. "
        combined_text += f"건강상 유의점: {info.get('건강상 유의점', '')}. "
        combined_text += f"털 관리: {info.get('털 관리', '')}. "
        combined_text += f"운동/활동: {info.get('운동/활동', '')}. "
        combined_text += f"성격: {info.get('성격', '')}. "
        combined_text += f"기타: {info.get('기타', '')}. "
        
        breed_feature_texts.append(combined_text)
        breed_names.append(breed_name)

    print(f"Found {len(breed_feature_texts)} breed feature texts to process.")
    print("Generating breed feature embeddings... (This may take a while)")
    
    breed_embeddings = model.encode(breed_feature_texts, show_progress_bar=True, convert_to_numpy=True)

    np.save("breed_feature_embeddings.npy", breed_embeddings)
    with open("breed_feature_names.json", "w", encoding="utf-8") as f:
        json.dump(breed_names, f, ensure_ascii=False, indent=2)

    print("Successfully saved breed feature embeddings to 'breed_feature_embeddings.npy'")
    print("Successfully saved breed names to 'breed_feature_names.json'")

if __name__ == "__main__":
    # Build RAG embeddings
    build_and_save_rag_embeddings()
    
    # Build breed feature embeddings
    build_and_save_breed_embeddings()