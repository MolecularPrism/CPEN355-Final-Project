from datasets import load_dataset
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------
# 1. DATA SAMPLING
# -----------------------
def sample_amazon_hf(categories, output_dir="amazon_samples", k=2000):
    os.makedirs(output_dir, exist_ok=True)

    for category in categories:
        print(f"Processing {category}...")

        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            streaming=True,
            trust_remote_code=True
        )

        output_path = os.path.join(output_dir, f"{category}.jsonl")

        with open(output_path, "w") as out:
            count = 0

            for sample in dataset:
                text = sample.get("text")
                rating = sample.get("rating")

                if text and rating:
                    out.write(json.dumps({
                        "text": text,
                        "rating": rating
                    }) + "\n")
                    count += 1

                if count >= k:
                    break

    print("\nDataset File Creation Complete!")


# -----------------------
# 2. LABELING
# -----------------------
def label_from_rating(rating):
    if rating >= 4:
        return 1
    elif rating <= 2:
        return 0
    else:
        return None


# -----------------------
# 3. CLEAN TEXT
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------
# 4. LOAD + PREPROCESS
# -----------------------
def load_and_preprocess(data_dir="amazon_samples"):
    texts = []
    labels = []

    for file in os.listdir(data_dir):
        if file.endswith(".jsonl"):
            path = os.path.join(data_dir, file)

            with open(path, "r") as f:
                for line in f:
                    data = json.loads(line)

                    label = label_from_rating(data["rating"])
                    if label is None:
                        continue

                    cleaned = clean_text(data["text"])

                    texts.append(cleaned)
                    labels.append(label)

    return texts, labels

def run_preprocess(generate_dataset=True):
    categories = [
        "All_Beauty", "Amazon_Fashion", "Appliances",
        "Arts_Crafts_and_Sewing", "Automotive", "Baby_Products",
        "Beauty_and_Personal_Care", "Books", "CDs_and_Vinyl",
        "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
        "Digital_Music", "Electronics", "Gift_Cards",
        "Grocery_and_Gourmet_Food", "Handmade_Products",
        "Health_and_Household", "Health_and_Personal_Care",
        "Home_and_Kitchen", "Industrial_and_Scientific",
        "Kindle_Store", "Magazine_Subscriptions", "Movies_and_TV",
        "Musical_Instruments", "Office_Products",
        "Patio_Lawn_and_Garden", "Pet_Supplies", "Software",
        "Sports_and_Outdoors", "Subscription_Boxes",
        "Tools_and_Home_Improvement", "Toys_and_Games",
        "Video_Games", "Unknown"
    ]

    # STEP 1: Sample dataset
    if(generate_dataset):
        sample_amazon_hf(categories, k=2000)

    # STEP 2: Load + preprocess
    texts, labels = load_and_preprocess()

    return texts, labels


