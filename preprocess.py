from datasets import load_dataset
import json
import os

def sample_amazon_hf(categories, output_dir="amazon_samples", k=2000):
    os.makedirs(output_dir, exist_ok=True)

    for category in categories:
        print(f"Processing {category}...")

        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            streaming=True  
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