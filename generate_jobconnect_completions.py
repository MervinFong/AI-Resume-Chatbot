from dotenv import load_dotenv
import openai
import pandas as pd
import time
import os

load_dotenv()
# Initialize OpenAI API
print("Loaded key:", os.getenv("OPENAI_API_KEY"))

# Load prompts
df = pd.read_csv("jobconnect_finetune_1000.csv")
results = []

# Safety: limit to avoid overuse
MAX_ROWS = 1000  # adjust to 100/250 for test runs
model = "gpt-3.5-turbo"

for idx, row in df.iterrows():
    if idx >= MAX_ROWS:
        break

    prompt = row["prompt"]

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful career guidance and resume refinement assistant named JobConnect. Reply in markdown format. Use MBTI/resume/job knowledge when relevant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=512
        )

        reply = response.choices[0].message.content.strip()
        print(f"[{idx}] Prompt: {prompt}")
        print(f"     Response: {reply[:80]}...")

        results.append({"prompt": prompt, "completion": reply})
        time.sleep(1.2)  # Avoid hitting rate limits

    except Exception as e:
        print(f"❌ Error on row {idx}: {e}")
        results.append({"prompt": prompt, "completion": "ERROR"})

# Save output
out_df = pd.DataFrame(results)
out_df.to_csv("jobconnect_finetune_ready.csv", index=False)
print("✅ Finished generating completions.")
