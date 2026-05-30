import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_recommendation(research_report):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a stock trading analyst. Based on research reports, you decide exactly what to buy, how many shares, and at what price. Always respond in this exact format: SYMBOL: [ticker] SHARES: [number] REASON: [one sentence]"},
            {"role": "user", "content": f"Based on this research, give me one specific trade to make:\n{research_report}"}
        ]
    )
    
    decision = response.choices[0].message.content
    print("ANALYST AGENT DECISION:")
    print(decision)
    return decision

if __name__ == "__main__":
    test_report = "Microsoft and NVIDIA are top opportunities based on valuation and price position."
    analyze_recommendation(test_report)
    