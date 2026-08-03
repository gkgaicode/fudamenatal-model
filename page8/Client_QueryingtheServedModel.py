import openai

# Bind the client target pointer configuration to point directly at your vLLM local instance
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-placeholder-not-needed-for-local-vllm"
)

def query_custom_foundation_api(user_prompt):
    """Sends a completion request utilizing OpenAI-standard ChatCompletion schemas."""
    response = client.chat.completions.create(
        model="./my_hf_model_awq_4bit", # Must match the path string passed to vllm script
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=150
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # Test your deployment end-to-end
    # output = query_custom_foundation_api("Explain quantization in simple terms.")
    # print(f"Model Response:\n{output}")
    pass
