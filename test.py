from langchain_ollama import ChatOllama

# Connect directly to your local Ollama instance
llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
    temperature=0.2
)

# Test query
response = llm.invoke("Confirm that you are running locally on my M1 Mac!")
print(response.content)
