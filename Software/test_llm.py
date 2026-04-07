# test_llm.py
from my_llm_module import LocalLLM, speak

llm = LocalLLM()
speak("Hallo Welt")
print(llm.generate("Test Prompt"))