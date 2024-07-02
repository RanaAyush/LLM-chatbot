This is a Chat bot build using Retrieval Augmented Generation

Hi,

I have used Ollama to locally run create embeddings and store in FAISS vector store.
I used qwen2:1.5b model beacause that's because other models require higher RAMs.

Steps to run:

  1-> Download and Install Ollama from https://ollama.com/
  
  2-> Open terminal and type ollama run qwen2:1.5b (or lama2 or any other llm model you want to run).
  
  3-> Again type ollama run nomic-embed-text (to download embeddings)
  
  4-> Navigate this git repository and open terminal and run - pip install requirements.txt
  
  5->Finally run streamlit server using - streamlit run app.py

Congratulations!!, it's working in your web browser

  Upload pdfs and click process and ask questions in prompt.
