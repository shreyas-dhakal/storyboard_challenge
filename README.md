
  
  
  

# MOOC Storyboard Generator

A simple AI agent-based program to generate presentation slides and lecturer dialogues from a manual.

  

## Setup

 1. Clone the project into your local machine and navigate into the directory.



```

git clone https://github.com/shreyas-dhakal/storyboard_challenge

cd ./storyboard_challenge

```

  
  

2. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install libraries from requirements.txt

  

```bash

pip  install  -r  requirements.txt

```

  

3. Then, run the program.

```bash

python  storyboard_generator.py

```

4. Enter your OpenAI API Key into the prompt and you're good to go.

  

## Methodology

  

The program makes use of Langchain and OpenAI models like GPT-4o to create the slide contents, o1 model to generate the lecturer dialogues and text-embedding-3-large model for embedding the document.

A brief flow of how the code works:

 1. The models for embedding, parsing and generating are loaded.
 2. The document is loaded, parsed and a vectorstore is created to store the embedded document.
 3. The document is split into chunks.
 4. The splitted documents are first summarized iteratively to extract key points, headings and code snippets for the presentation and then, it is further refined to generate presentation slides that cover the important aspects of the document. This is all done by using multiple API calls to the GPT-4o model using a [Refine](https://python.langchain.com/v0.1/docs/use_cases/summarization/#option-3-refine) summarizing chain. The Refine method provides superior output than other summarization methods.
 5. To reduce API calls, the dialogues are generated using a o1 model and combined together with the slide contents to parse the output as a Pydantic custom object within a same LLM invocation.
 6. Finally, the output is exported as a CSV file by appending corresponding slide contents and lecturer dialogues based on its slide number.

  

## Output

The pre-run output is stored as storyboard.csv. To view the results quickly, I have stored the output as output_table.html as the code takes a long time to run.