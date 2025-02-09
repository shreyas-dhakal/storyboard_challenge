


# MOOC Storyboard Generator
A simple AI agent-based program to generate presentation slides and lecturer dialogues from a manual.

## How to run
Clone the project into your local machine and navigate into the directory.
```
git clone https://github.com/shreyas-dhakal/storyboard_challenge
cd ./storyboard_challenge
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install libraries from requirements.txt

```bash
pip install -r requirements.txt
```

Then, run the program.
```bash
python storyboard_generator.py
```
Enter your OpenAI API Key into the prompt and you're good to go.

## Methodology

The program uses Langchain Refine method which handles the large document effectively without overloading the API Requests. It iteratively extracts key points, code snippets and explanations that could be useful for the presentation with multiple API calls to the GPT-4o model. Then to further improve the contents of the slide, it goes through a refine process which reviews the extracted information from its earlier calls to generate a final presentation. Furthermore, the slide content is then sent as an input to o1 model. This model generates lecturer dialogues with much more reasoning than the GPT-4o model. The response received is a Pydantic object which contains keys: slides and dialogues. The data is finally converted into csv and exported.

## Output 
The pre-run output is stored as storyboard.csv. To view the results in a convinient way, I have stored the output as output_table.html.

