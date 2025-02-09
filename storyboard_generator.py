import os
import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd
from tabulate import tabulate
import openai

#Validate the OpenAI API Key
class OpenAIKeyValidator:
    @staticmethod
    def validate(api_key):
        try:
            openai.api_key = api_key
            openai.models.list()
            return True
        except:
            return False

class DocumentParser:
    def __init__(self, chunk_size=1000, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    def parse(self, loader):
        #Load and split the document into appropriate chunks.
        print("Parsing the document...")
        pages = loader.load_and_split()
        vectorstore = InMemoryVectorStore.from_documents(pages, embedding=self.embeddings)
        retriever = vectorstore.as_retriever()
        text_question_gen = ''.join(page.page_content for page in pages)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        text_chunks = text_splitter.split_text(text_question_gen)
        docs = [Document(page_content=t) for t in text_chunks]
        return docs

class PresentationGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate(self, docs):
        #Generate slide using Refine method to handle large amount of data iteratively.
        print("Generating slides...")
        prompt_template = """
        You are an expert in creating presentation slides for MOOC based on context provided.
        Your goal is to extract key-points, explanations and syntax for the presentation:
        -----------
        {text}
        -----------
        Create the content for the presentation. Make sure not to lose any important information.
        Points:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=['text'])
        refine_template = """
        You are an expert in creating presentation slides for MOOC storyboard in the following format:
        Slide Content
        Slide 1: Title Slide
        (Course Title)
        Slide 2: Outline 
        (Topics Covered)
        Slide 3: Key Concept 1 
        (Definition): AI
        Your goal is to prepare presentation slide points(optimal number of slides, at least 5).
        Make sure the course outline matches the content which is presented later on.
        Refine the contents from the given text: {existing_answer}.
        INCLUDE the explanations and syntax in the slide content to make it informative.
        DO NOT generate any other text than the said task.
        ------------
        {text}
        -----------
        Given the new context, refine the points.
        If the context is not helpful, please create content based on the prepared points. 
        """
        refine_prompt = PromptTemplate(input_variables=['existing_answer', 'text'], template=refine_template)
        chain = load_summarize_chain(llm=self.llm, chain_type='refine', verbose=False, question_prompt=prompt, refine_prompt=refine_prompt)
        return chain.invoke(docs)

class DialogueGenerator:
    def __init__(self, parse_llm):
        self.parse_llm = parse_llm

    class Presentation(BaseModel):
        #Pydantic model to store the slide contents and the dialogues.
        slides: list = Field(description="All the contents of slides with the following keys only: slideNumber and content. NO SUBKEYS")
        dialogues: list = Field(description="All the lecturer dialogues with the following keys only: slideNumber and dialogue")

    def generate(self, slide_content):
        #Generate the lecturer dialogues relevant to the slides which was generated earlier and store it in Pydantic format.
        print("Generating dialogues...")
        parser = PydanticOutputParser(pydantic_object=self.Presentation)
        format_instruction = parser.get_format_instructions()
        template_string = """
        Your task is to generate the lecturer dialogues explaining all the terms and concepts in the slides.
        Use a elegant and pragmatic approach to make the dialogues complementing the slides. Add some extra information relevant to the content as well and make sure enough dialogue is there to present the slides propoerly.
        Seperate the slides and lecturer dialogues.
        {text}
        Maintain the line breaks and formats.
        Convert it into the given unstructured Pydantic Format.
        {format_instruction}
        """
        prompt = ChatPromptTemplate.from_template(template=template_string)
        message = prompt.format_messages(text=slide_content['output_text'], format_instruction=format_instruction)
        output = self.parse_llm.invoke(message)
        return parser.parse(output.content)

class StoryboardConverter:
    @staticmethod
    def convert_to_table(presentation):
        #Join the corresponding slides to its lecturer dialogues based on slideNumber.
        slides = presentation.slides
        dialogues = presentation.dialogues
        combined = [{'Slide Content': slide['content'], 'Dialogue': next((d['dialogue'] for d in dialogues if d['slideNumber'] == slide['slideNumber']), '')} for slide in slides]
        return pd.DataFrame(combined)

    @staticmethod
        #Save as the DataFrame as csv.
    def save_to_csv(df, filename="storyboard.csv"):
        df.to_csv(filename, index=False)
        print(f"Saved the storyboard as {filename}")

def main():
    while True:
        api_key = getpass.getpass("Enter your OpenAI API Key: ")
        if OpenAIKeyValidator.validate(api_key):
            os.environ["OPENAI_API_KEY"] = api_key
            print("API key is valid and has been set.")
            break
        else:
            print("Invalid API key. Please try again.")
    #Use of GPT-4o model to generate slides and o1 model to parse the output and generate lecturer dialogues.
    llm = ChatOpenAI(model="gpt-4o")
    parse_llm = ChatOpenAI(model='o1')

    #Input documents.
    loader = PyPDFLoader("manual.pdf")
    parser = DocumentParser()
    docs = parser.parse(loader)

    presentation_gen = PresentationGenerator(llm)
    slides = presentation_gen.generate(docs)

    dialogue_gen = DialogueGenerator(parse_llm)
    presentation = dialogue_gen.generate(slides)

    converter = StoryboardConverter()
    df = converter.convert_to_table(presentation)
    #Save to csv
    converter.save_to_csv(df)

if __name__ == "__main__":
    main()