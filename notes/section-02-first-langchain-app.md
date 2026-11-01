Section 2: Your First LangChain App

Course: LangChain & LangGraph AI Engineer: Agents, Agentic RAG & MCP
Instructor: Eden Marco
Section: 2 — Your First LangChain App
Status: Learning and Revision
Primary Goal: Understand how to build an LLM application using LangChain, Python, and a locally running model.

⸻

Table of Contents

1. Learning Objectives
2. What Are We Building?
3. Understanding the Architecture
4. What Is an LLM?
5. What Is LangChain?
6. Understanding Ollama and Gemma
7. Setting Up the Python Project
8. Understanding the Application Code
9. Prompt and PromptTemplate
10. Chat Models and ChatOllama
11. Understanding LCEL
12. Understanding invoke()
13. Complete Application
14. How Data Flows Through the Application
15. Understanding Output Parsers
16. Running an LLM Without LangChain
17. Common Errors and Troubleshooting
18. Practical Exercises
19. Revision Questions
20. Key Takeaways

⸻

1. Learning Objectives

The objective of this section is to understand how a basic LLM-powered application works.

By the end of this section, I should be able to:

* Explain what an LLM is.
* Explain what LangChain does.
* Connect a locally running LLM to a Python application.
* Understand the purpose of a prompt.
* Create a reusable prompt using PromptTemplate.
* Connect a model using ChatOllama.
* Build a chain using LangChain Expression Language (LCEL).
* Execute a chain using invoke().
* Understand how information moves through a LangChain application.
* Understand the difference between Ollama, Gemma, and LangChain.
* Build a simple LLM application without copying the original code.

⸻

2. What Are We Building?

The first application takes information about a person and generates a summary.

The application should produce:

1. A short summary of the person.
2. Two interesting facts about the person.

For this exercise, the information is about Elon Musk.

The application does not need to search the internet. All the information is already provided to the model.

Example input

Elon Musk is an entrepreneur known for his work in
electric vehicles and space technology.
He founded SpaceX and became CEO of Tesla.

Expected output

Summary:
Elon Musk is an entrepreneur known for his work
in electric vehicles and space technology.
Interesting Facts:
1. He founded SpaceX.
2. He became CEO of Tesla.

The actual response depends on the information provided and the model’s generation.

What is the purpose of this application?

This is a simple example of how we can use an LLM to process information.

The same pattern can be extended to:

* Summarize job descriptions.
* Summarize technical documents.
* Extract important information from resumes.
* Generate descriptions of products.
* Summarize customer feedback.
* Answer questions about provided documents.

The core idea is that we provide information and instructions, and the LLM generates a response based on them.

⸻

3. Understanding the Architecture

The application has four main components:

1. Input information.
2. Prompt template.
3. Chat model.
4. Output.

Architecture diagram

                  Python Application
                          |
                          |
                          v
                  Input Information
                          |
                          v
                   PromptTemplate
                          |
                          v
                  Formatted Prompt
                          |
                          v
                     ChatOllama
                          |
                          v
                    Ollama Server
                          |
                          v
                     Gemma Model
                          |
                          v
                  Generated Response
                          |
                          v
                     Python Output

Explanation

Input Information

The information that we want the model to process.

In this application, it is the information about Elon Musk.

PromptTemplate

The component that combines the instructions with the information.

ChatOllama

The LangChain integration that communicates with the Ollama server.

Gemma

The actual language model that generates the response.

Important distinction

The model does not automatically know which information from our Python variable is important.

We provide the information and instructions through a prompt.

The model processes the prompt and generates a response.

⸻

4. What Is an LLM?

LLM stands for Large Language Model.

An LLM is an AI model trained on large amounts of text and other data to understand and generate language.

Examples of LLMs include:

* GPT models.
* Gemini models.
* Claude models.
* Llama models.
* Gemma models.

What can an LLM do?

An LLM can perform tasks such as:

* Summarization.
* Question answering.
* Text generation.
* Translation.
* Classification.
* Information extraction.
* Code generation.

How does an LLM generate a response?

At a high level:

1. It receives an input.
2. It processes the input using its learned parameters.
3. It generates an output, typically one token at a time.

A token can represent a word, part of a word, or another text unit.

For example:

Input:
Explain Java.
LLM:
Java is a programming language...

The model generates a response based on its training and the current input.

Important concept

An LLM is not the same thing as an application.

The LLM is the model that generates responses.

A Python application can use an LLM to perform a specific task.

⸻

5. What Is LangChain?

LangChain is a framework for building applications powered by language models.

It provides reusable components for connecting:

* Prompts.
* Chat models.
* Output parsers.
* Tools.
* Document processing.
* Retrieval systems.
* Other application components.

Why do we need LangChain?

We can communicate with an LLM directly using Python.

For a simple application, direct communication may be enough.

However, applications become more complicated when we need to:

* Process information through multiple steps.
* Reuse prompts.
* Connect different models.
* Retrieve information from documents.
* Call external tools.
* Build agent workflows.

LangChain helps organize these operations.

Example without LangChain

Python
   |
   v
Create Prompt
   |
   v
Send HTTP Request
   |
   v
Receive Model Response
   |
   v
Process Response

Example with LangChain

Input
  |
  v
PromptTemplate
  |
  v
Chat Model
  |
  v
Output Parser
  |
  v
Final Output

Important distinction

LangChain is not an LLM.

It is a framework that helps us build applications using LLMs.

⸻

6. Understanding Ollama and Gemma

This application uses Gemma 3 (270M) through Ollama.

These are two different things.

What is Gemma?

Gemma is a family of language models.

In this application, the model is:

gemma3:270m

The model generates the actual response.

What is Ollama?

Ollama is software that runs and serves language models locally.

It provides an interface through which Python applications can communicate with the model.

What is ChatOllama?

ChatOllama is a LangChain integration for communicating with a model served by Ollama.

It allows us to use Ollama models inside LangChain applications.

Complete relationship

Python Application
        |
        v
     LangChain
        |
        v
    ChatOllama
        |
        v
  Ollama Server
        |
        v
    Gemma Model

Does the model run inside VS Code?

No.

VS Code is the development environment.

Ollama runs the model locally on the Mac.

The Python application communicates with the Ollama server.

The model uses the available hardware resources of the Mac to generate responses.

Do we need an OpenAI API key?

No.

This application uses a local model through Ollama.

An OpenAI API key is not required.

⸻

7. Setting Up the Python Project

I am using uv to manage the Python project and its dependencies.

Step 1: Create the project

uv init langchain-course

If the project already exists, there is no need to initialize it again.

Step 2: Enter the project directory

cd langchain-course

Step 3: Create a virtual environment

uv venv

This creates a virtual environment in the .venv directory.

A virtual environment isolates the project’s Python dependencies from other projects.

Step 4: Install dependencies

uv add langchain langchain-ollama python-dotenv

These packages provide:

Package	Purpose
langchain	Core LangChain functionality
langchain-ollama	Ollama integration
python-dotenv	Loads environment variables from a .env file

Step 5: Run the application

uv run main.py

uv run executes the Python file using the project’s managed environment.

Project structure

langchain-course/
│
├── .venv/
│
├── main.py
│
├── pyproject.toml
│
├── uv.lock
│
└── notes/
    └── section-02-first-langchain-app.md

⸻

8. Understanding the Application Code

The application contains several important components.

Before understanding the entire program, we should understand each component individually.

Imports

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

load_dotenv()

This loads environment variables from a .env file into the Python environment.

Environment variables are useful for storing configuration values.

For example:

API_KEY=your_api_key

However, the current local Ollama application does not require an OpenAI API key.

PromptTemplate

from langchain_core.prompts import PromptTemplate

This imports the class used to create reusable prompts.

ChatOllama

from langchain_ollama import ChatOllama

This imports the LangChain integration for communicating with Ollama.

⸻

9. Prompt and PromptTemplate

What is a prompt?

A prompt is the input or instruction provided to an LLM.

Example:

prompt = "Explain Java microservices."

The model receives this prompt and generates a response.

What is a PromptTemplate?

A PromptTemplate is a reusable template containing variables.

Instead of creating a completely new prompt every time, we define a template and provide different values.

Example

from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple English."
)

Here:

{topic}

is a placeholder.

We can provide different values for this placeholder.

Example 1

template.format(topic="Kafka")

Output:

Explain Kafka in simple English.

Example 2

template.format(topic="Spring Boot")

Output:

Explain Spring Boot in simple English.

The same template can be reused for different inputs.

Why is this useful?

Suppose we are building an application that summarizes thousands of documents.

We can create one reusable prompt template and provide different document contents.

This avoids rewriting the entire prompt for every document.

The summary template

In our application:

summary_template = """
Given the following information about a person:
{information}
Create:
1. A short summary.
2. Two interesting facts.
"""

The variable:

{information}

will be replaced with the actual information about the person.

Creating the PromptTemplate

summary_prompt_template = PromptTemplate(
    input_variables=["information"],
    template=summary_template
)

The input_variables argument identifies the variables required by the template.

Here, the template expects one variable:

information

The template can then be used to format different information into the same prompt.

⸻

10. Chat Models and ChatOllama

What is a chat model?

A chat model is a model interface that accepts messages and generates responses.

Chat models commonly support messages with different roles, such as:

* System.
* User.
* Assistant.

For example:

System: You are a helpful assistant.
User: Explain Kafka.
Assistant: Kafka is a distributed event streaming platform...

The exact message structure depends on the model integration.

Connecting to Ollama

from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="gemma3:270m",
    temperature=0
)

This creates a chat model integration.

model

model="gemma3:270m"

This specifies the Ollama model to use.

The model must be available in the local Ollama installation.

temperature

temperature=0

Temperature controls the randomness of model generation.

A lower temperature generally produces more predictable responses.

A higher temperature generally allows more variation.

Temperature does not guarantee factual accuracy.

Important: model vs model_name

For the current ChatOllama integration, use:

ChatOllama(
    model="gemma3:270m"
)

Do not use:

ChatOllama(
    model_name="gemma3:270m"
)

The model field is required by the current integration.

⸻

11. Understanding LCEL

LCEL stands for LangChain Expression Language.

It is a way to compose multiple LangChain components into a processing pipeline.

Example

chain = summary_prompt_template | llm

The | operator connects the prompt template to the chat model.

The output from the prompt template becomes the input to the model.

Architecture

summary_prompt_template
           |
           v
          llm
           |
           v
      Model Response

Why is LCEL useful?

It allows us to create reusable pipelines.

Instead of manually calling every component, we can connect them together and execute the entire chain.

Extending the chain

We can add an output parser:

from langchain_core.output_parsers import StrOutputParser
chain = (
    summary_prompt_template
    | llm
    | StrOutputParser()
)

Now the pipeline has three components.

PromptTemplate
      |
      v
  ChatOllama
      |
      v
 StrOutputParser
      |
      v
  Plain String

This will be explained in more detail in the output parser section.

⸻

12. Understanding invoke()

The invoke() method executes a chain with the provided input.

Example:

response = chain.invoke({
    "information": information
})

The input is a dictionary.

The key:

information

must match the variable defined in the prompt template.

What happens when invoke() is called?

Step 1: Input is provided

{
    "information": "Information about Elon Musk"
}

Step 2: PromptTemplate formats the prompt

The placeholder:

{information}

is replaced with the supplied information.

Step 3: The formatted prompt is passed to the model

LangChain passes the formatted prompt to ChatOllama.

Step 4: Ollama processes the request

The Ollama server receives the request and runs the specified model.

Step 5: The model generates a response

Gemma processes the prompt and generates text.

Step 6: The response is returned

LangChain returns the model response to the Python application.

Important concept

invoke() is the execution method.

It does not mean that the model is being trained.

It means that the chain is being executed with the supplied input.

⸻

13. Complete Application

This is the complete version of the first application.

Save it as:

main.py
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
information = """
Elon Musk is an entrepreneur known for his work
in electric vehicles and space technology.
He founded SpaceX and became CEO of Tesla.
"""
summary_template = """
Given the following information about a person:
{information}
Create:
1. A short summary.
2. Two interesting facts.
Use simple English.
"""
summary_prompt_template = PromptTemplate(
    input_variables=["information"],
    template=summary_template
)
llm = ChatOllama(
    model="gemma3:270m",
    temperature=0
)
chain = summary_prompt_template | llm
def main():
    response = chain.invoke({
        "information": information
    })
    print(response.content)
if __name__ == "__main__":
    main()

Run the application

uv run main.py

The program will send the information to the local Gemma model and print the generated response.

⸻

14. How Data Flows Through the Application

This is the most important execution flow to remember.

Complete flow

1. Python starts
       |
       v
2. Input information is loaded
       |
       v
3. PromptTemplate is created
       |
       v
4. ChatOllama is initialized
       |
       v
5. Chain is created
       |
       v
6. chain.invoke() is called
       |
       v
7. Input dictionary is passed
       |
       v
8. PromptTemplate formats the prompt
       |
       v
9. ChatOllama sends the request
       |
       v
10. Ollama runs the model
       |
       v
11. Gemma generates a response
       |
       v
12. Response is returned to Python
       |
       v
13. print(response.content)

Backend analogy

As a Java backend developer, I can think of this application as a request-processing pipeline.

LangChain concept	Backend analogy
Input dictionary	Request payload
PromptTemplate	Request transformation
ChatOllama	Client for an external or local service
LLM	Processing service
Model response	Service response
Output parser	Response transformation
invoke()	Execute the processing pipeline

The analogy is not exact, but it helps visualize the flow.

⸻

15. Understanding Output Parsers

The model response is not always returned as a plain Python string.

For chat models, LangChain commonly returns an AI message object.

Example:

response = chain.invoke({
    "information": information
})

To access the generated text:

print(response.content)

What is an output parser?

An output parser transforms the model’s response into a desired format.

For example:

* Plain text.
* JSON.
* Structured data.
* Lists.
* Other application-specific formats.

StrOutputParser

from langchain_core.output_parsers import StrOutputParser

This parser extracts the text from the model’s response.

Example

chain = (
    summary_prompt_template
    | llm
    | StrOutputParser()
)

Now:

response = chain.invoke({
    "information": information
})

The response is a string.

We can print it directly:

print(response)

Before and after

Without the parser:

response = chain.invoke(input_data)
print(response.content)

With the parser:

response = chain.invoke(input_data)
print(response)

The parser makes the chain return the extracted text directly.

⸻

16. Running an LLM Without LangChain

We can also communicate with Ollama directly using Python.

LangChain is optional.

Architecture without LangChain

Python
   |
   v
HTTP Request
   |
   v
Ollama Server
   |
   v
Gemma Model
   |
   v
HTTP Response
   |
   v
Python

Example using requests

Install:

uv add requests

Python code:

import requests
OLLAMA_URL = "http://localhost:11434/api/generate"
def generate_response(prompt):
    payload = {
        "model": "gemma3:270m",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(
        OLLAMA_URL,
        json=payload
    )
    response.raise_for_status()
    return response.json()["response"]
if __name__ == "__main__":
    prompt = "Explain Java microservices"
    result = generate_response(prompt)
    print(result)

What is the difference?

Without LangChain, we manage the HTTP request and response ourselves.

With LangChain, we use reusable components such as prompt templates, chat models, and output parsers.

Both approaches can communicate with the same local model.

⸻

17. Common Errors and Troubleshooting

Error 1: ChatOllama model field missing

Example:

ValidationError: model
Field required

Incorrect:

ChatOllama(
    model_name="gemma3:270m"
)

Correct:

ChatOllama(
    model="gemma3:270m"
)

Reason:

The current integration expects the model parameter.

⸻

Error 2: Import cannot be resolved

Example:

Import "langchain_ollama" could not be resolved

Install the package:

uv add langchain-ollama

Then select the correct Python interpreter in VS Code.

The interpreter should point to the project’s virtual environment.

⸻

Error 3: Ollama server is unavailable

If the Ollama server is not running, the Python application may fail to connect.

Check running models:

ollama ps

If needed, start the Ollama service.

On a Mac using Homebrew:

brew services start ollama

Then execute the Python application again.

⸻

Error 4: Model not found

Check installed models:

ollama list

If the required model is missing, download it:

ollama pull gemma3:270m

Then run the application again.

⸻

Error 5: Python package installed in the wrong environment

Install dependencies using:

uv add langchain langchain-ollama

Run the application using:

uv run main.py

This helps ensure that the project’s dependencies are used consistently.

⸻

18. Practical Exercises

The best way to understand the concepts is to modify the application yourself.

Exercise 1: Change the input

Replace the information about Elon Musk with information about another person.

The application should still work without changing the prompt template.

Concept practiced:

PromptTemplate

Exercise 2: Change the output requirements

Modify the prompt to generate:

1. A short summary.
2. Three interesting facts.
3. Important achievements.

Concept practiced:

Prompt design.

Exercise 3: Add an output parser

Modify the chain to use:

StrOutputParser()

Concept practiced:

Output parsing and LCEL.

Exercise 4: Create a dynamic application

Allow the user to enter information through the terminal.

Example:

information = input("Enter information: ")

Pass the input to the chain.

Concept practiced:

Dynamic input and reusable prompts.

Exercise 5: Build without LangChain

Create a second application using the Ollama REST API directly.

Compare the code with the LangChain implementation.

Concept practiced:

Understanding the underlying communication mechanism.

⸻

19. Revision Questions

Try answering these questions without looking at the notes.

Basic concepts

1. What is an LLM?
2. What is LangChain?
3. What is the difference between an LLM and LangChain?
4. What is Ollama?
5. What is Gemma?
6. What is the purpose of ChatOllama?

Prompt engineering

7. What is a prompt?
8. What is a PromptTemplate?
9. Why do we use variables inside prompt templates?
10. What is the purpose of input_variables?

LCEL and execution

11. What is LCEL?
12. What does the | operator do?
13. What is a chain?
14. What happens when invoke() is called?
15. What is the difference between a chain and an LLM?

Output parsing

16. What is an output parser?
17. What does StrOutputParser() do?
18. Why do we use response.content?

Practical understanding

19. Can we use Ollama without LangChain?
20. Can we use LangChain with a cloud-hosted LLM?
21. Does the model run inside VS Code?
22. Why is model used instead of model_name in the current ChatOllama integration?

⸻

20. Key Takeaways

The most important concepts from this section are:

1. An LLM is the model that generates responses.
2. LangChain is a framework for building LLM-powered applications.
3. Ollama allows us to run and serve models locally.
4. Gemma is the model used in this application.
5. PromptTemplate creates reusable prompts.
6. ChatOllama connects LangChain to Ollama.
7. LCEL connects multiple components into a pipeline.
8. invoke() executes the pipeline.
9. Output parsers transform model responses into useful formats.
10. We can communicate with an LLM directly using Python without LangChain.

Final Revision Task

Close the original code.

Rebuild the entire application from scratch.

Make sure you can explain:

* What each component does.
* Why the components are connected in this order.
* How the input reaches the model.
* How the model response reaches Python.
* How to change the input without modifying the entire application.

The goal is not to memorize the code. The goal is to understand how each component works and why it exists.