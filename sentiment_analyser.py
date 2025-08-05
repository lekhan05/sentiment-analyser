import getpass
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm = ChatGroq(model="llama-3.1-8b-instant")

parse_template = PromptTemplate(
    input_variables=["raw_feedback"],
    template="Parse and clean the following customer feedback for key information:\n\n{raw_feedback}",
)

summary_template = PromptTemplate(
    input_variables=["parsed_feedback"],
    template="Summarise this customer feedback in once consice sentence:\n\n{parsed_feedback}",
)

sentiment_template = PromptTemplate(
    input_variables=["feedback"],
    template="Determine the sentiment of this feedback and reply in one word as either 'Positive', 'Neutral', or 'Negative':\n\n{feedback}",
)

thankyou_template = PromptTemplate(
    input_variables=["feedback"],
    template="Given the feedback, draft a thank you message for the user and request them to leave a positive rating on our webpage:\n\n{feedback}",
)

details_template = PromptTemplate(
    input_variables=["feedback"],
    template="Given the feedback, draft a message for the user and request them provide more details about their concern:\n\n{feedback}",
)

apology_template = PromptTemplate(
    input_variables=["feedback"],
    template="Given the feedback, draft an apology message for the user and mention that their concern has been forwarded to the relevant department:\n\n{feedback}",
)

thankyou_chain = thankyou_template | llm | StrOutputParser()
details_chain = details_template | llm | StrOutputParser()
apology_chain = apology_template | llm | StrOutputParser()


def route(info):
    if "positive" in info["sentiment"].lower():
        return thankyou_chain
    elif "negative" in info["sentiment"].lower():
        return apology_chain
    else:
        return details_chain


format_parsed_output_runnable = RunnableLambda(
    lambda output: {"parsed_feedback": output}
)

summary_chain = parse_template | llm | format_parsed_output_runnable | summary_template | StrOutputParser()
sentiment_chain = sentiment_template | llm | StrOutputParser()

user_feedback = input("Enter a user feedback for analysis: ")

format_parsed_output = RunnableLambda(lambda output: {"parsed_feedback": output})

summary_chain = parse_template | llm | format_parsed_output | summary_template | llm | StrOutputParser()
sentiment_chain = sentiment_template| llm | StrOutputParser()

summary = summary_chain.invoke({'raw_feedback' : user_feedback})
sentiment = sentiment_chain.invoke({'feedback': summary})

print("The summary of the user's message is:", summary)
print("The sentiment was classifed as:", sentiment)

full_chain = {"feedback": lambda x: x['feedback'], 'sentiment' : lambda x : x['sentiment']} | RunnableLambda(route)
print(full_chain.invoke({'feedback': summary, 'sentiment': sentiment}))
