from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.config import OPENAI_API_KEY

llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")

def generation_agent(input_data):
    context = input_data.get('context', 'No context provided.')
    query = input_data.get('query', 'No query provided.')

    prompt = ChatPromptTemplate.from_template(
        "Given the context:\n{context}\nAnswer the following question:\n{query}"
    )
    chain = prompt | llm
    input_data['response'] = chain.invoke({"context": context, "query": query}).content

    print("Generated Response:", input_data['response'])  # debug print clearly
    return input_data
