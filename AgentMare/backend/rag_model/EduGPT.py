from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

from custom_logging import setup_logger


class EduGPT:
    SYSTEM_TEMPLATE = """
Acționează ca un asistent educațional virtual pentru studenți. Ai acces doar la materialele de curs furnizate și trebuie să răspunzi exclusiv pe baza informațiilor din aceste materiale. Dacă o întrebare nu are răspuns în materialele disponibile, spune clar că nu ai informația necesară. Nu inventa răspunsuri și nu oferi informații din surse externe.
Materialele tale sunt:
<context>
{context}
</context>

Reguli de răspuns:
1. Fii clar și concis.
2. Explică conceptele folosind exemple simple, dacă sunt disponibile în materiale.
3. Dacă întrebarea nu are răspuns în material, spune: „Această informație nu este disponibilă în materialele de curs.”
4. Nu oferi opinii personale sau informații speculative.
Acum, aștept întrebările studenților.
"""
    # model = ="gpt-3.5-turbo-1106"
    def __init__(self, retriever, api_key, model, temperature=0.3, include_history = False):
        self.logger = setup_logger(__name__)
        self.retriever = retriever
        self.openai_key = api_key
        self.include_history = include_history
        self.chat = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=api_key)
        # self.chat = ChatOpenAI(model=model, temperature=temperature, openai_api_key=openai_key)
        self.question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.SYSTEM_TEMPLATE
                ),
                MessagesPlaceholder(variable_name="messages")
            ]
        )
        self.document_chain = create_stuff_documents_chain(self.chat, self.question_answering_prompt)
        self.init_query_creator()
        if include_history:
            self.conversational_retrieval_chain = RunnablePassthrough.assign(
                answer=self.document_chain,
            )
        else:
            def parse_retreiver_input(params):
                return params["messages"][-1].content
            self.conversational_retrieval_chain = RunnablePassthrough.assign(
                context=parse_retreiver_input | retriever
            ).assign(answer=self.document_chain)
        self.history = []
        self.full_history = []

    def init_query_creator(self):
        """
        Pe baza query-ului primit de la utilizator si a istoricului, creaza query-ul actual
        """
        self.query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="messages"),
                (
                    "user",
                    "Dandu-se conversatia de mai sus, genereaza un query de cautare astfel incat sa obti informatia relevanta pentru conversatie. Raspunde doar cu query-ul, nimic altceva.",
                ),
            ]
        )
        self.query_transformation_chain = self.query_transform_prompt | self.chat

        self.query_transforming_retriever_chain = RunnableBranch((
                lambda x : len(x.get("messages", [])) == 1, # avem doar un mesaj
                RunnableLambda(lambda x: {
                    "parsedQuery": x["messages"][-1].content,
                    "context": self.retriever.invoke(x["messages"][-1].content)
                }),
            ),
            self.query_transform_prompt 
            | self.chat 
            | StrOutputParser() 
            | RunnableLambda(lambda parsed_query : {
                "parsedQuery": parsed_query,
                "context": self.retriever.invoke(parsed_query)
            })
        ).with_config(run_name="chat_retriever_chain")
        

    def invoke_query_creator(self, history):
        """
        History is a list of messages
        """
        res = self.query_transformation_chain.invoke(
            {
                "messages": [
                    HumanMessage(content="Cati acizi grasi ar trebui sa mananc?"),
                    AIMessage(
                        content = "Acizi grași saturați ar trebui să reprezinte mai puțin de 10% din aportul total de energie conform ghidului de bune practici."
                    ),
                    HumanMessage(content="Dar pentru copii?")
                ]
            }
        )
        return res

    def ask(self, query):
        # self.__init__(self.retriever, self.openai_key)
        self.logger.info(f"Asking question: {query}")
        if not self.include_history:
            self.history = []
            self.full_history = []
        self.history.append(HumanMessage(content=query))
        context_with_processed_query = self.query_transforming_retriever_chain.invoke({"messages" : [*self.history]})
        self.logger.info(f"As context {len(context_with_processed_query)} documents: {context_with_processed_query['context'].__repr__()[:200]}")
        result = self.conversational_retrieval_chain.invoke(
            {
                "messages": [
                    *self.history
                ],
                "context": context_with_processed_query['context']
            }
        )
        result = {
            **result,
            "processed_query": context_with_processed_query['parsedQuery'],
            "context": context_with_processed_query['context']
        }
        self.history.append(AIMessage(content=result.get("answer")))
        if len(result.get("context")) > 0 and result.get("answer") != "Această informație nu este disponibilă în materialele de curs.": # We had context for this
            self.full_history.append(query)
            self.full_history.append(result)
        elif len(self.history) > 0:
            self.history.pop() # Remove irrrelevant answer from context
            self.history.pop() # Remove irrelevant question from context
        return result.get("answer"), result

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []
        pass
