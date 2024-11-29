from qdrant_client import QdrantClient
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

import warnings
warnings.filterwarnings("ignore")

class AIVoiceAssistant:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)
        self._llm = Ollama(model="llama3.2:1b", request_timeout=120.0)
        self._service_context = ServiceContext.from_defaults(llm=self._llm, embed_model="local")
        self._index = None
        self._chat_engine = None

        self._create_kb()
        self._create_chat_engine()

    def _create_chat_engine(self):
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )
    
    
    def _create_kb(self):
        try:
                    # Check if "Botmer_db" collection already exists
            # collections = self._client.get_collections().collections
            # if any(collection.name == "Botmer_db" for collection in collections):
            #     print("Knowledgebase already exists. Skipping creation.")
            
            reader = SimpleDirectoryReader(
                input_files=[r"rag\botmer_file.txt"]
            )
            documents = reader.load_data()
            vector_store = QdrantVectorStore(client=self._client, collection_name="Botmer_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents, service_context=self._service_context, storage_context=storage_context
            )
            print("Knowledgebase created successfully!")
        except Exception as e:
            print(f"Error while creating knowledgebase: {e}")
    




    def interact_with_llm(self, customer_query):
        AgentChatResponse = self._chat_engine.chat(customer_query)
        answer = AgentChatResponse.response
        return answer

    @property
    def _prompt(self):
        return """
                You are a professional AI Assistant "Emily" receptionist working at Botmer, an advanced customer service platform offering seamless, 24/7 support. Your role is to handle customer inquiries with politeness and efficiency.
                You should always start first with greetings.You should ask the questions mentioned inside square brackets, but always ask one question at a time. Keep the conversation engaging and polite.
                [Ask the customer's Name, Email, and contact number. Then ask what they want the service they need. End the conversation with a polite greeting.]
                you have a complete knowledge of company [Botmer International] there services contact information ,email,phone number etc, but If you don’t know the answer, simply say that you don’t know and don`t use your own knowledge.
                Be concise and provide short answers (no more than 10 words). Make sure to personalize the interaction by addressing the customer by their name when appropriate.
            """


