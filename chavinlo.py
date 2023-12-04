from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch
from langchain import PromptTemplate, LLMChain

class Alpaca:
    def __init__(self) -> None:
        self.tokenizer = LlamaTokenizer.from_pretrained("chavinlo/gpt4-x-alpaca")
        self.quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
        self.base_model = LlamaForCausalLM.from_pretrained(
            "chavinlo/gpt4-x-alpaca",
            load_in_8bit=True,
            device_map='auto',
            quantization_config=self.quantization_config
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.base_model, 
            tokenizer=self.tokenizer, 
            max_length=256,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.2
        )
        self.local_llm = HuggingFacePipeline(pipeline=self.pipe)

        
        self.template = """respond to the instruction below. behave like a chatbot and respond to the user. try to be helpful.
        ### Instruction: 
        {instruction}
        Answer:"""
        self.prompt = PromptTemplate(template=self.template, input_variables=["instruction"])


        self.llm_chain = LLMChain(prompt=self.prompt, 
                            llm=self.local_llm
                            )
    def respond(self,Query):
        return self.llm_chain.run(question)



if __name__ == "__main__":
    Model = Alpaca()
    question = "i want to order a pizza"
    print(Model.respond(question))
