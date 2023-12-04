from llama_cpp import Llama

class Alpaca:
    def __init__(self):
        self.llm = Llama(model_path="Models/claude2-alpaca-7b.Q8_0.gguf",
                    verbose=False,
                    )
        
    def respond(self,Query):
        '''User writes a Query (A sentence for completion)'''
        return self.llm(Query,max_tokens=100)['choices'][0]['text']

if __name__ == "__main__":
    model=Alpaca()
    model.respond("Name 5 vegetables")