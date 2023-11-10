import openai
import time
import os
import yfinance as yf

#======== API KEY=================
os.environ["OPENAI_API_KEY"] = "put you key here"


#======== Create a client ========
client = openai.OpenAI()


#======== Knowledge base ==========
file = client.files.create(
    file=open("./a_beginners_guide_to_the_stock_market.pdf", "rb"),
    purpose='assistants'
)


#======== Function ================
def stock_price_func(symbol: str) -> float:
    stock = yf.Ticker(symbol)
    price = stock.history(period="1d")['Close'].iloc[-1]
    return price


#======= List of tools ============
tools_list = [{
    "type": "function",
    "function": {

        "name": "stock_price_func",
        "description": "Retrieve the latest closing price of a stock using its ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The ticker symbol of the stock"
                }
            },
            "required": ["symbol"]
        }
    }
}]


#======= Step 1: Create an Assistant===
assistant = client.beta.assistants.create(
    name="Stock Analyst Assistant",
    instructions="You are a personal Stock Analyst Assistant",
    model="gpt-4-1106-preview",
    tools=tools_list,
    file_ids=[file.id]
)


#===== Step 2:Create a Thread ========
thread = client.beta.threads.create()


#===== Step 3:Add Message to a Thread ========
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="What can you invest in the stock market?"
)


#===== Step 4: Run the Assistant==============
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
    instructions="Please address the user as LLM Facile"
)

#===== Step 5: Display the messages ===========


while True:
    # Wait for 5 seconds
    time.sleep(5)

    # get the run status
    run_status = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    print(run_status.model_dump_json(indent=4))

    # If run is completed, get messages
    if run_status.status == 'completed':
        messages = client.beta.threads.messages.list(
            thread_id=thread.id
        )

        # print content based on role
        for msg in messages.data:
            role = msg.role
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")

        break
    elif run_status.status == 'requires_action':
        print("Function Calling")
        required_actions = run_status.required_action.submit_tool_outputs.model_dump()
        print(required_actions)
        tool_outputs = []
        import json
        for action in required_actions["tool_calls"]:
            func_name = action['function']['name']
            arguments = json.loads(action['function']['arguments'])
            
            if func_name == "stock_price_func":
                output = stock_price_func(symbol=arguments['symbol'])
                tool_outputs.append({
                    "tool_call_id": action['id'],
                    "output": output
                })
            else:
                raise ValueError(f"Unknown function: {func_name}")
            
        print("Submitting outputs back to the Assistant...")
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )
    else:
        print("Waiting for the Assistant to process...")
        time.sleep(5)





