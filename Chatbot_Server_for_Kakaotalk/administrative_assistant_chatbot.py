from common import client, model
import openai
import json
from pprint import pprint
import time
import datetime
from retry import retry
from typing import Tuple
import re

instructions = """
당신은 고려대학교 학생들의 질문에 대답하는 행정 조교입니다.
학생들이 하는 질문에 대답하기 위해서는 파일들을 읽어보고
스스로 생각하여 대답하시면 됩니다.
"""

tools = [
    {"type": "file_search"}
]


class Chatbot:
    
    def __init__(self, **args):
        self.assistant = client.beta.assistants.retrieve(assistant_id = args.get("assistant_id"))
        self.thread = client.beta.threads.retrieve(thread_id=args.get("thread_id"))
        self.runs = list(client.beta.threads.runs.list(thread_id=args.get("thread_id")))

    @retry(tries=3, delay=2)
    def add_user_message(self, user_message):
        try:
            client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role="user",
                content=user_message,
            )
        except openai.BadRequestError as e:
            if len(self.runs) > 0:
                print("add_user_message BadRequestError", e)
                client.beta.threads.runs.cancel(thread_id=self.thread.id, run_id=self.runs[0])
            raise e
        
    def _run_action(self, retrieved_run):
        tool_calls = retrieved_run.model_dump()['required_action']['submit_tool_outputs']['tool_calls']
        pprint(("tool_calls", tool_calls))
        tool_outputs=[]
        for tool_call in tool_calls:
            pprint(("tool_call", tool_call))
            id = tool_call["id"]
            function = tool_call["function"]
            func_name = function["name"]
            # 챗GPT가 알려준 함수명에 대응하는 실제 함수를 func_to_call에 담는다.
            func_to_call = self.available_functions[func_name]
            try:
                func_args = json.loads(function["arguments"])
                # 챗GPT가 알려주는 매개변수명과 값을 입력값으로하여 실제 함수를 호출한다.
                print("func_args:",func_args)
                func_response = func_to_call(**func_args)
                tool_outputs.append({
                    "tool_call_id": id,
                    "output": str(func_response)
                })
            except Exception as e:
                    print("_run_action error occurred:",e)
                    client.beta.threads.runs.cancel(thread_id=self.thread.id, run_id=retrieved_run.id)
                    raise e
                    
        client.beta.threads.runs.submit_tool_outputs(
            thread_id = self.thread.id, 
            run_id = retrieved_run.id, 
            tool_outputs= tool_outputs
        )    
    
    @retry(tries=3, delay=2)    
    def create_run(self):
        try:
            run = client.beta.threads.runs.create(
                thread_id=self.thread.id,
                assistant_id=self.assistant.id,
            )
            self.runs.append(run.id)
            return run
        except openai.BadRequestError as e:
            if len(self.runs) > 0:
                print("create_run BadRequestError", e)
                client.beta.threads.runs.cancel(thread_id=self.thread.id, run_id=self.runs[0])
            raise e            
        
    def get_response_content(self, run) -> Tuple[openai.types.beta.threads.run.Run, str]:

        max_polling_time = 60 # 최대 1분 동안 폴링합니다.
        start_time = time.time()

        retrieved_run = run
        
        while(True):
            elapsed_time  = time.time() - start_time
            if elapsed_time  > max_polling_time:
                client.beta.threads.runs.cancel(thread_id=self.thread.id, run_id=run.id)
                return retrieved_run, "대기 시간 초과(retrieve)입니다."
            
            retrieved_run = client.beta.threads.runs.retrieve(
                thread_id=self.thread.id,
                run_id=run.id
            )            
            print(f"run status: {retrieved_run.status}, 경과:{elapsed_time: .2f}초")
            
            if retrieved_run.status == "completed":
                break
            elif retrieved_run.status == "requires_action":
                self._run_action(retrieved_run)
            elif retrieved_run.status in ["failed", "cancelled", "expired"]:
                # 실패, 취소, 만료 등 오류 상태 처리
                #raise ValueError(f"run status: {retrieved_run.status}, {retrieved_run.last_error}")
                code = retrieved_run.last_error.code
                message = retrieved_run.last_error.message
                return retrieved_run, f"{code}: {message}"
            
            time.sleep(1) 
            
        # Run이 완료된 후 메시지를 가져옵니다.
        self.messages = client.beta.threads.messages.list(
            thread_id=self.thread.id
        )
        resp_message = [m.content[0].text for m in self.messages if m.run_id == run.id][0]
        
        # 출처 정보를 제거하기 위한 정규 표현식
        resp_message_cleaned = re.sub(r"【\d+:\d+†.*?】", "", resp_message.value)
        
        return retrieved_run, resp_message_cleaned
        
    def get_interpreted_code(self, run_id):
        run_steps_dict = client.beta.threads.runs.steps.list(
            thread_id=self.thread.id,
            run_id=run_id
        ).model_dump()
        for run_step_data in run_steps_dict['data']:
            step_details = run_step_data['step_details']
            print("step_details", step_details)
            tool_calls = step_details.get('tool_calls', [])
            for tool_call in tool_calls:
                if tool_call['type'] == 'code_interpreter':
                    return tool_call['code_interpreter']['input']
        return ""


if __name__ == "__main__":

    vector_store = client.beta.vector_stores.create(
        name="academic_store"
    )

    file_paths = ["files/academic_regulation.pdf", "files/cultural_sports.pdf", "files/global_business.pdf", "files/pharmacy.pdf", "files/public_policy.pdf", "files/science.pdf", "files/smart_city.pdf", "files/total_fire_scholoarship.pdf"]
    file_streams = [open(path, "rb") for path in file_paths]

    # Use the upload and poll SDK helper to upload the files, add them to the vector store,
    # and poll the status of the file batch for completion.
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
      vector_store_id = vector_store.id, files=file_streams,
    )
    
    assistant = client.beta.assistants.create(
                    model=model.advanced,  
                    name="OCB",
                    instructions=instructions,
                    tools=tools,
                    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
                )
    thread = client.beta.threads.create()   
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 출력할 내용
    assistants_ids = f"{assistant.id}, {thread.id}, {file_batch.id}"
    print(assistants_ids)   
    # 파일에 기록 (파일명은 예시로 'output_log.txt'를 사용)
    with open("./files/assistants_ids.txt", "a") as file:
        file.write(f"{current_time} - {assistants_ids}\n")
    
    
     