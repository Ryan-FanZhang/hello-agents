import os 
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from serp_search import search
from tools import ToolExecutor
import re

load_dotenv()

REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 finish(answer="...") 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

class HelloAgentsLLM:
    """
    Design LLM class to enable it to call any openai-based services and 
    use streaming services as default
    """

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("Model, API key or Base Url must be provided either as arguments or environment variables.")
        
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        Method to call OpenAI chat completion with streaming support
        """
        print(f"🧠 正在调用 {self.model} 模型...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True
            )

            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end='', flush=True)
                collected_content.append(content)
            print()  # for newline after streaming
            return ''.join(collected_content)
        
        except Exception as e:
            print("❌ 大语言模型响应失败:", str(e))
            return None 
        
class ReActAgent:
    def __init__(self, llmClient: HelloAgentsLLM, toolExecutor: ToolExecutor, maxSteps: int = 5):
        self.llmClient = llmClient
        self.toolExecutor = toolExecutor
        self.maxSteps = maxSteps
        self.history = []
    
    def _parse_output(self, text: str):
        """
        简单解析 LLM 输出，提取 Thought 和 Action
        """
        thought_match = re.search(r"Thought: (.*)", text)
        action_match = re.search(r"Action: (.*)", text)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。"""
        match = re.match(r"(\w+)\[(.*)\]", action_text)
        if match:
            return match.group(1), match.group(2)
        return None, None
            

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.maxSteps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # format prompt
            tool_desc = self.toolExecutor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tool_desc,
                question=question,
                history=history_str
            )

            messages = [
                {"role": "user", "content": prompt}
            ]

            response_text = self.llmClient.think(messages=messages)

            if not response_text:
                print("LLM 未返回任何响应，终止执行。")
                break
            
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break
            
            # 解析并执行 Action
            if action.startswith("Finish"):
                # if the action is Finish, extract the final answer
                final_ansnwer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"最终答案: {final_ansnwer}")
                return final_ansnwer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print("警告:未能解析出有效的工具调用，流程终止。")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_function = self.toolExecutor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            
            else:
                observation = tool_function(tool_input)

                print(f"🔎 观察结果: {observation}")

                self.history.append(f"Action: {action}")
                self.history.append(f"Observation: {observation}")
            
            # 如果达到最大步骤数，终止执行
        print("已达到最大步骤数，终止执行。")
        return None


# # --- 客户端使用示例 ---
# if __name__ == '__main__':
#     try:
#         llmClient = HelloAgentsLLM()
        
#         exampleMessages = [
#             {"role": "system", "content": "You are a helpful assistant that writes Python code."},
#             {"role": "user", "content": "写一个快速排序算法"}
#         ]
        
#         print("--- 调用LLM ---")
#         responseText = llmClient.think(exampleMessages)
#         if responseText:
#             print("\n\n--- 完整模型响应 ---")
#             print(responseText)

#     except ValueError as e:
#         print(e)

# --- 工具初始化与使用示例 ---
# if __name__ == '__main__':
#     toolExecutor = ToolExecutor()
    
#     # 注册搜索工具
#     search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
#     toolExecutor.registerTool(
#         name="serp_search",
#         description=search_description,
#         func=search
#     )
    
#     # 展示可用工具
#     print("\n--- 可用工具列表 ---")
#     print(toolExecutor.getAvailableTools())
    
#     # 智能体的Action调用，这次我们问一个实时性的问题
#     print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
#     tool_name = "serp_search"
#     tool_inpuit = "Whjat is the latest GPU model from Nvidia?"

#     tool_function = toolExecutor.getTool(tool_name)
#     if tool_function:
#         observation = tool_function(tool_inpuit)
#         print(f"\n--- 工具观察结果 ---\n{observation}")
#         print(observation)

#     else:
#         print(f"工具 '{tool_name}' 未找到。")

if __name__ == "__main__":
    # 1) 初始化工具执行器
    toolExecutor = ToolExecutor()
    toolExecutor.registerTool(
        name="serp_search",
        description="一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
        func=search
    )

    # 2) 初始化 LLM
    llmClient = HelloAgentsLLM()

    # 3) 初始化 ReAct Agent
    agent = ReActAgent(llmClient=llmClient, toolExecutor=toolExecutor, maxSteps=5)

    # 4) 运行测试问题（尽量选“必须查实时”的）
    question = "华为最新手机型号及主要卖点是什么？"
    answer = agent.run(question)

    print("\n=== Agent 返回 ===")
    print(answer)