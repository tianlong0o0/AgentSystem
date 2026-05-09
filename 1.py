image_base64 = 1






# 1.大模型用于判断的输入输出示例
input = [{"role": "system",
          "content": "你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否有被困人员。" },
          {"role": "user",
           "content": [{"type": "text",
                        "text": "参考YOLO模型的识别结果判断附近是否可能有被困人员，请回答'有'或'没有'。"},
                       {"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}]
output = [{"role": " assistant","content": "有" }]
# 2.大模型用于决策的输入输出示例
input = [{"role": "system",
          "content": "你是一个执行搜救任务的人工智能助手，请根据信息判断附近是否有被困人员，"
          "如果有，请从操作库中选择需要执行的操作(每次只可选择1种操作)。" },
          {"role": "user",
           "content": [{"type": "text", "text": "请选择需要执行的操作。"
                        "(可执行操作包含:'移动至被困人员处','在被困人员物品周围搜寻被困人员',"
                        "'通知总部找到被困人员','向被困人员发放紧急救援物资','安抚被困人员',"
                        "'继续寻找其他被困人员')"},
                       {"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}]
output = [{"role": " assistant","content": "移动至被困人员处" }]

