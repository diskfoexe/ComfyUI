import time

class ExecutionTimerStart:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any_input": ("*", {})}}
    
    RETURN_TYPES = ("FLOAT", "*")
    RETURN_NAMES = ("start_time", "passthrough")
    FUNCTION = "start_timer"
    CATEGORY = "Diskfo Nodes/Utils"

    def start_timer(self, any_input):
        return (time.time(), any_input)

class ExecutionTimerEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_time": ("FLOAT", {"default": 0}),
                "any_input": ("*", {}),
            }
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "*")
    RETURN_NAMES = ("elapsed_time_str", "passthrough")
    FUNCTION = "end_timer"
    CATEGORY = "Diskfo Nodes/Utils"

    def end_timer(self, start_time, any_input):
        end_time = time.time()
        duration = end_time - start_time
        result = f"{duration:.2f}s"
        
        # אנחנו מחזירים מילון ui, ה-JS שלנו יקשיב לו
        return {"ui": {"text": [result]}, "result": (result, any_input)}

NODE_CLASS_MAPPINGS = {
    "ExecutionTimerStart": ExecutionTimerStart,
    "ExecutionTimerEnd": ExecutionTimerEnd
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExecutionTimerStart": "⏱️ Timer Start",
    "ExecutionTimerEnd": "⏱️ Timer End"
}
