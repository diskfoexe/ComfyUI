import time

class ExecutionTimerStart:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"any_input": ("*", {})}} # מקבל כל דבר כדי להשתלב בשרשרת
    
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
    
    RETURN_TYPES = ("STRING", "*")
    RETURN_NAMES = ("elapsed_time_str", "passthrough")
    FUNCTION = "end_timer"
    CATEGORY = "Diskfo Nodes/Utils"

    def end_timer(self, start_time, any_input):
        end_time = time.time()
        duration = end_time - start_time
        
        # פורמט יפה למחרוזת
        result = f"Execution time: {duration:.2f} seconds"
        print(f"✨ {result}") # זה ידפיס גם לטרמינל השחור
        
        return (result, any_input)

# מיפוי עבור הקובץ הזה
NODE_CLASS_MAPPINGS = {
    "ExecutionTimerStart": ExecutionTimerStart,
    "ExecutionTimerEnd": ExecutionTimerEnd
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExecutionTimerStart": "⏱️ Timer Start",
    "ExecutionTimerEnd": "⏱️ Timer End"
}
