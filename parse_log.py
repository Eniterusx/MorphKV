import re

def parse_log():
    try:
        with open("output.log", "r", encoding="utf-16", errors="ignore") as f:
            lines = f.readlines()
            
        last_stats = None
        for line in lines:
            if "KV Cache size" in line:
                last_stats = line.strip()
                
        if last_stats:
            print("Found stats:")
            print(last_stats)
        else:
            print("No stats found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parse_log()
