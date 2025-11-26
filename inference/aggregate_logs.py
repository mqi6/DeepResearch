import os
import re
import glob

def aggregate_logs():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print(f"Directory '{log_dir}' does not exist.")
        return

    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    total_context_tokens = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_search_calls = 0
    total_visit_calls = 0
    
    files_processed = 0
    files_with_stats = 0

    print(f"Found {len(log_files)} log files.")

    for log_file in log_files:
        files_processed += 1
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            
            # Extract Final Context Tokens
            # Look for the last occurrence of FINAL | context_tokens=...
            context_matches = re.findall(r"FINAL \| context_tokens=(\d+)", content)
            if context_matches:
                total_context_tokens += int(context_matches[-1])
            
            # Extract Stats from FINAL_STATS
            # FINAL_STATS | orch_input_tokens=... | orch_output_tokens=... | search_calls=... | visit_calls=...
            stats_match = re.search(r"FINAL_STATS \| (.*)", content)
            if stats_match:
                files_with_stats += 1
                stats_str = stats_match.group(1)
                
                # Parse key-value pairs
                pairs = stats_str.split(" | ")
                stats = {}
                for pair in pairs:
                    if "=" in pair:
                        k, v = pair.split("=")
                        stats[k.strip()] = int(v)
                
                total_input_tokens += stats.get("orch_input_tokens", 0)
                total_output_tokens += stats.get("orch_output_tokens", 0)
                total_search_calls += stats.get("search_calls", 0)
                total_visit_calls += stats.get("visit_calls", 0)
            else:
                # Fallback for older logs (partial data)
                # We can sum input/output tokens from ROUND lines if FINAL_STATS is missing
                # But we can't get tool calls accurately without parsing the full content which isn't in the log
                # So we'll just sum tokens here if needed, or skip to keep it clean.
                # Let's try to sum tokens from ROUND lines for completeness on tokens
                input_matches = re.findall(r"input_tokens=(\d+)", content)
                output_matches = re.findall(r"output_tokens=(\d+)", content)
                
                # Note: FINAL_STATS includes these, so only sum if stats_match is None to avoid double counting
                # However, the regex above matches the ROUND lines too.
                # Actually, let's stick to FINAL_STATS for consistency if possible.
                # If the user wants totals across ALL logs, we should try to get tokens at least.
                
                # Let's do a more robust approach:
                # If FINAL_STATS exists, use it.
                # If NOT, sum from ROUND lines.
                pass
                
                # Re-calculating for missing stats files
                current_input = sum(int(x) for x in re.findall(r"ROUND \d+ \| input_tokens=(\d+)", content))
                current_output = sum(int(x) for x in re.findall(r"ROUND \d+ .* output_tokens=(\d+)", content))
                
                total_input_tokens += current_input
                total_output_tokens += current_output

    print("-" * 40)
    print(f"Files Processed: {files_processed}")
    print(f"Files with FINAL_STATS: {files_with_stats}")
    print("-" * 40)
    print(f"Total Context Tokens (Final): {total_context_tokens}")
    print(f"Total Input Tokens:           {total_input_tokens}")
    print(f"Total Output Tokens:          {total_output_tokens}")
    print(f"Total Search Calls:           {total_search_calls}")
    print(f"Total Visit Calls:            {total_visit_calls}")
    print("-" * 40)

if __name__ == "__main__":
    aggregate_logs()
