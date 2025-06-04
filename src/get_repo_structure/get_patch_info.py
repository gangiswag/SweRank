import argparse
import json
import re
from collections import defaultdict

def split_github_patch(patch_string):
    # Regular expression to match the change indicator pattern
    pattern = r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@ "
    
    # Split the string using the pattern
    split_patches = re.split(pattern, patch_string)
    
    # Remove any empty strings from the result
    split_patches = [patch.strip() for patch in split_patches if patch.strip()]
    
    return split_patches

def parse_patch(patch):
    """
    Parse a git patch into a structured format.

    Parameters:
        patch (str): The git patch as a string.

    Returns:
        list: A list of dictionaries representing the file changes and hunks.
    """
    file_changes = []
    current_file = None
    current_hunk = None
    deleted_lines = 0

    patch_lines = patch.split("\n")
    for line in patch_lines:
        if line.startswith("diff --git"):
            # Reset for new files
            if current_file:
                file_changes.append(current_file)
            current_file = {"file": "", "hunks": []}
        elif line.startswith("--- a/"):
            pass
        elif line.startswith("+++ b/"):
            if current_file is not None:
                current_file["file"] = line[6:]
        elif line.startswith("@@ "):
            if current_file is not None:
                match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if match:
                    current_hunk = {"start_line": int(match.group(2)), "changes": []}
                    current_file["hunks"].append(current_hunk)
                    deleted_lines = 0
                    added_lines = 0
        elif line.startswith("+") or line.startswith("-"):
            if current_hunk is not None:
                change_type = "add" if line.startswith("+") else "delete"
                if change_type == "delete":
                    deleted_lines += 1
                    current_hunk["changes"].append(
                        {
                            "type": change_type,
                            "content": line[1:].strip(),
                            "line": current_hunk["start_line"] - added_lines,
                        }
                    )
                    current_hunk["start_line"] += 1
                else:
                    added_lines += 1
                    current_hunk["changes"].append(
                        {
                            "type": change_type,
                            "content": line[1:].strip(),
                            "line": current_hunk["start_line"] - deleted_lines,
                        }
                    )
                    current_hunk["start_line"] += 1
        else:
            if current_hunk is not None:
                current_hunk["start_line"] += 1

    if current_file:
        file_changes.append(current_file)

    return file_changes


def patch_to_dict(patch_string):
    patch_lines = patch_string.splitlines()
    patch_dict = {}
    
    current_file = None
    current_content = []

    for line in patch_lines:
        # Check if this line indicates the start of a new file
        if line.startswith('diff --git'):
            # If we were already processing a file, save its content
            if current_file:
                patch_dict[current_file] = process_hunks(current_content)
            # Reset for the new file
            current_file = None
            current_content = []
        
        # Identify the file name from the '+++ b/' line
        elif line.startswith('+++ b/'):
            current_file = line[6:]  # Extract file name from '+++ b/filename'
        
        # If we are processing a file, keep adding content
        elif current_file:
            current_content.append(line)
    
    # Don't forget to add the last file processed to the dictionary
    if current_file:
        patch_dict[current_file] = process_hunks(current_content)
    
    return patch_dict

def process_hunks(file_content):
    """Process the content of a file into hunks keyed by the '@@' lines."""
    hunk_dict = {}
    current_hunk_key = None
    current_hunk_content = []
    
    for line in file_content:
        if line.startswith('@@'):
            # If there is an existing hunk, save it before processing the new one
            if current_hunk_key:
                hunk_dict[current_hunk_key] = "\n".join(current_hunk_content)
            # Set the new hunk key and reset the hunk content
            current_hunk_key = line
            current_hunk_content = []
        
            # Keep adding lines to the current hunk content
        current_hunk_content.append(line)
    
    # Save the last hunk processed
    if current_hunk_key:
        hunk_dict[current_hunk_key] = "\n".join(current_hunk_content)
    
    return hunk_dict


def get_last_function_or_method(structure):
    last_function = None
    last_method = None
    last_class_name = None

    # Check the last function in the functions list
    if structure['functions']:
        last_function = max(structure['functions'], key=lambda func: func['end_line'])

    # Check the last method in each class and store the class name
    for cls in structure['classes']:
        if cls['methods']:
            last_class_method = max(cls['methods'], key=lambda method: method['end_line'])
            if not last_method or last_class_method['end_line'] > last_method['end_line']:
                last_method = last_class_method
                last_class_name = cls['name']

    # Compare the last function and last method
    if last_function and last_method:
        if last_function['end_line'] > last_method['end_line']:
            return {'function_name': last_function['name'], 'class_name': None}
        else:
            return {'function_name': last_method['name'], 'class_name': last_class_name}
    elif last_function:
        return {'function_name': last_function['name'], 'class_name': None}
    elif last_method:
        return {'function_name': last_method['name'], 'class_name': last_class_name}
    else:
        return None

def find_class_only(line_num, structure):
    for cls in structure['classes']:
        if (cls['start_line'] <= line_num <= cls['end_line']):
            return cls 
    return None      
def find_class_or_function(line_num, structure):
    for cls in structure['classes']:
        if (cls['start_line'] <= line_num <= cls['end_line']):
            for method in cls['methods']:
                if (method['start_line'] <= line_num <= method['end_line']):
                    return cls, method
            return cls, None  # No method, but inside the class
    for func in structure['functions']:
        if (func['start_line'] <= line_num <= func['end_line']):
            return None, func
    return None, None



# Helper function to find the class or function for a line
# def find_class_or_function(line_num, structure):
#     def find_precise_function(line_num, functions):
#         """Recursively find the most precise nested function at the given line number."""
#         for func in functions:
#             if func['start_line'] <= line_num <= func['end_line']:
#                 # Check if there are nested functions inside this function
#                 precise_nested = find_precise_function(line_num, func.get('nested_functions', []))
#                 return func if precise_nested is None else precise_nested
#         return None

#     # Check if the line number falls within a class
#     for cls in structure['classes']:
#         if cls['start_line'] <= line_num <= cls['end_line']:
#             # Check if it falls within a method inside the class
#             for method in cls['methods']:
#                 if method['start_line'] <= line_num <= method['end_line']:
#                     # Check for the most precise nested function within this method
#                     precise_function = find_precise_function(line_num, [method])
#                     return cls, precise_function
#             return cls, None  # No method, but inside the class

#     # Check if the line number falls within a top-level function
#     for func in structure['functions']:
#         if func['start_line'] <= line_num <= func['end_line']:
#             # Check for the most precise nested function within this top-level function
#             precise_function = find_precise_function(line_num, [func])
#             return None, precise_function

#     return None, None  # Not inside any class or function

def create_hunk_result(class_changed, function_changed, num_lines_changed, num_lines_added,
                       num_lines_removed, newly_added):
    result =  {
            'class_changed': class_changed,
            'function_changed': function_changed,
            'num_lines_changed': num_lines_changed,
            'num_lines_added': num_lines_added,
            'num_lines_removed': num_lines_removed,
            'newly_added': newly_added
        }
    return result



def analyze_hunks(hunks, structure):
    results = []
    # Iterate through each hunk and analyze changes
    for hunk_header, hunk_content in hunks.items():
        # Parse hunk header to get line ranges (e.g., @@ -101,10 +101,11 @@)
        hunk_lines = hunk_content.splitlines()
        old_line_num, new_line_num = extract_line_numbers(hunk_header)
        
        class_changed = None
        function_changed = None

        current_old_line = old_line_num
        # Process each line in the hunk content
        i = 1
        while i < len(hunk_lines):
            if hunk_lines[i].startswith('+') and not hunk_lines[i].startswith('+++'):
                if 'def ' in hunk_lines[i]:
                    while i < len(hunk_lines) and hunk_lines[i].startswith('+') and not hunk_lines[i].startswith('+++'):
                        i += 1

                elif 'class ' in hunk_lines[i]:
                    while i < len(hunk_lines) and hunk_lines[i].startswith('+') and not hunk_lines[i].startswith('+++'):
                        if 'def ' in hunk_lines[i]:
                            while i < len(hunk_lines) and hunk_lines[i].startswith('+') and not hunk_lines[i].startswith('+++'):
                                i += 1
                        else:
                            i += 1
                
                else:
                    # Find the associated class or function for the added line
                    cls, func = find_class_or_function(current_old_line, structure)
                    if cls:
                        class_changed = cls['name']
                    if func:
                        function_changed = func['name']
                    
                    result = {'class_changed': class_changed, 'function_changed': function_changed, 'newly_added': True}
                    class_changed = None
                    function_changed = None

                    results.append(result)
                    i += 1 
            
            elif hunk_lines[i].startswith('-') and not hunk_lines[i].startswith('---'):
                # This line was removed
                cls, func = find_class_or_function(current_old_line, structure)
                if cls:
                    class_changed = cls['name']
                if func:
                    function_changed = func['name']
                result = {'class_changed': class_changed, 'function_changed': function_changed, 'newly_added': False}
                class_changed = None
                function_changed = None
                results.append(result)
                current_old_line += 1
                i += 1
            else:
                current_old_line += 1
                i += 1


        # Create the result for this hunk
        class_changed = None
        function_changed = None

    return results

def extract_line_numbers(hunk_header):
    # Example hunk header format: @@ -101,10 +101,11 @@
    parts = hunk_header.split()
    old_range = parts[1][1:].split(',')  # "-101,10"
    new_range = parts[2][1:].split(',')  # "+101,11"

    old_line_num = int(old_range[0])
    new_line_num = int(new_range[0])

    return old_line_num, new_line_num


def extract_structure(path_str, structure):
    # Split the path by '/'
    keys = path_str.split('/')
    
    # Initialize structure if it's not provided
    if structure is None:
        structure = {'structure': {}}
    
    # Start with the base dictionary (the 'structure' key)
    current_level = structure['structure']
    
    # Iterate through each part of the path
    for key in keys:
        current_level = current_level[key]
    
    return current_level


def parse_patch_full(patch, repo_structure):
    """
    Parse a git patch into a structured format.

    Parameters:
        patch (str): The git patch as a string.

    Returns:
        list: A list of dictionaries representing the file changes and hunks.
    """
    #TODO: get info about which functions/classes added, removed, renamed
    hunks_per_file_dct = patch_to_dict(patch)
    full_patch_info = {}
    for file, hunks in hunks_per_file_dct.items():
        if not file.endswith('.py'):
            continue
        try:
            file_structure = extract_structure(file, repo_structure)
        except:
            continue
        full_patch_info[file] = analyze_hunks(hunks, file_structure)

    return full_patch_info

def check(lines,i,  indentation):
    if lines[i].startswith('- ') or lines[i].startswith('+ '):
        return indentation < len(lines[i].strip('+').strip('-')) - len(lines[i].strip('+').strip('-').lstrip())
    elif lines[i].startswith('-') or lines[i].startswith('+'):
        return indentation <= len(lines[i].strip('+').strip('-')) - len(lines[i].strip('+').strip('-').lstrip())
    
    return True 

def extract_changed_functions(patch_string, type = 'function'):
    # Regular expression to match function definitions
    if type == 'function':
        function_pattern = r'def\s+(\w+)\s*\('
    else:
        class_pattern = r'class\s+(\w+)\s*\('
    # Split the patch into lines
    lines = patch_string.split('\n')

    changed_functions = set()
    i = 0
    while i < len(lines):
        match = re.search(function_pattern, lines[i].strip('+').strip('-'))
        
        if match and lines[i].startswith('-'):
            changed_functions.add(match.group(1))
            i += 1
            continue

        elif match and not lines[i].startswith('+'):
          indentation = len(lines[i]) - len(lines[i].lstrip())
          i += 1
          while i < len(lines) and not re.search(function_pattern, lines[i].strip('+').strip('-')) and check(lines,i,  indentation):
            if (lines[i].startswith('-') or lines[i].startswith('+')):
                changed_functions.add(match.group(1))
            i += 1

        else:
          i += 1
    
    return changed_functions


def find_py_or_non_dict_with_path(d, cond = False):
    results = {}
    stack = [(d, [])] 

    while stack:
        current_dict, path = stack.pop()  

        for key, value in current_dict.items():
            current_path = path + [key]  

            if key.endswith('.py'):
                if ('test' not in key.lower() and (('test' not in '/'.join(current_path).lower()) or cond)):
                    file_path = '/'.join(current_path)
                    
                    for fun in value['functions']:
                        func_path = f"{file_path}/{fun['name']}"
                        results[func_path] = f'{func_path}\n' + '\n'.join(fun['text'])
                    
                    for clas in value['classes']:
                        for fun in clas['methods']:
                            func_path = f"{file_path}/{clas['name']}/{fun['name']}"
                            results[func_path] = f'{func_path}\n' + f'class {clas["name"]}:' + '\n'.join(fun['text'])
                
            elif isinstance(value, dict):
                stack.append((value, current_path)) 
    
    return results




def search_errored_funcs(d, file_name, class_name, func_name, line_number):
    errored_function = func_name if func_name else None
    errored_class = class_name if class_name else None
    
    stack = [(d, [])] 

    while stack and not (errored_function and errored_class):
        current_dict, path = stack.pop()  

        for key, value in current_dict.items():
            current_path = path + [key]
              

            if key.endswith('.py') and 'test' not in key.lower() and 'test' not in '/'.join(current_path).lower() and (key in file_name if file_name != '' else True):
                file_path = '/'.join(current_path)
                
                for fun in value['functions']:
                    if line_number != '' and fun['start_line'] <= line_number <= fun['end_line']:
                        errored_function =  f"{file_path}/{fun['name']}"
                    
                    if errored_function:
                        break
                    
                    
                
                for clas in value['classes']:
                    if line_number != '' and fun['start_line'] <= line_number <= fun['end_line']:
                         errored_class = f"{file_path}/{clas['name']}"
                    
                    for fun in clas['methods']:
                        if line_number != '' and fun['start_line'] <= line_number <= fun['end_line']:
                            errored_function = f"{file_path}/{clas['name']}/{fun['name']}"
                        
                        if errored_function:
                            break 
                    
                    if errored_class:
                        break
                            
            
            elif isinstance(value, dict):
                stack.append((value, current_path)) 
    
    return errored_function, errored_class