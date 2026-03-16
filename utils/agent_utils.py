import json
import re
import os

def extract_json(text_content, output_path=None):
    """
    Extracts a JSON object from a string and saves it to a file.
    The JSON is expected to be after '### Output'.
    Saves both the raw text_content and parsed JSON in the same file.
    """
    try:
        # Find the content after '### Output'
        output_section = text_content.split("### Output\n")[-1]
        
        # A more robust way to find the JSON object, even with surrounding text
        match = re.search(r'\{.*\}', output_section, re.DOTALL)
        if not match:
            print("Error: No JSON object found in the output section.")
            return

        json_string = match.group(0)

        # Parse the JSON string
        data = json.loads(json_string)
        
        # Add raw text to the parsed data as a sibling field
        data["raw_response"] = text_content
        
        # Save the JSON object to a file
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Successfully extracted and saved JSON to {output_path}")
        
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the text.")
    except IndexError:
        print("Error: '### Output' section not found in the text.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return data

def post_process(response, images, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    extracted_data = extract_json(response, f"{output_dir}/grounding_agent_output.json")
    object_part = extracted_data["object_part"]
    bboxes = [extracted_data["part_bbox"]]
    points = extracted_data["key_points"]
    width, height = images[0].size
    if isinstance(points[0][0], float):
        points = [[int(point[0] * width), int(point[1] * height)] for point in points]
    if isinstance(bboxes[0][0], float):
        bboxes = [[int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)] for bbox in bboxes]

    return bboxes, points, object_part