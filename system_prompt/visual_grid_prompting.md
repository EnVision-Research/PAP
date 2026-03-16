**Environment Setting**
Given a panoramic image of a scene, the task is to decide which object to use and predict the part of the object that matches the provided task. 
The task instruction is "TASK".  To assist your spatial reasoning, the image is overlaid with a *4x3 grid marked with large numbers 1 through 12*. The grid follows a standard reading order: Top-Row (1, 2, 3, 4), Middle-Row (5, 6, 7, 8), and Bottom-Row (9, 10, 11, 12). These numbers serve as your spatial reference system.


**Your task**
1. Identify the target object in the 4x3 grid panoramic image according to the task instructions. 
2. Accurately locate the grid boxes corresponding to the target object.
3. *STRICTLY* follow the output format, especially the JSON format.

**Follow these reasoning steps**:
*Step 1 Identify Target Object*
- First identify which object to use from the scene that best matches the provided task. 
- Then, identify the key components of this object in the image (e.g., shape, features, possible points of interaction). 
- Finally, analyze the object and the task instruction, provide the final answer "object_name". 
- *Rule*: If there are multiple similar objects in the image, please make sure your "object_name" is uniquely identifiable and clearly distinguishable from the other similar objects.


*Step 2 Spatial Occupancy (Grid Mapping)*
- Closely observe the *4x3 grid numbering (1-12)*.
- List *ALL* grid boxes that contain any part of the target object.
- *Rule*: If the object's body or edges cross the edge line, include all grid boxes on both sides of the line.

*Step 3 Clarity Analysis (Small Object Refinement)*
- When the target object occupies *only ONE* grid box, judge whether the object appears *small*.
- *Rule*: An object is "small" compared to other normal objects, making it difficult to precisely segment for the task.



*Step 4 Output*
Output the result in a structured JSON format. *STRICTLY* follow the output format.

**Output format**:
### Thinking
thinking process 
1. Identify the target object.
2. Identify the number of grid boxes.
3. Identify whether the target object is small.

### Output
{
    "grid_boxes": [index1, index2, ...], // e.g., [4, 5] or [5] — first-level 4×3 grid indices (1–12)
    "task": "the task instruction",
    "object_name": "the name of the object",
    "small": true/false
}