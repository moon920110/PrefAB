# llm/prompt/task_prompt.py

TASK_INSTRUCTION_LABELED = """
# Comparative Video Analysis for Emotional Appraisals

You are an expert game analyst specializing in emotional appraisal theory. 
You are provided with two 3-second video clips from a game session: **Video A** and **Video B**.
You are also provided with the **Context** of the specific game being played (Rules, Controls, Objects).

Based on a previous evaluation, the following appraisals were determined for these clips. 
The label (A or B) indicates which clip more strongly represents that appraisal.

## Appraisal Data
- **Novelty**: {novelty} (New, unexpected, or sudden events)
- **Goal Relevance**: {goal_relevance} (Importance to current goals)
- **Outcome Probability**: {outcome_probability} (Likelihood of consequences)
- **Goal Conduciveness**: {goal_conduciveness} (Helps or hinders goals)
- **Urgency**: {urgency} (Need for immediate action)
- **Coping Potential**: {coping_potential} (Ability to handle the event)

## Task
Analyze both video segments utilizing the **Game Context** provided above.
Explain **WHY** these specific appraisals (A or B) were assigned. 

For each of the 6 appraisal dimensions, provide:
1. **Rationale**: A direct comparison referencing specific visual cues, game mechanics, and player actions (e.g., "In Video A, the player crashed into a wall (Novelty), whereas Video B showed routine driving").
2. **Description**: A general summary of the event related to that dimension.
""".strip()

TASK_INSTRUCTION_UNLABELED = """
# Comparative Video Analysis for Emotional Appraisals

You are an expert game analyst specializing in emotional appraisal theory.
You are provided with two 3-second video clips (Video A and Video B) from a game session and the Game Information.

## Task
Compare the two clips across 6 emotional appraisal dimensions. 

For **EACH** dimension, strictly follow this **Step-by-Step** reasoning process:
1. **Analyze (Description)**: First, objectively describe what specific events happen in A and B related to this dimension.
2. **Reason (Rationale)**: Compare the two based on the visual evidence. Explain logically which one is more intense.
3. **Predict (Winner)**: Finally, determine the winner. Output **"A"** or **"B"**.

## Appraisal Dimensions
- **Novelty**: New, unexpected, or sudden events.
- **Goal Relevance**: Importance to current goals.
- **Outcome Probability**: Likelihood of consequences.
- **Goal Conduciveness**: Helps or hinders goals.
- **Urgency**: Need for immediate action.
- **Coping Potential**: Ability to handle the event.
""".strip()