from agent.schemas import TutorFeedback

# test instantiation
fb = TutorFeedback(
    priority_focus="TA",
    encouragement="Here is the evaluation: {'task_achievement': {'score': 8.0, ...}}",
    lesson_plan=["A", "B", "C"],
    next_essay_tips=["X", "Y"],
    rewrite_examples=[]
)
print(fb.model_dump())
