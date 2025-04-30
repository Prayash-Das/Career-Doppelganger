import re
import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
import pandas as pd
import faiss
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import numpy as np
from Career_Quiz import start_quiz, evaluate_answer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from difflib import get_close_matches
from resume_parser import extract_resume_text, extract_skills, extract_experience
import gradio as gr
# --------------------- Config ---------------------
CSV_PATH = "Profiles_2000_with_Embeddings.csv"
INDEX_PATH = "profiles_2000_faiss.index"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "deepseek-ai/deepseek-llm-7b-chat"
MENTOR_CSV = "Mentor_Profiles_2000_with_Embeddings.csv"
MENTOR_INDEX = "mentor_2000_faiss.index"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device}")

# --------------------- Load Models and Data ---------------------
df = pd.read_csv(CSV_PATH)
# Convert embedding strings to numpy arrays
df['embedding'] = df['embedding'].apply(eval).apply(np.array)
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
# Load the main FAISS index
index = faiss.read_index(INDEX_PATH)

# --------------------- Conversation History ---------------------
# We use a Gradio State component so that each user/session has its own conversation history.
# This state stores a list of dictionaries, each with keys "query" and "response".
# It is not displayed by default.
# --------------------- Helper Functions ---------------------
def detect_intent(query):
    query = query.lower()
    if "salary" in query or "compensation" in query:
        return "salary"
    elif "skills" in query or "tools" in query:
        return "skills"
    elif "career path" in query or "trajectory" in query:
        return "career_path"
    else:
        return "general"

def build_prompt(query, profiles, history):
    # Build a history block from the last 3 exchanges
    history_block = "\n".join(
        [f"Q: {item['query']}\nA: {item['response']}" for item in history[-3:]]
    )
    profile_context = "\n".join([
        f"{row['name']} - {row['current_title']} | {row['years_experience']} yrs in {row['industry']} | Skills: {row['skills']} | Path: {row['career_path']}"
        for _, row in profiles.iterrows()
    ])
    return f"""You are a career assistant. 
Answer strictly and specifically based on the user’s question and if the user asks anything related to the profile, then only bring up those relevant profiles. Be mindful, do not hallucinate and don't give wrong answers. 
Respond in a helpful, natural, human-style paragraph. Use only the facts below — do not hallucinate.
### User History:
{history_block}

### Current Question:
{query}
### Matching Profiles:
{profile_context}

Answer:"""

def generic_match(user_input, candidates):
    matches = get_close_matches(user_input.lower(), [str(c).lower() for c in candidates], n=1, cutoff=0.6)
    return matches[0] if matches else user_input

# --------------------- Core Function Wrappers for Gradio ---------------------
def gradio_career_qa(query, history_state):
    """Handles a single career Q&A query.
       history_state is a list (stored in gr.State) that holds prior conversation turns.
    """
    if query.lower() == "menu":
        return "Exiting Career Q&A mode.", history_state
    query_embedding = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(query_embedding, k=3)
    top_profiles = df.iloc[I[0]]
    prompt = build_prompt(query, top_profiles, history_state)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=650,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    new_history = history_state + [{"query": query, "response": response}]
    return response, new_history

def get_history_text(history_state):
    """Return conversation history as a formatted string."""
    return "\n".join([f"Q: {item['query']}\nA: {item['response']}" for item in history_state])

def gradio_doppelganger(title, years, industry, skills):
    try:
        years = int(years)
    except:
        return "⚠️ Please enter a valid number for years of experience."
    user_desc = f"{title} with {years} years in {industry}, skilled in {skills}"
    user_embedding = embedder.encode([user_desc], normalize_embeddings=True).astype("float32")
    D, I = index.search(user_embedding, k=10)
    for idx in I[0]:
        profile = df.iloc[idx]
        try:
            exp = int(profile['years_experience'])
        except:
            continue
        # Match if candidate is between 2 and 3 years ahead and in the same industry
        if (years + 2) <= exp <= (years + 3) and profile['industry'].strip().lower() == industry.strip().lower():
            return f"""🪞 Your Career Doppelgänger:
👤 Name: {profile['name']}
Title: {profile['current_title']}
Experience: {profile['years_experience']} yrs
Industry: {profile['industry']}
Skills: {profile['skills']}
Career Path: {profile['career_path']}"""
    return "❌ No suitable match found with 2–3 years more experience."

def summarize_experience(experience_list):
    """
    Summarizes the first 1–2 experience entries into readable sentences.
    """
    if not isinstance(experience_list, list) or not experience_list:
        return "N/A"

    summary_lines = []
    for exp in experience_list[:2]:  # summarize top 2 roles
        position = exp.get("position", "a role")
        company = exp.get("company", "a company")
        location = exp.get("location", "")
        timeline = exp.get("timeline", "a timeframe")
        loc_part = f" in {location}" if location else ""
        summary_lines.append(f"{position} at {company}{loc_part}  ({timeline})")

    return "\n".join(summary_lines)

    return sentences[0]
def gradio_resume_doppelganger(file):
    try:
        if file is None:
            return "❌ No file uploaded."

        with open(file.name, "rb") as f:
            file_bytes = f.read()

        # Extract details
        text = extract_resume_text(file_bytes)
        if "ERROR_PARSING_PDF" in text:
            return f"❌ Failed to parse PDF: {text}"

        skills, _ = extract_skills(text)
        experience_list = extract_experience(file_bytes)

        # Validate experience and skills
        has_valid_experience = bool(experience_list)
        has_valid_skills = bool(skills)

        if not has_valid_experience and not has_valid_skills:
            return "⚠️ Couldn't extract relevant experience or skills from your resume."

        # Prepare experience summary
        if has_valid_experience:
            summarized_exp = "\n".join(
                f"- {exp['position']} at {exp['company']} in {exp.get('location', 'Unknown')} ({exp['timeline']})"
                for exp in experience_list
            )

        # Prepare embedding description (just first role for matching)
        desc_parts = []
        if has_valid_experience:
            first_exp = experience_list[0]
            desc_parts.append(
                f"{first_exp['position']} at {first_exp['company']} in {first_exp.get('location', 'Unknown')} ({first_exp['timeline']})"
            )
        if has_valid_skills:
            desc_parts.append("skilled in " + ", ".join(skills))

        user_desc = " ".join(desc_parts)

        # Check embedding/index existence
        if 'embedder' not in globals() or 'index' not in globals() or 'df' not in globals():
            return "⚠️ Embedding system not loaded. Cannot find similar profiles."

        user_embedding = embedder.encode([user_desc], normalize_embeddings=True).astype("float32")
        D, I = index.search(user_embedding, k=3)
        matched_profiles = df.iloc[I[0]]

        # Compose final output
        if has_valid_experience and has_valid_skills:
            intro = (
                f"Based on your resume, you have experience in the following roles:\n{summarized_exp}\n\n"
                f"You also possess skills such as {', '.join(skills)}."
            )
        elif has_valid_experience:
            intro = (
                f"Based on your resume, you have experience in the following roles:\n{summarized_exp}"
            )
        elif has_valid_skills:
            intro = f"Based on your resume, you possess skills such as {', '.join(skills)}."

        match_intro = "Here are the top 3 professionals with similar profiles to yours:"
        matches = []
        for _, profile in matched_profiles.iterrows():
            match = (
                f"- {profile['name']} works as a {profile['current_title']} with "
                f"{profile['years_experience']} years of experience in the {profile['industry']} industry."
            )
            matches.append(match)

        return "\n".join([intro, match_intro] + matches)

    except Exception as e:
        return f"❌ ERROR_PROCESSING: {e}"

def gradio_mentor_mode(industry, role, switch, target, learn, skill):
    mentor_df = pd.read_csv(MENTOR_CSV)
    mentor_df['embedding'] = mentor_df['embedding'].apply(eval).apply(np.array)
    matched_industry = generic_match(industry, mentor_df['industry'].unique())
    matched_title = generic_match(role, mentor_df['career_path'].dropna().tolist())
    fallback_mode = False
    if switch.lower() == "yes":
        matched_target_industry = generic_match(target, mentor_df['industry'].unique())
        filtered_mentors = mentor_df[mentor_df['industry'].str.lower() == matched_target_industry.lower()]
        if filtered_mentors.empty:
            fallback_mode = True
        else:
            goal_description = f"Switching from {matched_industry} as {matched_title} to {matched_target_industry}"
    if switch.lower() != "yes" or fallback_mode:
        if learn.lower() == "yes":
            matched_skill = generic_match(skill, mentor_df['skills'].dropna().tolist())
            goal_description = f"{matched_title} in {matched_industry} looking to learn {matched_skill}"
            filtered_mentors = mentor_df[
                (mentor_df['industry'].str.lower() == matched_industry.lower()) &
                (mentor_df['skills'].str.lower().str.contains(matched_skill.lower()))
            ]
            if filtered_mentors.empty:
                return f"⚠️ No mentors found with skill '{matched_skill}' in {matched_industry}."
        else:
            goal_description = f"{matched_title} continuing in {matched_industry}"
            filtered_mentors = mentor_df[mentor_df['industry'].str.lower() == matched_industry.lower()]
    if filtered_mentors.empty:
        return "⚠️ No matching mentors found. Try a broader query."
    user_embedding = embedder.encode([goal_description], normalize_embeddings=True).astype("float32")
    mentor_embeddings = np.vstack(filtered_mentors['embedding'].tolist()).astype("float32")
    sub_index = faiss.IndexFlatL2(user_embedding.shape[1])
    sub_index.add(mentor_embeddings)
    D, I = sub_index.search(user_embedding, k=1)
    mentor = filtered_mentors.iloc[I[0][0]]
    result = f"""👤 Your Recommended Mentor:
Name: {mentor['name']}
Title: {mentor['current_title']}
Industry: {mentor['industry']}
Company: {mentor['company']}
Skills: {mentor['skills']}
Career Path: {mentor['career_path']}"""
    return result

# --------------------- Gradio UI ---------------------
with gr.Blocks(title="TwinPath Career Assistant") as demo:
    gr.Image(value="Logo.jpeg", show_label=False, show_download_button=False, width=250)
    gr.Markdown("## 👋 Welcome to TwinPath Career Assistant\nPersonalized career Q&A, mentor matching, and doppelgänger matching")
    
    with gr.Tab("Career Q&A"):
        qa_query = gr.Textbox(label="Ask your career question")
        qa_btn = gr.Button("Ask")
        qa_output = gr.Textbox(label="Answer")
        qa_history_state = gr.State([])

        with gr.Accordion("🕘 View Conversation History", open=False):
            qa_history_output = gr.Textbox(interactive=False, show_label=False, lines=20)

        qa_btn.click(fn=gradio_career_qa, inputs=[qa_query, qa_history_state], outputs=[qa_output, qa_history_state])
        qa_btn.click(fn=get_history_text, inputs=qa_history_state, outputs=qa_history_output)
	
    
    with gr.Tab("Doppelgänger Match"):
        dp_title = gr.Textbox(label="Your current title")
        dp_years = gr.Textbox(label="Years of experience")
        dp_industry = gr.Textbox(label="Your industry")
        dp_skills = gr.Textbox(label="Comma-separated skills")
        dp_output = gr.Textbox(label="Doppelgänger Match")
        dp_btn = gr.Button("Find Doppelgänger")
        dp_btn.click(fn=gradio_doppelganger, inputs=[dp_title, dp_years, dp_industry, dp_skills], outputs=dp_output)
    
    with gr.Tab("Resume Upload Doppelgänger"):
        # Use type="file" to help mobile users upload files
        resume_input = gr.File(label="Upload your Resume (PDF)", type="file")
        resume_output = gr.Textbox(label="Doppelgänger Match from Resume")
        resume_btn = gr.Button("Match Doppelgänger")
        resume_btn.click(fn=gradio_resume_doppelganger, inputs=resume_input, outputs=resume_output)
    
    with gr.Tab("Mentor Mode"):
        mentor_ind = gr.Textbox(label="Your current industry")
        mentor_role = gr.Textbox(label="Your current role/title")
        mentor_switch = gr.Textbox(label="Switch industry? (yes/no)")
        mentor_target = gr.Textbox(label="Target industry (if yes)")
        mentor_learn = gr.Textbox(label="Learn new skill? (yes/no)")
        mentor_skill = gr.Textbox(label="Skill to learn (if yes)")
        mentor_output = gr.Textbox(label="Recommended Mentor")
        mentor_btn = gr.Button("Find Mentor")
        mentor_btn.click(fn=gradio_mentor_mode, inputs=[mentor_ind, mentor_role, mentor_switch, mentor_target, mentor_learn, mentor_skill], outputs=mentor_output)
demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
