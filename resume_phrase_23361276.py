import os
import fitz  # PyMuPDF
import spacy
import pandas as pd
from collections import Counter
from spacy.matcher import PhraseMatcher
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn

# Load spaCy's small English model
nlp_model = spacy.load("en_core_web_sm")

# Define the path to resumes and skills CSV
resume_dir = r'./Resume'  # Path where resumes are saved
pdf_files = [os.path.join(resume_dir, f) for f in os.listdir(resume_dir) if os.path.isfile(os.path.join(resume_dir, f))]
skills_path = r'./skills2.csv'

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    document = fitz.open(pdf_file)
    content = []
    for page in document:
        page_text = page.get_text()
        content.append(page_text)
    return " ".join(content)

# Function to create a skill profile from a resume
def generate_skill_profile(pdf_file):
    resume_text = extract_text_from_pdf(pdf_file)
    resume_text = resume_text.lower().replace("\\n", " ")

    # Read skills from CSV file
    skills_df = pd.read_csv(skills_path, encoding='ansi')
    
    # Define categories and their abbreviations
    skill_categories = {
        'Statistics': 'Stats',
        'Mathematics': 'Math',
        'Artificial Intelligence': 'AI',
        'Programming': 'Prog',
        'Cloud Computing': 'CloudComp',
        'Digital Transformation Manager': 'DTManager'
    }
    
    phrase_matcher = PhraseMatcher(nlp_model.vocab)
    
    for category, abbreviation in skill_categories.items():
        skill_phrases = [nlp_model(phrase) for phrase in skills_df[category].dropna()]
        phrase_matcher.add(abbreviation, None, *skill_phrases)
    
    doc = nlp_model(resume_text)
    matches = phrase_matcher(doc)
    
    # Collect matched skills and their categories
    matched_skills = [(nlp_model.vocab.strings[match_id], span.text) for match_id, start, end in matches for span in [doc[start:end]]]
    
    # Create a DataFrame from matched skills
    skills_counter = Counter(matched_skills)
    
    if not skills_counter:
        return pd.DataFrame(columns=['Candidate Name', 'Category', 'Skill', 'Count'])
    
    skills_df = pd.DataFrame(skills_counter.items(), columns=['Category_Skill', 'Count'])
    
    # Split Category_Skill into separate columns
    skills_df[['Category', 'Skill']] = skills_df['Category_Skill'].apply(lambda x: pd.Series(x))
    skills_df.drop(columns=['Category_Skill'], inplace=True)
    
    # Extract candidate name from filename
    filename = os.path.basename(pdf_file)
    candidate_name = filename.split('_')[0].lower()
    skills_df['Candidate Name'] = candidate_name
    
    return skills_df[['Candidate Name', 'Category', 'Skill', 'Count']]

# Create profiles for all resumes
combined_profiles = pd.DataFrame()

for pdf_file in pdf_files:
    profile = generate_skill_profile(pdf_file)
    combined_profiles = pd.concat([combined_profiles, profile], ignore_index=True)

# Count skills under each category and visualize with Matplotlib
category_count = combined_profiles.groupby(['Candidate Name', 'Category'])['Skill'].count().unstack(fill_value=0).reset_index()
profile_data = category_count.set_index('Candidate Name')

plt.rcParams.update({'font.size': 10})
ax = profile_data.plot.barh(title="Skills per category", stacked=True, figsize=(25, 7), color=sns.color_palette("Set2"))

labels = []
for column in profile_data.columns:
    for value in profile_data[column]:
        labels.append(f"{column}: {value}")

patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x() + width / 2.
        y = rect.get_y() + rect.get_height() / 2.
        ax.text(x, y, label, ha='center', va='center')

plt.show()
