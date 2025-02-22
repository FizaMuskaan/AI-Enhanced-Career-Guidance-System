import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import base64

# Path to the background image
background_image_path = 'cg bg.jpg'  # Updated to new image name

# Add custom CSS for styling
def add_custom_css():
    st.markdown(
        f"""
        <style>
        body {{
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }}
        .stApp {{
            background-image: url(data:image/jpg;base64,{get_image_base64(background_image_path)});
            background-size: cover;
            background-position: center;
        }}
        h1 {{
            font-family: 'Georgia', serif;
            font-size: 50px;
            font-weight: bold;
            color: #000000;
            text-align: center;
            text-shadow: 3px 3px 5px rgba(0, 0, 0, 0.5);
        }}
        h2 {{
            font-family: 'Georgia', serif;
            font-size: 35px;
            font-weight: bold;
            color: #000000;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
        }}
        h3 {{
            font-family: 'Georgia', serif;
            font-size: 30px;
            font-weight: bold;
            color: #000000;
            text-align: center;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        }}
        .stButton button {{
            background-color: #334b5f;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
        }}
        .stButton button:hover {{
            background-color: #1e3449;
        }}
        .description-text {{
            font-size: 20px;
            color: #000000;
            text-align: center;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 40px;  /* Add space below the description */
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Call the function to apply the CSS
add_custom_css()

# Streamlit Interface
st.title("**CareerLaunch**")

# Project Description
st.markdown("""
    <div class="description-text">
        Welcome to CareerLaunch – your personalized career guidance platform. This system uses advanced machine learning 
        algorithms to recommend the best career paths based on your skills and experience. Whether you're just starting out or 
        looking for a career change, we provide tailored recommendations to help you take the next step towards your dream job.
        Start by providing your skills and experience, and let us guide you towards your ideal career!
    </div>
    """, unsafe_allow_html=True)

# Additional Title: Career Path Recommendation System
st.subheader("**Career Path Recommendation System**")

# Load the dataset
try:
    career_data = pd.read_csv('dataset9000.csv')
    st.subheader("Dataset Preview 📊")
    st.dataframe(career_data.head())
except FileNotFoundError:
    st.error("CSV file not found. Please ensure 'dataset9000.csv' is in the same directory as this script.")
    st.stop()


experience_map = {
    'Not Interested': 0,
    'Poor': 1,
    'Beginner': 2,
    'Average': 3,
    'Intermediate': 4,
    'Excellent': 5,
    'Professional': 6
}

# Ensure all skills columns are correctly mapped
try:
    for column in career_data.columns[:-1]:  # Exclude 'Role' column
        career_data[column] = career_data[column].map(experience_map)
except KeyError:
    st.error("Ensure all columns except 'Role' contain valid experience levels (e.g., 'Beginner', 'Professional').")
    st.stop()

# Drop rows with missing or invalid data
career_data = career_data.dropna()

# Ensure the target column is categorical
career_data['Role'] = career_data['Role'].astype('category')

# Split data into features and target
X = career_data.drop(columns='Role')
y = career_data['Role']

# Train the model
model = KNeighborsClassifier(n_neighbors=5) # Added random_state for reproducibility
try:
    model.fit(X, y)
except ValueError as e:
    st.error(f"Error training the model: {e}")
    st.stop()

# Career details (same as before)
career_details = {
    "Database Administrator": {
        "Description": "Responsible for managing databases, ensuring they run efficiently, and overseeing database security.",
        "Skills Required": "Database Fundamentals, SQL, Data Management",
        "Salary": "₹6,00,000 - ₹9,00,000 per year",
        "Education": "Bachelor's degree in Computer Science or related fields",
        "Growth": "Steady demand as databases are integral to businesses of all sizes.",
        "Location": "Major cities like Bengaluru, Hyderabad, Pune, and Delhi"
    },
    "Hardware Engineer": {
        "Description": "Designs, tests, and develops physical hardware components and systems.",
        "Skills Required": "Circuit Design, Troubleshooting, Embedded Systems",
        "Salary": "₹4,50,000 - ₹8,00,000 per year",
        "Education": "Bachelor's degree in Electronics, Electrical Engineering, or related fields",
        "Growth": "With the rise in IoT and consumer electronics, demand is growing.",
        "Location": "Bengaluru, Chennai, Pune, and Delhi"
    },
    "Application Support Engineer": {
        "Description": "Provides technical support for software applications, ensuring smooth operations.",
        "Skills Required": "Troubleshooting, Software Applications, Customer Support",
        "Salary": "₹4,00,000 - ₹7,00,000 per year",
        "Education": "Bachelor's degree in Computer Science or related fields",
        "Growth": "Demand is stable, especially in IT service companies.",
        "Location": "Chennai, Bengaluru, Noida, Mumbai"
    },
    "Cyber Security Specialist": {
        "Description": "Protects systems and networks from cyber threats, ensuring data security.",
        "Skills Required": "Network Security, Ethical Hacking, Encryption",
        "Salary": "₹8,00,000 - ₹15,00,000 per year",
        "Education": "Bachelor's degree in Computer Science, Cyber Security, or related fields",
        "Growth": "High demand due to increasing cybersecurity threats globally.",
        "Location": "Bengaluru, Hyderabad, Pune, Delhi"
    },
    "Networking Engineer": {
        "Description": "Designs and manages computer networks for businesses, ensuring optimal performance.",
        "Skills Required": "Routing, Switching, Network Security, Troubleshooting",
        "Salary": "₹5,00,000 - ₹9,00,000 per year",
        "Education": "Bachelor's degree in Computer Networking, Computer Science, or related fields",
        "Growth": "Steady demand as businesses continue to expand their networks.",
        "Location": "Bengaluru, Pune, Hyderabad, Delhi"
    },
    "Software Developer": {
        "Description": "Develops and maintains software applications, ensuring functionality and performance.",
        "Skills Required": "Programming, Problem-Solving, System Design",
        "Salary": "₹6,00,000 - ₹12,00,000 per year",
        "Education": "Bachelor's degree in Computer Science or related fields",
        "Growth": "High demand, with significant growth in technology-driven sectors.",
        "Location": "Bengaluru, Pune, Hyderabad, Chennai"
    },
    "API Specialist": {
        "Description": "Designs and manages APIs to ensure smooth integration between software systems.",
        "Skills Required": "API Design, RESTful Services, Programming",
        "Salary": "₹7,00,000 - ₹13,00,000 per year",
        "Education": "Bachelor's degree in Computer Science or related fields",
        "Growth": "Growing demand with the increase in microservices architecture.",
        "Location": "Bengaluru, Pune, Noida, Delhi"
    },
    "Project Manager": {
        "Description": "Manages projects from initiation to completion, ensuring timely delivery.",
        "Skills Required": "Project Management, Leadership, Agile Methodologies",
        "Salary": "₹10,00,000 - ₹20,00,000 per year",
        "Education": "Bachelor's degree in any field, with certifications like PMP or Scrum Master",
        "Growth": "Steady growth in demand, especially in IT and construction industries.",
        "Location": "Bengaluru, Delhi, Mumbai, Chennai"
    },
    "Information Security Specialist": {
        "Description": "Ensures the protection of an organization's sensitive data and systems from security breaches.",
        "Skills Required": "Risk Management, Encryption, Security Tools",
        "Salary": "₹8,00,000 - ₹15,00,000 per year",
        "Education": "Bachelor's degree in Information Security or related fields",
        "Growth": "Demand is increasing as organizations focus more on data protection.",
        "Location": "Bengaluru, Pune, Hyderabad, Delhi"
    },
    "Technical Writer": {
        "Description": "Creates user manuals, guides, and documentation for technical products and services.",
        "Skills Required": "Writing Skills, Technical Knowledge, Documentation Tools",
        "Salary": "₹5,00,000 - ₹8,00,000 per year",
        "Education": "Bachelor's degree in English, Communications, or a technical field",
        "Growth": "Stable demand, with increasing need in IT and software industries.",
        "Location": "Chennai, Bengaluru, Pune, Noida"
    },
    "AI/ML Specialist": {
        "Description": "Develops and implements AI and machine learning models to solve complex problems.",
        "Skills Required": "Machine Learning, Data Science, Neural Networks",
        "Salary": "₹12,00,000 - ₹25,00,000 per year",
        "Education": "Master's degree in AI, Machine Learning, or related fields",
        "Growth": "High demand with advancements in automation, AI, and data analysis.",
        "Location": "Bengaluru, Hyderabad, Pune, Delhi"
    },
    "Software Tester": {
        "Description": "Tests software applications to ensure quality and performance standards are met.",
        "Skills Required": "Manual Testing, Automation Testing, Bug Tracking",
        "Salary": "₹4,00,000 - ₹7,00,000 per year",
        "Education": "Bachelor's degree in Computer Science or related fields",
        "Growth": "Steady demand in IT services, with the rise of DevOps and automation.",
        "Location": "Chennai, Pune, Bengaluru, Noida"
    },
    "Business Analyst": {
        "Description": "Analyzes business requirements and helps implement solutions to improve efficiency.",
        "Skills Required": "Data Analysis, Requirements Gathering, Communication",
        "Salary": "₹6,00,000 - ₹12,00,000 per year",
        "Education": "Bachelor's degree in Business, IT, or related fields",
        "Growth": "High demand in diverse industries, including IT and finance.",
        "Location": "Bengaluru, Hyderabad, Chennai, Mumbai"
    },
    "Customer Service Executive": {
        "Description": "Handles customer queries, provides support, and ensures customer satisfaction.",
        "Skills Required": "Communication, Problem-Solving, Customer Service",
        "Salary": "₹3,00,000 - ₹5,00,000 per year",
        "Education": "Bachelor's degree or relevant experience in customer service",
        "Growth": "Consistent demand, particularly in BPO, retail, and e-commerce.",
        "Location": "Pune, Noida, Bengaluru, Delhi"
    },
    "Data Scientist": {
        "Description": "Uses statistical and computational techniques to analyze and interpret large datasets.",
        "Skills Required": "Data Analysis, Python, Machine Learning, Big Data",
        "Salary": "₹10,00,000 - ₹18,00,000 per year",
        "Education": "Master's or Bachelor's degree in Data Science, Statistics, or related fields",
        "Growth": "Rapid demand due to the increasing reliance on data-driven decisions.",
        "Location": "Bengaluru, Hyderabad, Pune, Delhi"
    },
    "Helpdesk Engineer": {
        "Description": "Provides technical support for IT issues, ensuring users' systems are functional.",
        "Skills Required": "Troubleshooting, Networking, Communication",
        "Salary": "₹3,00,000 - ₹5,00,000 per year",
        "Education": "Bachelor's degree in Computer Science or related fields",
        "Growth": "Stable demand in IT support and services.",
        "Location": "Bengaluru, Chennai, Pune, Noida"
    },
    "Graphics Designer": {
        "Description": "Creates visual content for marketing, branding, and communication purposes.",
        "Skills Required": "Graphic Design, Adobe Suite, Creativity",
        "Salary": "₹4,50,000 - ₹8,00,000 per year",
        "Education": "Bachelor's degree in Design, Arts, or related fields",
        "Growth": "Growing demand in advertising, branding, and digital media.",
        "Location": "Mumbai, Bengaluru, Delhi, Chennai"
    }
}

# User Input Section
st.header("🎓 Enter Your Skills and Experience")
skills_input = {}
for column in X.columns:  # Loop through feature columns
    skills_input[column] = st.selectbox(
        f"Rate your {column} experience:",
        options=["Not Interested", "Poor", "Beginner", "Average", "Intermediate", "Excellent", "Professional"]
    )

# Process the inputs and make a prediction
if st.button("🔮 Predict Career Path"):
    try:
        # Map user inputs to numerical values
        user_data = [experience_map[skills_input[column]] for column in X.columns]
        user_data = np.array(user_data).reshape(1, -1)

        # Predict the career role
        predicted_role = model.predict(user_data)[0]

        # Display predicted role and career details
        st.success(f"✨ Suggested Career Path: {predicted_role} ✨")
        details = career_details.get(predicted_role, {})
        st.markdown(f"**📝 Description:** {details.get('Description', 'N/A')}")
        st.markdown(f"**🔧 Skills Required:** {details.get('Skills Required', 'N/A')}")
        st.markdown(f"**💰 Salary Range:** {details.get('Salary', 'N/A')}")
        st.markdown(f"**🎓 Education Required:** {details.get('Education', 'N/A')}")
        st.markdown(f"**📈 Growth Potential:** {details.get('Growth', 'N/A')}")
        st.markdown(f"**📍 Location:** {details.get('Location', 'N/A')}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Select your skills and click 'Predict Career Path' to get recommendations.")
