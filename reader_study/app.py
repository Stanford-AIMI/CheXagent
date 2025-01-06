import ast
import json
import os
import re
from datetime import datetime
from itertools import count

import pandas as pd
import pymongo
import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import streamlit_survey as ss
import yaml
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from yaml.loader import SafeLoader

# Streamlit app session variables
st.set_page_config(layout="wide")
if 'initialize' not in st.session_state:
    st.session_state.initialize = True
if 'edit_durations' not in st.session_state:
    st.session_state["edit_durations"] = []
if 'next' not in st.session_state:
    st.session_state['next'] = False
if 'edit_completed' not in st.session_state:
    st.session_state['edit_completed'] = False
if "edflg" not in st.session_state:
    st.session_state.edflg = False
if "edited" not in st.session_state:
    st.session_state['edited'] = []
if "previously_edited" not in st.session_state:
    st.session_state['previously_edited'] = []
if "df_reader" not in st.session_state:
    st.session_state['df_reader'] = None


# Initialize connection
# Uses st.cache_resource to only run once
@st.cache_resource
def init_connection():
    uri = st.secrets['mongo']['uri']
    return MongoClient(uri, server_api=ServerApi('1'))


client = init_connection()


def get_data(reader_id):
    db = client['chexagent']
    if int(reader_id) in [10, 99]:
        return {}
    else:
        return db[f"reader-{reader_id}"].find_one(sort=[('_id', pymongo.DESCENDING)])


# Authenticate
with open('readers.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)
_, authentication_status, username = authenticator.login()
if authentication_status:
    st.write(f'Welcome to the CheXagent Reader Study!')
elif authentication_status == False:
    st.error('Username or password is incorrect!')
elif authentication_status == None:
    st.warning('Please enter your provided username and password.')

# Data directories
data_folder = "data/"

# Set reader ID
if username:
    reader_id = config['credentials']['usernames'][username]['id']
else:
    reader_id = None

# Start survey
if reader_id:
    survey = ss.StreamlitSurvey()
    try:
        # Load previous responses on initialization
        if st.session_state.initialize:
            print('Looking for previous responses...')
            item = get_data(reader_id)
            if item:
                print('Previous responses found.')
                previously_edited = item['edited']
                st.session_state['previously_edited'] = previously_edited
                print(f"Previously edited samples: {previously_edited}")
                item.pop('_id', None)
                st.session_state["edit_durations"] = item.pop('edit_durations', None)
                st.session_state["edited"] = item.pop('edited', None)
                os.makedirs('temp', exist_ok=True)
                with open(f"temp/{str(reader_id)}.json", 'w') as f:
                    json.dump(item, f)
                survey.from_json(path=f"temp/{str(reader_id)}.json")

                # Filter out the samples that have already been submitted
                df_reader = pd.read_csv(os.path.join(data_folder, f"{reader_id}.csv"))
                df_reader = df_reader[~df_reader['study_id'].isin(st.session_state.previously_edited)]
                st.session_state['df_reader'] = df_reader
            else:
                print('No previous responses found.')
                df_reader = pd.read_csv(os.path.join(data_folder, f"{reader_id}.csv"))
                st.session_state['df_reader'] = df_reader
            st.session_state.initialize = False
        else:
            df_reader = st.session_state.df_reader
    except Exception as e:
        print('Error loading previous responses.')
        print(e)
        df_reader = st.session_state.df_reader
        survey = ss.StreamlitSurvey()
        st.session_state.initialize = False

    # Format data as dicts
    df_reader.sort_values('sample_id')
    reader_dict_reports = df_reader.set_index('study_id')['candidate'].to_dict()
    reader_dict_indications = df_reader.set_index('study_id')['indication'].to_dict()
    reader_dict_images = df_reader.set_index('study_id')['images'].to_dict()

    # Process image view names if nan
    for each_sample in reader_dict_images:
        button_number_count = 1
        counter = count(button_number_count)
        new_string = re.sub(r'nan', lambda x: f"'View {str(next(counter))}'", reader_dict_images[each_sample])
        string_representation = ast.literal_eval(new_string)
        sample_images_dict = {}
        for each in string_representation:
            key = each[0]
            original_key = each[0]
            counter = 1
            while key in sample_images_dict:
                key = f"{original_key} ({counter})"
                counter += 1
            sample_images_dict[key] = each[1]
        reader_dict_images[each_sample] = sample_images_dict

    # Number of samples
    study_ids = list(reader_dict_reports.keys())
    num_samples = len(study_ids)

    print('\n-----------------------------------')
    print(f"Reader ID: {reader_id}")
    print(f"Sample IDs: {study_ids}")
    print(f"Number of samples: {num_samples}")
    print('-----------------------------------\n')


    # Save function for pymongo
    def save(survey, reader_id=reader_id):
        json_file = json.loads(survey.to_json())
        json_file[f"edit_durations"] = st.session_state.get('edit_durations', 0)
        json_file[f"edited"] = st.session_state.get('edited', 0)
        db = client['chexagent']
        mycol = db[f"reader-{reader_id}"]
        x = mycol.insert_one(json_file)
        print(f"Saved to {x.inserted_id}!")


    # Stop if all samples have been reviewed
    if len(study_ids) == 0:
        st.error("No samples to review.")
        st.stop()

    # Survey pages
    pages = survey.pages(num_samples, on_submit=lambda: (
        save(survey), st.success("All responses have been recorded. You may close the window. Thank you!")))
    with pages:
        sample = study_ids[pages.current]
        images = reader_dict_images[sample]


        # Page buttons (Next, Submit, Previous)
        def next_sample():
            pages.next()
            st.session_state.pop('started', None)
            st.session_state.next = False
            st.session_state.edit_completed = False
            st.session_state.edflg = False


        pages.next_button = lambda pages: st.button(
            "Next",
            type="primary",
            use_container_width=True,
            on_click=next_sample,
            disabled=pages.current == pages.n_pages - 1,
            key=f"{pages.current}_btn_next",
        ) if st.session_state['next'] else None
        pages.submit_button = lambda pages: st.button(
            "Submit",
            type="primary",
            use_container_width=True,
        ) if st.session_state['next'] else None
        pages.prev_button = lambda pages: None

        # Start time
        if 'started' not in st.session_state:
            print(f"Now viewing sample {pages.current}: {sample}")
            st.session_state[f"start_edit_time_reader-{reader_id}_{sample}"] = datetime.now()
            st.session_state.started = True

        # Page content
        left, right = st.columns([0.6, 0.4])
        with left:
            """#### 1. Review DICOM:"""
            st.write(
                f"**Tips:** Scroll to Zoom, Scroll Click/Drag to Pan, Left Click/Drag to adjust Constrast (Left/Right) and Brightness (Up/Down).")

            options = st.radio('**DICOM View:**',
                               list(images.keys()), horizontal=True, key=f"radio_reader-{reader_id}_{sample}")

            # Retrieve image
            img_file = images[options]
            print(img_file)
            img_url = f"https://storage.googleapis.com/chexagent_reader_study/images/{sample}/{img_file}"

            # URL where cornerstone React component is hosted
            cornerstone_react_component_url = f"https://vilmedic.app/study/NLG?width=800&height=800&fileUrl={img_url}"
            # Embed via iframe
            components.iframe(cornerstone_react_component_url, width=700, height=700)
        with right:
            """#### 2. Review Report:"""
            st.write('**Exam Indication:**')
            st.write(reader_dict_indications[sample])
            st.write('**Report (from a Model or a Radiologist):**')

            # Display report
            st.write(reader_dict_reports[sample])

            # Edit report box, empty if written from scratch
            survey.text_area(
                "**Write:**" if reader_dict_reports[sample] == "Empty. Please write from scratch." else "**Edit:**",
                value="" if reader_dict_reports[sample] == "Empty. Please write from scratch." else reader_dict_reports[
                    sample].replace("**", ""), id=f"edit_reader-{reader_id}_{sample}", height=150,
                disabled=st.session_state.edflg)


            def disable_edit():
                st.session_state.edflg = True


            # Submit process
            if st.button("SUBMIT", key=f"edit_save_reader-{reader_id}_{sample}", on_click=disable_edit,
                         use_container_width=True, disabled=st.session_state.edflg):
                st.success(f"Submitted!")
                st.session_state[f"end_edit_time_reader-{reader_id}_{sample}"] = datetime.now()
                time = {
                    "id": f"time_reader-{reader_id}_{sample}",
                    "start": st.session_state[f"start_edit_time_reader-{reader_id}_{sample}"],
                    "end": st.session_state[f"end_edit_time_reader-{reader_id}_{sample}"],
                    "duration": st.session_state[f"end_edit_time_reader-{reader_id}_{sample}"] - st.session_state[
                        f"start_edit_time_reader-{reader_id}_{sample}"]
                }
                time['duration'] = time['duration'].total_seconds()
                st.session_state[f"edit_durations"].append(time)
                st.session_state['edited'].append(sample)
                st.session_state['edit_completed'] = True

            # Show feedback form if edit is completed
            if st.session_state['edit_completed']:
                """#### 3. Log Feedback:"""
                survey.multiselect('**3.1: Why did you make those edits?**',
                                   options=["[Content] False report of a finding in the image",
                                            "[Content] Missing a finding present in the image",
                                            "[Content] Misidentification of a finding's anatomic location in the image",
                                            '[Content] Misassessment of the severity of a finding in the image',
                                            '[Style] Poor report writing style',
                                            '[Style] Not written in a style that I prefer/am used to',
                                            '[Style] Inappropriate ordering of findings', 'Other'],
                                   id=f"why_reader-{reader_id}_{sample}")
                survey.text_area('**3.2: Please provide more details:**',
                                 id=f"why_detailed_reader-{reader_id}_{sample}")
                survey.select_slider(
                    "**3.3: This impression helps answer the exam indication:**", value='Neutral/Not Applicable',
                    options=["Strongly Disagree", "Disagree", "Neutral/Not Applicable", "Agree", "Strongly Agree"],
                    id=f"indication_reader-{reader_id}_{sample}"
                )
                survey.text_area('**3.4: Please explain your selection:**',
                                 id=f"indication_detailed_reader-{reader_id}_{sample}")
                survey.multiselect(
                    "**3.5: Do you think this drafted report improved your efficiency?**",
                    options=["Not Applicable (written report from scratch)", "No (did not improve efficiency)",
                             "Yes (improved interpretation efficiency)", "Yes (improved writing efficiency)"],
                    id=f"efficiency_reader-{reader_id}_{sample}"
                )
                survey.text_area("**3.6: Additional feedback, if any:**", id=f"feedback_reader-{reader_id}_{sample}")


                def enable_next():
                    save(survey)
                    st.session_state['next'] = True


                if st.button("SAVE RESPONSES", key=f"feedback_save_reader-{reader_id}_{sample}",
                             use_container_width=True, on_click=enable_next):
                    st.success(f"Feedback saved!")
