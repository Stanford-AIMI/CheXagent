# CheXagent Reader Study

Website for the CheXagent Reader Study

The responses are saved to a MongoDB dataset. The DICOM images are loaded from a GCP Bucket and displayed via an
embedded version of a Cornerstone3D component, which has windowing, zooming, and panning.

To run the app:

```
streamlit run app.py
```

--------------
Organize the data specifying the samples in ```data/<id>.csv``` with the following structure:

```
sample_id,study_id,images,indication,candidate,reference
0,p14_p14841168_s52365850,['ffd311aa-b1ad24f7-29b178ef-4423264a-d0298e46.dcm'],"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.","Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.","Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
...
```

- ```sample_id```: integer specifying the order the samples will be presented in
- ```study_id```: string specifying the folder name containing the sample DICOM files
- ```images```: string representation of a list of .dcm file names
- ```indication```: string of exam indication
- ```candidate```: string of CheXagent generated report
- ```reference```: string of reference report (not used in the reader study)

--------------
```readers.yaml```: Reader credentials, used to specify the samples each readers see so you can show each reader
different samples.

```
credentials:
  usernames:
    reader1:                                 # username
      password: reader1                      # password
      id: 1                                  # loads samples from 1.csv
      name: Reader 1                         # name
    ...
cookie:
  expiry_days: 0
  key: random_signature_key # Must be string
  name: random_cookie_name
```

--------------
Other Relevant Files:

- ```cors.json```: CORS configuration for ViLMedic