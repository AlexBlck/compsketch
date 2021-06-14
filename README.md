# Compositional Sketch Search

Official repository for ICIP 2021 Paper: Compositional Sketch Search

## Requirements
Install and activate conda environment
```commandline
conda env create -f env.yml
conda activate sketchsearch
cd src
```
## Indexing
Either download faiss index for OpenImages test set [here](www.google.com) or index your own dataset like so:
```commandline
python indexing.py --root PathToImages --name IndexName 
```

## Searching
Run the streamlit app
```commandline
streamlit run streamlit_app.py
```
If you are using a custom dataset:
- Pick 'Custom' in the Dataset selection box.
- Pick index file in the second selection box.
- Paste the same dataset root you used during indexing.

Sketch objects, using one colour per object.
Press "Search" to see results!
