# Diabetes Check

A minimalist diabetes risk screening app built with Streamlit.

## Run locally

```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deploy publicly

The easiest public option is Streamlit Community Cloud:

1. Push this project to a GitHub repository.
2. Open https://streamlit.io/cloud
3. Connect the GitHub repo.
4. Choose the app file: `app.py`
5. Confirm the Python version and dependencies from `requirements.txt`.
6. Deploy.

The app will then be available online with the model running in the cloud.

## Notes

- This is an educational screening demo, not a medical diagnosis.
- It uses a model trained on the Pima Indians Diabetes dataset.
- The model and threshold are saved in `diabetes_model.joblib`.
