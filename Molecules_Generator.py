import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem

# Load the trained model
@st.cache_data
def load_model():
    model = AutoModelForCausalLM.from_pretrained("/mnt/c/Users/user/DD_Pipeline/model_training/smiles_gpt2/checkpoint-17496")
    return model

model = load_model()

# Load the tokenizer
@st.cache_data
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure the tokenizer uses the same pad token as during training
    return tokenizer

tokenizer = load_tokenizer()

def generate_smiles(prompt, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        input_ids, 
        max_length=max_length, 
        num_return_sequences=num_return_sequences,
        do_sample=True, 
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated sequences into SMILES strings
    smiles_list = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    # Filter valid SMILES using RDKit
    valid_smiles = []
    for smiles in smiles_list:
        if is_valid_smiles(smiles):
            valid_smiles.append(smiles)
    
    return valid_smiles

def is_valid_smiles(smiles):
    """Checks if the generated SMILES is valid using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return True
    except:
        return False
    return False

def convert_df_to_csv(df):
    """Convert dataframe to CSV for downloading."""
    return df.to_csv(index=False)

# Streamlit UI
st.title('SMILES Generator')

# Application description
st.markdown("""
**SMILES Generator Using Fine-Tuned GPT-2 Model**

Welcome to the SMILES Generator, a sophisticated tool built using a GPT-2 model with 137 million parameters, meticulously fine-tuned on a dataset of 100,000 molecular structures. This application is designed to generate novel molecular structures based on the Simplified Molecular Input Line Entry System (SMILES) format.

### Key Features:
- **Novel Molecule Generation**: Leverages a fine-tuned GPT-2 model to create novel and unique molecular structures that are not found in the training dataset.
- **Validity Checks**: Utilizes RDKit, a powerful cheminformatics software, to ensure that each generated molecule is valid according to standard chemical rules.
- **Drug-Like Molecule Focus**: Specifically generates molecules with drug-like properties, making this tool particularly valuable for researchers and professionals in drug discovery and pharmaceutical sciences.

### How It Works:
1. **Input Fragment**: Users can input a starting fragment of a SMILES string to guide the generation process.
2. **Customization**: Adjust the number of molecules to generate and the maximum length of the SMILES strings to tailor the output to your needs.
3. **Generation and Validation**: The model generates SMILES strings, which are then validated and filtered by RDKit to ensure that they represent viable, drug-like molecules.
""")

st.sidebar.header("Settings")
user_input = st.sidebar.text_input("Enter a starting fragment of a SMILES string:", "CCO")
num_sequences = st.sidebar.number_input("Number of SMILES to generate:", min_value=1, max_value=500, value=5)
max_len = st.sidebar.number_input("Maximum length of SMILES string:", min_value=50, max_value=1000, value=100)

if st.sidebar.button('Generate SMILES'):
    with st.spinner('Generating SMILES...'):
        generated_smiles = generate_smiles(user_input, max_length=max_len, num_return_sequences=num_sequences)
    
    if generated_smiles:
        # Create a DataFrame to store the valid SMILES
        smiles_df = pd.DataFrame(generated_smiles, columns=["SMILES"])
        
        # Display the valid SMILES
        st.write("### Valid Generated SMILES:")
        st.dataframe(smiles_df)

        # Provide the option to download the SMILES as a CSV
        csv_data = convert_df_to_csv(smiles_df)
        st.download_button(
            label="Download Generated SMILES as CSV",
            data=csv_data,
            file_name="generated_smiles.csv",
            mime="text/csv"
        )
    else:
        st.error("No valid SMILES generated. Try adjusting the parameters or input fragment.")
