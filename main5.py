import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from scipy.stats import mannwhitneyu
import matplotlib
# Agg backend prevents Windows GUI rendering errors
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import warnings

# Configuration
warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on computation device: {DEVICE}")

# ---------------------------------------------------------
# 1. DATA GENERATION (LUNG-ENDOTHELIUM AXIS)
# ---------------------------------------------------------

FILE_PATH = r"D:\ABC\RTI\Sepsis_BioMarker.csv"

def load_or_generate_data(path):
    if os.path.exists(path):
        print(f"Loading dataset from {path}...")
        df = pd.read_csv(path)
    else:
        print(f"File not found. Generating synthetic dataset based on paper schema...")
        n_rows = 12000
        # 40% RTI, 30% GU, 20% Abd, 10% Other
        infection_types = np.random.choice(['res', 'gu', 'abd', 'other'], n_rows, p=[0.4, 0.3, 0.2, 0.1])
        
        data = {
            'res_infection': (infection_types == 'res').astype(int),
            'gu_infection': (infection_types == 'gu').astype(int),
            'abdominal_infection': (infection_types == 'abd').astype(int),
            'age': np.random.normal(65, 15, n_rows).astype(int),
            'lactate': np.abs(np.random.normal(2.5, 1.5, n_rows)),
            'pct': np.abs(np.random.normal(5.0, 6.0, n_rows)),
            'death_binary': np.random.choice([0, 1], n_rows, p=[0.75, 0.25])
        }
        
        ang2, esel, psel, icam, sofa_res = [], [], [], [], []
        
        for i in range(n_rows):
            is_res = data['res_infection'][i]
            # SOFA Res: Higher in RTI
            sr = np.random.choice([0,1,2,3,4], p=[0.05, 0.1, 0.2, 0.35, 0.3] if is_res else [0.5, 0.3, 0.15, 0.05, 0.0])
            sofa_res.append(sr)
            
            # Angiopoietin-2: Specific Axis Marker
            base_ang = 28.0 if is_res else 12.0
            ang2.append(max(1.0, base_ang + (sr * 5.0) + np.random.normal(0, 8)))
            
            # E-Selectin: Correlated with Ang-2
            base_esel = 35.0 if is_res else 22.0
            esel.append(max(1.0, base_esel + (sr * 3.0) + np.random.normal(0, 10)))
            
            # P-Selectin & ICAM
            psel.append(max(1.0, np.random.normal(14 if is_res else 11, 4)))
            icam.append(max(50.0, np.random.normal(180 if is_res else 145, 40)))

        data['angiopoetin2'] = ang2
        data['eselectin'] = esel
        data['pselectin'] = psel
        data['icam1'] = icam
        data['vcam1'] = np.random.normal(90, 20, n_rows)
        data['sofa_res'] = sofa_res
        
        df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    imputer = SimpleImputer(strategy='median')
    cols = ['angiopoetin2', 'eselectin', 'pselectin', 'icam1', 'vcam1', 'sofa_res', 'lactate', 'pct']
    df[cols] = imputer.fit_transform(df[cols])

    df['Group_Label'] = -1
    mask_rti = df['res_infection'] == 1
    mask_norti = (df['res_infection'] == 0) & ((df['gu_infection'] == 1) | (df['abdominal_infection'] == 1))
    df.loc[mask_rti, 'Group_Label'] = 1
    df.loc[mask_norti, 'Group_Label'] = 0
    
    return df[df['Group_Label'] != -1].copy()

# ---------------------------------------------------------
# 2. PYTORCH MODEL
# ---------------------------------------------------------

class SepsisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

class LungEndoNet(nn.Module):
    def __init__(self, input_dim):
        super(LungEndoNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def train_model(X, y):
    print("Training PyTorch Model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    train_loader = DataLoader(SepsisDataset(X_train_sc, y_train), batch_size=64, shuffle=True)
    model = LungEndoNet(X.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(15):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model, scaler, X_train_sc, X_test_sc, y_test

# ---------------------------------------------------------
# 3. XAI IMPLEMENTATION (LIME & SHAP)
# ---------------------------------------------------------

def run_xai_analysis(model, scaler, X_train, X_test, feature_names):
    print("\n--- Starting XAI Analysis (LIME & SHAP) ---")
    
    def predict_proba_wrapper(x_numpy):
        model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(x_numpy, dtype=torch.float32).to(DEVICE)
            logits = model(tensor_x).cpu().numpy()
            # Return (N, 2)
            return np.hstack((1-logits, logits))

    # Find High-Risk Instance
    probs = predict_proba_wrapper(X_test)[:, 1]
    target_idx = np.argsort(probs)[-1]
    target_instance = X_test[target_idx] # Shape (Features,)
    print(f"Analyzing High-Confidence RTI Case at Index {target_idx} (Prob: {probs[target_idx]:.4f})")

    # --- LIME ---
    print("Generating LIME Explanation...")
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Non-RTI', 'RTI'],
        mode='classification',
        discretize_continuous=True
    )
    exp_lime = explainer_lime.explain_instance(target_instance, predict_proba_wrapper, num_features=len(feature_names))
    
    fig = exp_lime.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('Figure_XAI_1_LIME.png')
    plt.close()

    # --- SHAP ---
    print("Generating SHAP Explanations (KernelExplainer)...")
    background = shap.kmeans(X_train, 25)
    explainer_shap = shap.KernelExplainer(predict_proba_wrapper, background)
    
    # Subset
    X_subset = X_test[:50].copy()
    X_subset[0] = target_instance
    
    # Get SHAP values
    shap_values_raw = explainer_shap.shap_values(X_subset, nsamples=100)
    
    # --- CRITICAL FIX FOR SHAP DIMENSIONS ---
    # KernelExplainer might return:
    # 1. List of arrays [ (N,F), (N,F) ] (Classification)
    # 2. Single array (N,F,2) (Vector output)
    
    vals_rti = None
    base_val_rti = None
    
    # Handle Base Value
    if hasattr(explainer_shap.expected_value, '__iter__'):
        base_val_rti = float(explainer_shap.expected_value[1])
    else:
        base_val_rti = float(explainer_shap.expected_value)
        
    # Handle SHAP Values
    if isinstance(shap_values_raw, list):
        # It's a list, take index 1
        vals_rti = shap_values_raw[1] # Shape (N, F)
    elif len(np.array(shap_values_raw).shape) == 3:
        # It's (N, F, Classes), take slice 1 from last dim
        vals_rti = np.array(shap_values_raw)[:, :, 1] # Shape (N, F)
    else:
        # Fallback
        vals_rti = np.array(shap_values_raw)

    print(f"SHAP Base Value: {base_val_rti:.4f}")
    print(f"SHAP Values Shape (Class 1): {vals_rti.shape}")

    # Create Explanation Object
    # .values must be 1D for waterfall (Features,)
    shap_exp_single = shap.Explanation(
        values=vals_rti[0], 
        base_values=base_val_rti, 
        data=X_subset[0], 
        feature_names=feature_names
    )

    # 1. Waterfall
    print("Creating SHAP Waterfall Plot...")
    plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_exp_single, show=False, max_display=10)
    plt.tight_layout()
    plt.savefig('Figure_XAI_2_SHAP_Waterfall.png')
    plt.close()

    # 2. Force Plot (HTML)
    print("Creating SHAP Force Plot...")
    force_plot = shap.force_plot(base_val_rti, vals_rti[0], X_subset[0], feature_names=feature_names, show=False)
    shap.save_html("Figure_XAI_3_SHAP_Force.html", force_plot)

    # 3. Decision Plot
    print("Creating SHAP Decision Plot...")
    plt.figure(figsize=(8, 6))
    shap.decision_plot(base_val_rti, vals_rti[:20], features=X_subset[:20], feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('Figure_XAI_4_SHAP_Decision.png')
    plt.close()

    # 4. Summary Plot
    print("Creating SHAP Summary Plot...")
    plt.figure(figsize=(8, 6))
    shap.summary_plot(vals_rti, X_subset, feature_names=feature_names, show=False)
    plt.savefig('Figure_XAI_5_SHAP_Summary.png')
    plt.close()

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    df_raw = load_or_generate_data(FILE_PATH)
    df = preprocess_data(df_raw)
    
    biomarkers = ['angiopoetin2', 'eselectin', 'pselectin', 'icam1', 'vcam1', 'sofa_res', 'lactate', 'pct']
    features = df[biomarkers].values
    labels = df['Group_Label'].values
    
    model, scaler, X_train_sc, X_test_sc, y_test = train_model(features, labels)
    
    probs = model(torch.tensor(X_test_sc, dtype=torch.float32).to(DEVICE)).cpu().detach().numpy()
    auc = roc_auc_score(y_test, probs)
    print(f"Model AUC: {auc:.4f}")
    
    with open("Table_Model_Metrics.txt", "w") as f:
        f.write(f"Model: PyTorch EndoNet\nAUC: {auc:.4f}\n")
        
    # Demographics
    g1 = df[df['Group_Label']==1]
    g2 = df[df['Group_Label']==0]
    with open("Table_1_Stats.txt", "w") as f:
        f.write(f"Variable | RTI Mean | Non-RTI Mean | p-val\n")
        for v in ['angiopoetin2', 'eselectin']:
            s, p = mannwhitneyu(g1[v], g2[v])
            f.write(f"{v} | {g1[v].mean():.2f} | {g2[v].mean():.2f} | {p:.3e}\n")
            
    # Run XAI
    run_xai_analysis(model, scaler, X_train_sc, X_test_sc, biomarkers)
    print("\nAll Analysis Completed Successfully.")

if __name__ == "__main__":
    main()