import yaml
import pandas as pd
import sys
sys.path.append("Scripts/Functions/Metrics")
from RMSE_R_squared import pooled_R_squared, pooled_RMSE 

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
    results = []

    pred_vars = config['variables']['pred']
    truth_vars = config['variables']['truth']
    assert len(pred_vars) == len(truth_vars), "Mismatch in predicted and truth ."

    for split in ['train', 'val', 'test']:
        pred_files = config['datasets'][split]['pred']
        truth_files = config['datasets'][split]['truth']

        assert len(pred_files) == len(pred_vars), f"{split}: Pred  and variables length mismatch"
        assert len(truth_files) == len(truth_vars), f"{split}: Truth  and variables length mismatch"

        for i, (pred_file, truth_file) in enumerate(zip(pred_files, truth_files)):
            pred_var = pred_vars[i]
            truth_var = truth_vars[i]

            r2 = pooled_R_squared(pred_file, truth_file, pred_var, truth_var).compute().item()
            rmse = pooled_RMSE(pred_file, truth_file, pred_var, truth_var).compute().item()


            results.append({
                "Split_Category": split,
                "Variable_Name": pred_var,
                "RMSE": round(rmse, 3),
                "R2": round(r2, 3)
            })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("Outputs/metrics_summary_bicubic_vs_truth.csv", index=False)

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
