import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(processed_dir='processed_data', test_size=0.2, seed=42):
    """Simplified dataset splitter based on subject identifiers"""
    # Configure paths
    input_csv = os.path.join(processed_dir, 'metadata.csv')
    output_dir = os.path.join(processed_dir, 'splits')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare metadata
    df = pd.read_csv(input_csv)
    
    # Extract unique biological identifiers from filenames
    df['subject_id'] = df['image_path'].apply(
        lambda x: '_'.join(os.path.basename(x).split('_')[:2])
    )
    
    # Stratified split: train + temp
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['subject_id'],
        random_state=seed
    )
    
    # Cleanup temporary columns
    for split_df in [train_df, test_df]:
        split_df.drop(columns=['subject_id'], inplace=True)
    
    # Save splits
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

if __name__ == '__main__':
    split_dataset()
