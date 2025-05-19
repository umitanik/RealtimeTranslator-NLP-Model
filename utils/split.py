from sklearn.model_selection import train_test_split

def split_data(data, test_size=0.1, random_state=42):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data