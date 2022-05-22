import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def label_attacks(attack):
    """
    Maps attack labels to integer values
    
    Args:
        attack: String. Label for the attack value of a certain input entry.
    
    Returns:
        Integer value associated with the traffic. (0 - normal, 1 - dos_attack, 2 - probe_attack, 3 - priviledge_attack, 4 - access_attack)
    """

    # lists to hold our attack classifications
    dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
    probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
    privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
    access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

    if attack in dos_attacks:
        return 1
    if attack in probe_attacks:
        return 2
    if attack in privilege_attacks:
        return 3
    if attack in access_attacks:
        return 4
    else: 
        return 0

def load_data(filename):
    """
    Loads and preprocesses data.
    
    Args:
        filename: String. Path to the dataset directory.
    Returns:
        Pandas dataframe of shape [num_examples, num_features]
    """
    data_df = pd.read_csv(filename)
    # add the column labels
    columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent'
    ,'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files'
    ,'num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate'
    ,'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate'
    ,'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate'
    ,'dst_host_srv_rerror_rate','attack','level'])    

    # Map attacks to integer values
    data_df.columns = columns
    data_labels = data_df.attack.apply(label_attacks)
    data_df['attack'] = data_labels
    

    # Encode categorical features
    categorical_features  = ['protocol_type', 'service', 'flag']
    cat_encoded = pd.get_dummies(data_df[categorical_features])

    # Extract numerical features
    numerical_features = ['duration', 'src_bytes', 'dst_bytes']
    X = cat_encoded.join(data_df[numerical_features])
    Y = data_df['attack']
    return X,Y

def compute_accuracy(predictions, true_labels):
    """
    Computes the accuracy of models predictions with respect to the true labels.
    
    Args:
        predictions: Numpy array of shape [num_examples, 1] of model predictions
        true_labels: Numpy array of shape [num_examples, 1] of true labels
        
    Returns:
        Float: Accuracy of predictions vs. true labels.
    """
    return accuracy_score(predictions, true_labels)
