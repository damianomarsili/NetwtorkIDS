import numpy as np
import pandas as pd

def label_attacks(attack):
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
        TODO: 
    """
    data_pd = pd.read_csv(filename)
    # add the column labels
    columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent'
    ,'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files'
    ,'num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate'
    ,'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate'
    ,'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate'
    ,'dst_host_srv_rerror_rate','attack','level'])    

    data_pd.columns = columns
    data_labels = data_pd.attack.apply(label_attacks)
    data_pd['attack'] = data_labels

    categorical_features  = ['protocol_type', 'service', 'flag']
    cat_encoded = pd.get_dummies(data_pd[categorical_features])

    numerical_features = ['duration', 'src_bytes', 'dst_bytes']
    data = cat_encoded.join(data_pd[numerical_features])
    return data.to_numpy(np.float32)

def compute_accuracy(true_labels, predictions):
    """
    Computes the accuracy of models predictions with respect to the true labels.
    
    Args:
        true_labels: TODO: determine type of labels
        predictions: TODO: determine type of predictions
    Returns:
        TODO: determine accuracy metric    
    """
    
    pass
