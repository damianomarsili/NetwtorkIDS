# NewtorkIDS
A series of machine-learning based implementations of an Intrustion Detection System for network traffic. Current completed implementations are:
- Random Forest
- Naive Bayes

The goal of the models is to classify network traffic features into 5 categories:
1. Normal traffic
2. DOS Attack
3. Probe Attack
4. Priviledge Attack
5. Access Attack

The models are trained and evaluated against the [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) benchmark dataset. Currently, the best model is the Random Forest implementation, which achieves an accuracy of upwards of 97.6% on held-out test data.

# Setup
Install dependencies
```
sudo apt update
apt install git python3-dev python3-pip build-essential libagg-dev pkg-config
```
Clone repository
```
git clone https://github.com/damianomarsili/NetwtorkIDS.git
cd NetworkIDS
```
Install requirements
```
python3 -m venv env
. env/bin/activate
pip3 install -r requirements.txt
```
Lastly, [download the NSL-KDD dataset](https://www.unb.ca/cic/datasets/nsl.html) and extract the dataset into the project directory.

# Program execution
The program can be run in two modes **train** and **predict**. In train, we feed the model labeled instances of the NSL-KDD dataset, and optimize the models parameters to best classify the input network traffic.
