Set up the Repository

### 1. Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/ruichenZz/cs260d-final-project.git
cd cs260d-final-project
```

Clone the AMBER repo into this parent repo and follow the instructions there to set up AMBER pipeline, download data, etc: https://github.com/junyangwang0410/AMBER

### 2. Create a Conda Environment

Run the following commands to create and activate a Conda environment:

```bash
conda create -yn toxic python=3.10
conda activate toxic
cd cs263-final-project
pip install -r requirements.txt
```

### 3. Download the kaggle dataset

Move your kaggle.json file into target directory
```bash
mv kaggle.json /home/ruichenbruin24/.config/kaggle/kaggle.json
chmod 600 /home/ruichenbruin24/.config/kaggle/kaggle.json
```

Then download the dataset into target directory
```bash
mkdir jigsaw-dataset
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification
unzip jigsaw-unintended-bias-in-toxicity-classification.zip
rm jigsaw-unintended-bias-in-toxicity-classification.zip
```