# GenAI Worksop
This GitHub repository contains all the resources, code, and instructions you need to build your very own generative AI model using Google Colab. Given below is a sample output from the generative ai created on this project.

![screenshot](https://github.com/aswin-asokan/GenAI_WS/assets/86108610/a284c762-ca0a-4409-aa89-fdba95a3ff65)

Sample file used in the above example: [pollution.pdf](https://github.com/aswin-asokan/GenAI_WS/files/15391901/pollution.pdf)

## Steps to implement the project by yourself:
### Step 1:
Clone the repository or simply download the Jupyter Notebook file from the repository.
### Step 2:
Visit [Google Colab](https://colab.research.google.com/) to start working on the project.
* In google colab choose File > Open notebook > upload
* upload the Jupyter notebook file
### Step 3:
* Sign upto [Hugging face](https://huggingface.co/) to generate an access token
* After signing up go to Settings > Access Tokens and press on New Token to create a token
* After this find the below line of code in the notebook you have already opened
  
  ``` subprocess.run(["huggingface-cli", "login", "--token", "{{API_TOKEN}}"]) ```

* Copy your Token from Hugging face and replace {{API_TOKEN}} with it
### Step 4:
* Now run the code and upload a file containing text information when asked
* Ask questions related to topics in the file to let the ai to generate answers for it
* Now you have your own generative ai and you can customize it however you like
