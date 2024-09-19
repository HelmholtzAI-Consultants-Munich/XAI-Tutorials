# Dataset Descriptions

## Acute Myeloid Leukemia (AML) dataset

Dataset is taken from https://doi.org/10.1016/j.isci.2019.100780
To get the data: 
- pull the docker image with ``docker pull schultzelab/aml_classifier:0.1``
- run the docker image interactively with: ``docker run --name aml_it -it schultzelab/aml_classifier:0.1 bash``
- copy the data from image to local machine: ``docker cp aml_it:/data ./ ``
- open data in R and write each dataset to csv
- reduce file size by rounding each expression value to 2 decinmal places as follows

   ```
   data_expression = pd.read_csv("AML_dataset.csv", index_col="sample_id")
   data_expression = data_expression.T.reset_index().rename(columns={"index": "sample_id"}).rename_axis(None, axis=1).round(2)
   data_expression.to_csv("AML_dataset_processed.csv")
   ```