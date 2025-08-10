# Waymo E2E Driving Dataset Nautilus Setup

Edit the fields with cruz-id and respective names

## Start Notebook

Run the following commands to start the notebook pod that attaches to the dataset pvc

```
kubectl -n <ns> create -f notebook.yaml
```

After the pod launches run the following command to attach the jupyterlab instance to localhost

```
kubectl -n <ns> port-forward pod/<pod name> 8888:8888
```


## Start a new PVC and Download Data from Google Cloud

> Only to be run if setting up a new dataset. Ensure to allocate enough storage in the PVC to download the dataset

Once you have access to the waymo dataset through google cloud, authenticate with that account. This only needs to run once when the dataset needs to be downloaded to the pvc

```
gcloud auth login --update-adc --no-launch-browser
```
Once authenticated,

```
gcloud storage cp -r gs://<bucket name> /data/<folder name>
```
