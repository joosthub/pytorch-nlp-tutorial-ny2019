# O'Reilly Artificial Intelligence Conference

## Natural Language Processing with Deep Learning training

#### Delip Rao, Brian McMahan

Please visit [dl4nlp.info](http://dl4nlp.info/en/latest/) for more information about the current training session.


## Docker Instructions

Start from the root directory which contains the Dockerfile,
please run the following commands (replacing LOCALPORT with whatever you'd like)

(data should come with repo, so no need to download fresh)

```
docker build -t dl4nlp .
docker run -p LOCALPORT:8888 -d dl4nlp
```

If running outside of docker, can do the following from root dir (where you can see day_1, day_2, etc):

```
jupyter notebook --notebook-dir=$(pwd)
```

Assuming all necessary things are installed.  Required Python packages are in `requirements.txt`, except for PyTorch.  Follow the installation instructions on the PyTorch website.  Also visit [dl4nlp.info](http://dl4nlp.info/en/latest/) for more information about the current training session and [nlproc.info](http://nlproc.info/) for more resources from us!
