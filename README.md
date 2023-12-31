# Fortumage

A simple Python script which uses [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)
to generate images based off random strings from fortune files. These can then
be used as slideshow in a photoframe (e.g. on a Raspberry Pi).

## Set up

Something along the lines of:
```
virtualenv diffusion
. diffision/bin/activate
pip install --upgrade diffusers transformers scipy torch
sudo apt install git-lfs
sudo apt install fortunes
mkdir ~/fortumages
./fortumage.py -o ~/fortumages -s 30
```

The instructions for using XL are sort of the same but different.

In order to display the results remotely their directory can be NFS-exported and
NFS-mounted on another machine, symlinking to the `latest.png` in the
directory. The `frame.sh` script is a simple way of doing this using `feh`.

## The results

The images produced by this are probably more weird than anything else. However,
if you are looking for a talking point then you probably have one. The prompt
which genrated the image is stored in the PNG metadata.

Some of the more interesting images can be found in the `images` directory. The
prompt text can be extracted using ImageMagick's `identify`, e.g.:
```
identify -verbose image/1687729918.png
```

## TODO

This was the result of an afternoon's work and is highly unpolished. It could be
so much better.
