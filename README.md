# U<sup>2</sup>-Net on tinygrad

This repository contains the [tinygrad](https://github.com/tinygrad/tinygrad)-ported version of the [U<sup>2</sup>-Net](https://github.com/xuebinqin/U-2-Net/tree/master) model.

## Setup

```sh
pip3 install -r requirements.txt
```

## Download weights

Download the files into `./weights/` folder:

[foreground segmentation (large)](https://drive.google.com/file/d/1m_Kgs91b21gayc2XLW0ou8yugAIadWVP/view?usp=sharing) save as `u2net_human_seg.pth`
[foreground segmentation (small)](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing)
save as `u2netp`
[portrait model (large)](https://drive.google.com/file/d/1IG3HdpcRiDoWNookbncQjeaPN28t90yW/view?usp=sharing)
save as `u2net_portrait.pth`

## Run inference

```sh
python3 u2net_run.py -i /path/to/image -m selected_model
```

Where `selected_model` is one of:
`seg`: human segmentation (large variant, 176 mb)
`seg_small`: human segmentation (small variant, 4.7 mb)
`portrait`: converts photo to pencil drawing (only large variant available, 176 mb)

## Human segmentation examples

![segmentation examples](./example_data/demo_results.png)

## License

Apache-2.0 license
