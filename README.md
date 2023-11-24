## Optimal and SubOptimal Decision Trees

**Disclaimer: This is a work In Progress**

Implementation of Less Greedy Decision Trees algorithms + DL8.5  and its extensions using rust and a python wrapper.


### Requirements:
- Python 3.8-9

### Installation:

Wheels are available for mac and linux for python 3.8-9. To install them just run:
```pip install pytrees-rs```.

If you want to compile the code yourself, you need to install [Rust](https://www.rust-lang.org/tools/install). Then you can run:
```pip install .``` within the repository folder.


### Usage:

The experiments folder contains the code to reproduce the experiments in the paper. There is also an example file showing how to use the library.

A detailed documentation will be available soon.



### Troubleshooting:
To compile on mac, add the following sections to your ~/.cargo/config (if you don't have this file feel free to create):
```
[target.x86_64-apple-darwin]
rustflags = [
"-C", "link-arg=-undefined",
"-C", "link-arg=dynamic_lookup",
]

[target.aarch64-apple-darwin]
rustflags = [
"-C", "link-arg=-undefined",
"-C", "link-arg=dynamic_lookup",
]
```
